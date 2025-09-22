"""
BitMar Model Architecture
BitNet-quantized Vision-Language Episodic Memory Transformer
Combines 1.58-bit quantization, DiNOv2 vision, and Larimar episodic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import math
import logging

logger = logging.getLogger(__name__)


class BitNetLinear(nn.Module):
    """1.58-bit Linear layer following BitNet b1.58 architecture"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (full precision for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Quantization scaling factors
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet b1.58 weight quantization: {-1, 0, +1}"""
        # Compute scaling factor with numerical stability
        scale = weight.abs().mean()
        self.weight_scale.data = scale.clamp(min=1e-5, max=1e3)  # Prevent extreme scales

        # Normalize weights with gradient clipping
        weight_norm = torch.clamp(weight / self.weight_scale, min=-10.0, max=10.0)

        # 1.58-bit quantization with threshold
        threshold = 2.0 / 3.0  # Optimal threshold for ternary quantization

        # Create ternary weights
        quantized = torch.zeros_like(weight_norm)
        quantized[weight_norm > threshold] = 1.0
        quantized[weight_norm < -threshold] = -1.0
        # Values between -threshold and threshold remain 0

        return quantized

    def quantize_activations_8bit(self, x: torch.Tensor) -> torch.Tensor:
        """8-bit activation quantization with numerical stability"""
        # Clamp extreme values to prevent overflow
        x_clamped = torch.clamp(x, min=-1e6, max=1e6)

        # Compute quantization parameters
        x_min, x_max = x_clamped.min(), x_clamped.max()

        # Prevent division by zero
        range_val = x_max - x_min
        if range_val < 1e-8:
            return x_clamped

        scale = range_val / 255.0
        self.input_scale.data = scale.clamp(min=1e-8, max=1e3)

        # Quantize to 8-bit
        zero_point = (-x_min / scale).round().clamp(0, 255)
        quantized = ((x_clamped / scale) + zero_point).round().clamp(0, 255)

        # Dequantize
        dequantized = scale * (quantized - zero_point)
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Full precision training with straight-through estimator
            # Forward pass with quantized weights but gradients flow through original weights
            weight_q = self.quantize_weights_1_58_bit(self.weight)
            weight_forward = weight_q * self.weight_scale

            # Use original weight for gradient computation
            weight_forward = weight_forward + \
                (self.weight - self.weight.detach())

            return F.linear(x, weight_forward, self.bias)
        else:
            # Inference with full quantization
            weight_q = self.quantize_weights_1_58_bit(
                self.weight) * self.weight_scale
            x_q = self.quantize_activations_8bit(x)
            return F.linear(x_q, weight_q, self.bias)


class BitNetMLP(nn.Module):
    """BitNet MLP block with 1.58-bit quantization"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = BitNetLinear(dim, hidden_dim)
        self.fc2 = BitNetLinear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class BitNetAttention(nn.Module):
    """Multi-head attention with BitNet quantization"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # BitNet quantized projections
        self.q_proj = BitNetLinear(dim, dim, bias=bias)
        self.k_proj = BitNetLinear(dim, dim, bias=bias)
        self.v_proj = BitNetLinear(dim, dim, bias=bias)
        self.out_proj = BitNetLinear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]

        # Validate input dimensions
        if query.size(-1) != self.dim:
            raise ValueError(f"Query dimension {query.size(-1)} doesn't match expected {self.dim}")
        if key.size(-1) != self.dim:
            raise ValueError(f"Key dimension {key.size(-1)} doesn't match expected {self.dim}")
        if value.size(-1) != self.dim:
            raise ValueError(f"Value dimension {value.size(-1)} doesn't match expected {self.dim}")

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Get key/value sequence length (handle different shapes)
        key_seq_len = key.size(1)
        
        # Reshape for multi-head attention with proper dimension checking
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Handle mask shape: expand to match attention scores shape
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

            # Expand mask to match attention scores shape [batch_size, num_heads, seq_len, key_seq_len]
            if mask.size(-1) != key_seq_len:
                # Adjust mask if needed
                if mask.size(-1) == seq_len:
                    # Pad or trim mask to match key_seq_len
                    if key_seq_len > seq_len:
                        pad_size = key_seq_len - seq_len
                        mask = torch.cat([mask, torch.zeros(*mask.shape[:-1], pad_size, device=mask.device, dtype=mask.dtype)], dim=-1)
                    else:
                        mask = mask[..., :key_seq_len]
            
            mask = mask.expand(batch_size, self.num_heads, seq_len, key_seq_len)
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attended)

        return output, attention_weights.mean(dim=1)  # Average across heads


class BitNetTransformerBlock(nn.Module):
    """BitNet Transformer block with quantized components"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = BitNetAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BitNetMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x, normed_x, normed_x, mask)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class BitNetTextEncoder(nn.Module):
    """BitNet-based text encoder"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings (kept full precision)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + \
            self.position_embedding(positions)
        x = self.dropout(x)

        # Transform through BitNet layers
        attention_patterns = []
        for layer in self.layers:
            # Convert attention mask to the right format for the layer
            layer_mask = None
            if attention_mask is not None:
                # Create a mask where 1 means attend, 0 means don't attend
                layer_mask = attention_mask.unsqueeze(
                    1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

            x, attn_weights = layer(x, layer_mask)
            attention_patterns.append(attn_weights)

        x = self.norm(x)
        return x, attention_patterns


class BitNetTextDecoder(nn.Module):
    """BitNet-based text decoder with causal masking"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Output projection to vocabulary
        self.lm_head = BitNetLinear(dim, vocab_size, bias=False)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        # Register causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)
                       ).unsqueeze(0).unsqueeze(0)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(
                seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + \
                self.position_embedding(positions)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            positions = torch.arange(
                seq_len, device=inputs_embeds.device).unsqueeze(0)
            x = inputs_embeds + self.position_embedding(positions)
        else:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided")

        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if attention_mask is not None:
            # Combine causal mask with padding mask
            mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
        else:
            mask = causal_mask

        # Transform through BitNet layers
        attention_patterns = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_patterns.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'attention_patterns': attention_patterns
        }


# class EpisodicMemory(nn.Module):
#     """Episodic Memory mechanism inspired by Larimar with performance optimizations and external storage support"""

#     def __init__(
#         self,
#         memory_size: int,
#         episode_dim: int,
#         alpha: float = 0.1,
#         direct_writing: bool = True,
#         observation_noise_std: float = 1e-6,
#         external_storage: bool = False,
#         memory_storage_path: str = None,
#         compression_enabled: bool = True,
#         lazy_loading: bool = False
#     ):
#         super().__init__()
#         self.memory_size = memory_size
#         self.episode_dim = episode_dim
#         self.alpha = alpha
#         self.direct_writing = direct_writing
#         self.observation_noise_std = observation_noise_std

#         # External storage configuration
#         self.external_storage = external_storage
#         self.memory_storage_path = memory_storage_path
#         self.compression_enabled = compression_enabled
#         self.lazy_loading = lazy_loading
#         self._memory_loaded = False
#         self._memory_version = 1

#         # Memory storage with improved initialization
#         if external_storage and lazy_loading:
#             # For lazy loading, we'll initialize empty and load when needed
#             self._memory_data = None
#             self._metadata = None
#         else:
#             # Standard initialization for compatibility
#             self.register_buffer('memory', torch.randn(memory_size, episode_dim) * 0.02)
#             self.register_buffer('memory_age', torch.zeros(memory_size))
#             self.register_buffer('memory_usage', torch.zeros(memory_size))

#         # Always initialize these for proper functioning
#         self.register_buffer('memory_quality', torch.zeros(memory_size))
#         self.register_buffer('memory_importance', torch.ones(memory_size))
#         self.register_buffer('memory_mean', torch.zeros(episode_dim))
#         self.register_buffer('memory_std', torch.ones(episode_dim))
#         self.register_buffer('update_count', torch.tensor(0))

#         # Enhanced memory access networks with residual connections
#         self.query_net = nn.Sequential(
#             BitNetLinear(episode_dim, episode_dim),
#             nn.LayerNorm(episode_dim),
#             nn.GELU(),
#             BitNetLinear(episode_dim, episode_dim)
#         )
#         self.key_net = nn.Sequential(
#             BitNetLinear(episode_dim, episode_dim),
#             nn.LayerNorm(episode_dim),
#             nn.GELU(),
#             BitNetLinear(episode_dim, episode_dim)
#         )
#         self.value_net = nn.Sequential(
#             BitNetLinear(episode_dim, episode_dim),
#             nn.LayerNorm(episode_dim),
#             nn.GELU(),
#             BitNetLinear(episode_dim, episode_dim)
#         )

#         # Add temperature parameter for attention sharpening
#         self.register_parameter('attention_temperature', nn.Parameter(torch.tensor(1.0)))

#         # Memory consolidation network for better episode encoding
#         self.consolidation_net = nn.Sequential(
#             BitNetLinear(episode_dim, episode_dim * 2),
#             nn.LayerNorm(episode_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             BitNetLinear(episode_dim * 2, episode_dim),
#             nn.LayerNorm(episode_dim)
#         )

#     def _ensure_memory_loaded(self):
#         """Ensure memory is loaded into device memory"""
#         if self.external_storage and self.lazy_loading and not self._memory_loaded:
#             self.load_external_memory()
#         elif not hasattr(self, 'memory'):
#             # Initialize if not present (compatibility mode)
#             self.register_buffer('memory', torch.randn(self.memory_size, self.episode_dim) * 0.02)
#             self.register_buffer('memory_age', torch.zeros(self.memory_size))
#             self.register_buffer('memory_usage', torch.zeros(self.memory_size))

#     def save_external_memory(self, path: str = None, compress: bool = None) -> str:
#         """Save episodic memory to external storage"""
#         import os
#         import json
#         from pathlib import Path

#         # Use provided path or default
#         save_path = path or self.memory_storage_path or "episodic_memory.pt"
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)

#         # Use provided compression setting or default
#         use_compression = compress if compress is not None else self.compression_enabled

#         # Prepare memory data
#         memory_data = {
#             'memory': self.memory.cpu() if hasattr(self, 'memory') else torch.randn(self.memory_size, self.episode_dim) * 0.02,
#             'memory_age': self.memory_age.cpu() if hasattr(self, 'memory_age') else torch.zeros(self.memory_size),
#             'memory_usage': self.memory_usage.cpu() if hasattr(self, 'memory_usage') else torch.zeros(self.memory_size),
#             'memory_quality': self.memory_quality.cpu(),
#             'memory_importance': self.memory_importance.cpu(),
#             'memory_mean': self.memory_mean.cpu(),
#             'memory_std': self.memory_std.cpu(),
#             'update_count': self.update_count.cpu(),
#             'version': self._memory_version,
#             'metadata': {
#                 'memory_size': self.memory_size,
#                 'episode_dim': self.episode_dim,
#                 'alpha': self.alpha,
#                 'creation_timestamp': torch.tensor(time.time()),
#                 'compression_enabled': use_compression
#             }
#         }

#         # Apply compression if enabled
#         if use_compression:
#             # Quantize memory to reduce storage size
#             memory_data['memory'] = self._compress_memory_tensor(memory_data['memory'])
#             memory_data['compressed'] = True
#         else:
#             memory_data['compressed'] = False

#         # Save to file
#         torch.save(memory_data, save_path)

#         # Also save metadata separately for quick access
#         metadata_path = save_path.with_suffix('.json')
#         with open(metadata_path, 'w') as f:
#             json.dump({
#                 'memory_size': self.memory_size,
#                 'episode_dim': self.episode_dim,
#                 'version': self._memory_version,
#                 'compressed': use_compression,
#                 'file_size_mb': save_path.stat().st_size / (1024 * 1024),
#                 'creation_timestamp': time.time()
#             }, f, indent=2)

#         logger.info(f"💾 Episodic memory saved to: {save_path}")
#         logger.info(f"📊 Memory size: {save_path.stat().st_size / 1024:.1f} KB")

#         return str(save_path)

#     def load_external_memory(self, path: str = None, device: str = None) -> bool:
#         """Load episodic memory from external storage"""
#         import json
#         from pathlib import Path

#         # Use provided path or default
#         load_path = path or self.memory_storage_path or "episodic_memory.pt"
#         load_path = Path(load_path)

#         if not load_path.exists():
#             logger.warning(f"⚠️ External memory file not found: {load_path}")
#             return False

#         try:
#             # Load memory data
#             memory_data = torch.load(load_path, map_location='cpu')

#             # Validate compatibility
#             if memory_data['metadata']['memory_size'] != self.memory_size:
#                 logger.error(f"❌ Memory size mismatch: expected {self.memory_size}, got {memory_data['metadata']['memory_size']}")
#                 return False

#             if memory_data['metadata']['episode_dim'] != self.episode_dim:
#                 logger.error(f"❌ Episode dimension mismatch: expected {self.episode_dim}, got {memory_data['metadata']['episode_dim']}")
#                 return False

#             # Set device
#             device = device or next(self.parameters()).device

#             # Decompress if needed
#             if memory_data.get('compressed', False):
#                 memory_tensor = self._decompress_memory_tensor(memory_data['memory'])
#             else:
#                 memory_tensor = memory_data['memory']

#             # Load memory tensors
#             if hasattr(self, 'memory'):
#                 self.memory.copy_(memory_tensor.to(device))
#                 self.memory_age.copy_(memory_data['memory_age'].to(device))
#                 self.memory_usage.copy_(memory_data['memory_usage'].to(device))
#             else:
#                 # Register buffers if not present (lazy loading case)
#                 self.register_buffer('memory', memory_tensor.to(device))
#                 self.register_buffer('memory_age', memory_data['memory_age'].to(device))
#                 self.register_buffer('memory_usage', memory_data['memory_usage'].to(device))

#             self.memory_quality.copy_(memory_data['memory_quality'].to(device))
#             self.memory_importance.copy_(memory_data['memory_importance'].to(device))
#             self.memory_mean.copy_(memory_data['memory_mean'].to(device))
#             self.memory_std.copy_(memory_data['memory_std'].to(device))
#             self.update_count.copy_(memory_data['update_count'].to(device))

#             self._memory_version = memory_data.get('version', 1)
#             self._memory_loaded = True

#             logger.info(f"✅ Episodic memory loaded from: {load_path}")
#             logger.info(f"📊 Memory version: {self._memory_version}")

#             return True

#         except Exception as e:
#             logger.error(f"❌ Failed to load external memory: {e}")
#             return False

#     def _compress_memory_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
#         """Compress memory tensor for storage"""
#         # Quantize to int8 to reduce storage size
#         tensor_min = tensor.min()
#         tensor_max = tensor.max()

#         # Avoid division by zero
#         tensor_range = tensor_max - tensor_min
#         if tensor_range < 1e-8:
#             return tensor

#         # Quantize to int8 range
#         quantized = ((tensor - tensor_min) / tensor_range * 255).round().clamp(0, 255).to(torch.uint8)

#         # Store quantization parameters
#         return {
#             'data': quantized,
#             'min': tensor_min,
#             'max': tensor_max,
#             'original_shape': tensor.shape
#         }

#     def _decompress_memory_tensor(self, compressed_data) -> torch.Tensor:
#         """Decompress memory tensor"""
#         if isinstance(compressed_data, dict):
#             quantized = compressed_data['data'].float()
#             tensor_min = compressed_data['min']
#             tensor_max = compressed_data['max']

#             # Dequantize
#             tensor_range = tensor_max - tensor_min
#             dequantized = (quantized / 255.0) * tensor_range + tensor_min

#             return dequantized.view(compressed_data['original_shape'])
#         else:
#             # Not compressed, return as-is
#             return compressed_data

#     def _update_memory_statistics(self, episodes: torch.Tensor):
#         """Update running statistics for memory normalization"""
#         with torch.no_grad():
#             batch_mean = episodes.mean(dim=0)
#             batch_var = episodes.var(dim=0, unbiased=False)

#             # Exponential moving average
#             momentum = 0.1
#             self.memory_mean = (1 - momentum) * self.memory_mean + momentum * batch_mean
#             self.memory_std = torch.sqrt((1 - momentum) * self.memory_std**2 + momentum * batch_var)
#             self.update_count += 1

#     def _normalize_episodes(self, episodes: torch.Tensor) -> torch.Tensor:
#         """Normalize episodes using running statistics"""
#         if self.update_count > 10:  # Only normalize after some updates
#             return (episodes - self.memory_mean) / (self.memory_std + 1e-8)
#         return episodes

#     def _compute_episode_quality(self, episode: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
#         """Compute quality score for memory episodes"""
#         # Quality based on diversity and relevance
#         similarity_to_memory = torch.cosine_similarity(
#             episode.unsqueeze(1), self.memory.unsqueeze(0), dim=-1
#         ).max(dim=1)[0]

#         # Encourage diversity - lower similarity = higher quality
#         diversity_score = 1.0 - similarity_to_memory

#         # Relevance score based on retrieval quality
#         retrieval_quality = torch.cosine_similarity(episode, retrieved, dim=-1)

#         # Combined quality score
#         return 0.7 * diversity_score + 0.3 * retrieval_quality

#     def write_memory(self, episode: torch.Tensor) -> torch.Tensor:
#         """Optimized memory writing with intelligent slot selection"""
#         batch_size = episode.size(0)

#         # Apply consolidation to improve episode representation
#         consolidated_episode = self.consolidation_net(episode) + episode  # Residual connection

#         # Update statistics
#         self._update_memory_statistics(consolidated_episode)

#         # Normalize episodes
#         normalized_episode = self._normalize_episodes(consolidated_episode)

#         if self.direct_writing:
#             # Enhanced slot selection combining age, usage, and quality
#             if batch_size <= self.memory_size:
#                 # Compute composite scores for slot selection
#                 age_scores = -self.memory_age  # Prefer older slots
#                 usage_scores = -self.memory_usage  # Prefer less used slots
#                 quality_scores = -self.memory_quality  # Prefer lower quality slots
#                 importance_scores = -self.memory_importance  # Prefer less important slots

#                 # Weighted combination
#                 composite_scores = (
#                     0.4 * age_scores +
#                     0.3 * usage_scores +
#                     0.2 * quality_scores +
#                     0.1 * importance_scores
#                 )

#                 _, best_indices = composite_scores.topk(batch_size, largest=True)

#                 # Update memory slots with momentum-based updates
#                 momentum = self.alpha
#                 self.memory[best_indices] = (
#                     (1 - momentum) * self.memory[best_indices] +
#                     momentum * normalized_episode.detach()
#                 )

#                 # Update metadata
#                 self.memory_age[best_indices] = self.memory_age.max() + 1
#                 self.memory_usage[best_indices] += 1

#                 # Update quality scores (will be computed during read)
#                 with torch.no_grad():
#                     # Temporary quality estimation based on internal consistency
#                     temp_quality = torch.norm(normalized_episode, dim=-1)
#                     self.memory_quality[best_indices] = temp_quality.detach()

#             else:
#                 # Handle large batches efficiently
#                 for i in range(0, batch_size, self.memory_size):
#                     end_idx = min(i + self.memory_size, batch_size)
#                     chunk_size = end_idx - i

#                     # Apply same logic for chunks
#                     age_scores = -self.memory_age
#                     usage_scores = -self.memory_usage
#                     quality_scores = -self.memory_quality
#                     importance_scores = -self.memory_importance

#                     composite_scores = (
#                         0.4 * age_scores +
#                         0.3 * usage_scores +
#                         0.2 * quality_scores +
#                         0.1 * importance_scores
#                     )

#                     _, chunk_indices = composite_scores.topk(chunk_size, largest=True)

#                     momentum = self.alpha
#                     self.memory[chunk_indices] = (
#                         (1 - momentum) * self.memory[chunk_indices] +
#                         momentum * normalized_episode[i:end_idx].detach()
#                     )

#                     self.memory_age[chunk_indices] = self.memory_age.max() + 1 + i
#                     self.memory_usage[chunk_indices] += 1

#                     temp_quality = torch.norm(normalized_episode[i:end_idx], dim=-1)
#                     self.memory_quality[chunk_indices] = temp_quality.detach()

#         return consolidated_episode

#     def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Optimized memory reading with enhanced attention"""
#         batch_size = query.size(0)

#         # Validate query dimensions
#         if query.size(-1) != self.episode_dim:
#             raise ValueError(f"Query dimension {query.size(-1)} doesn't match memory episode_dim {self.episode_dim}")

#         # Normalize query
#         normalized_query = self._normalize_episodes(query)

#         # Enhanced query, key, value computation with residual connections
#         q = self.query_net(normalized_query) + normalized_query  # Residual
#         k = self.key_net(self.memory) + self.memory  # Residual
#         v = self.value_net(self.memory) + self.memory  # Residual

#         # Scaled dot-product attention with learnable temperature
#         attention_scores = torch.matmul(q, k.transpose(0, 1)) / (
#             math.sqrt(self.episode_dim) * self.attention_temperature.clamp(min=0.1, max=10.0)
#         )

#         # Add importance weighting to attention scores
#         importance_weights = self.memory_importance.unsqueeze(0).expand(batch_size, -1)
#         attention_scores = attention_scores + torch.log(importance_weights + 1e-8)

#         # Apply attention with improved stability
#         attention_weights = F.softmax(attention_scores, dim=-1)

#         # Add attention dropout for regularization during training
#         if self.training:
#             attention_weights = F.dropout(attention_weights, p=0.1)

#         # Weighted memory retrieval
#         retrieved = torch.matmul(attention_weights, v)

#         # Update memory access statistics and importance
#         with torch.no_grad():
#             access_counts = attention_weights.sum(0)
#             self.memory_usage += access_counts

#             # Update importance based on usage frequency
#             self.memory_importance = 0.9 * self.memory_importance + 0.1 * (access_counts + 1e-8)

#             # Update quality scores based on retrieval effectiveness
#             if hasattr(self, '_last_query_quality'):
#                 quality_update = self._compute_episode_quality(query, retrieved)
#                 # Update quality for attended slots
#                 attended_indices = attention_weights.max(0)[1]  # Most attended slots
#                 self.memory_quality[attended_indices] = (
#                     0.8 * self.memory_quality[attended_indices] +
#                     0.2 * quality_update.mean()
#                 )

#         return retrieved, attention_weights

#     def forward(self, episode: torch.Tensor, mode: str = "read_write") -> Tuple[torch.Tensor, torch.Tensor]:
#         """Enhanced forward pass with memory consolidation"""
#         if mode == "write":
#             return self.write_memory(episode), None
#         elif mode == "read":
#             return self.read_memory(episode)
#         else:  # read_write
#             # Write episode to memory with consolidation
#             consolidated_episode = self.write_memory(episode)

#             # Read from memory using consolidated episode as query
#             retrieved, attention_weights = self.read_memory(consolidated_episode)

#             # Memory-augmented output combining input and retrieved memory
#             output = 0.7 * consolidated_episode + 0.3 * retrieved

#             return output, attention_weights

#     def get_memory_statistics(self) -> Dict[str, torch.Tensor]:
#         """Get comprehensive memory statistics for monitoring"""
#         return {
#             'memory_usage_distribution': self.memory_usage,
#             'memory_age_distribution': self.memory_age,
#             'memory_quality_scores': self.memory_quality,
#             'memory_importance': self.memory_importance,
#             'attention_temperature': self.attention_temperature,
#             'memory_utilization': (self.memory_usage > 0).float().mean(),
#             'memory_diversity': torch.std(self.memory, dim=0).mean(),
#             'update_count': self.update_count
#         }

#     def consolidate_memory(self):
#         """Explicit memory consolidation for improved organization"""
#         with torch.no_grad():
#             # Sort memory by importance and quality
#             importance_quality_score = 0.6 * self.memory_importance + 0.4 * self.memory_quality
#             sorted_indices = torch.argsort(importance_quality_score, descending=True)

#             # Reorganize memory to group similar episodes
#             sorted_memory = self.memory[sorted_indices]
#             self.memory.copy_(sorted_memory)

#             # Update corresponding metadata
#             self.memory_age[:] = self.memory_age[sorted_indices]
#             self.memory_usage[:] = self.memory_usage[sorted_indices]
#             self.memory_quality[:] = self.memory_quality[sorted_indices]
#             self.memory_importance[:] = self.memory_importance[sorted_indices]

#     def get_memory_info(self) -> Dict:
#         """Get comprehensive memory information"""
#         info = {
#             'memory_size': self.memory_size,
#             'episode_dim': self.episode_dim,
#             'external_storage': self.external_storage,
#             'compression_enabled': self.compression_enabled,
#             'lazy_loading': self.lazy_loading,
#             'memory_loaded': self._memory_loaded if self.external_storage else True,
#             'version': self._memory_version,
#             'storage_path': self.memory_storage_path
#         }

#         if hasattr(self, 'memory'):
#             info.update({
#                 'memory_utilization': (self.memory_usage > 0).float().mean().item(),
#                 'memory_diversity': torch.std(self.memory, dim=0).mean().item(),
#                 'update_count': self.update_count.item(),
#                 'memory_device': str(self.memory.device)
#             })

#         return info

#     def create_memory_snapshot(self, snapshot_name: str = None) -> str:
#         """Create a named snapshot of the current memory state"""
#         import time
#         from pathlib import Path

#         timestamp = int(time.time())
#         snapshot_name = snapshot_name or f"memory_snapshot_{timestamp}"

#         # Create snapshots directory
#         snapshots_dir = Path("memory_snapshots")
#         snapshots_dir.mkdir(exist_ok=True)

#         snapshot_path = snapshots_dir / f"{snapshot_name}.pt"

#         # Save current memory state
#         saved_path = self.save_external_memory(str(snapshot_path), compress=True)

#         logger.info(f"📸 Memory snapshot created: {saved_path}")
#         return saved_path

#     def load_memory_snapshot(self, snapshot_name: str) -> bool:
#         """Load a named memory snapshot"""
#         from pathlib import Path

#         snapshots_dir = Path("memory_snapshots")
#         snapshot_path = snapshots_dir / f"{snapshot_name}.pt"

#         if not snapshot_path.exists():
#             logger.warning(f"⚠️ Snapshot not found: {snapshot_path}")
#             return False

#         success = self.load_external_memory(str(snapshot_path))
#         if success:
#             logger.info(f"📸 Memory snapshot loaded: {snapshot_name}")

#         return success

#     def enable_external_storage(self, storage_path: str = None, compress: bool = True, lazy: bool = False):
#         """Enable external storage mode for edge deployment"""
#         self.external_storage = True
#         self.memory_storage_path = storage_path or "episodic_memory.pt"
#         self.compression_enabled = compress
#         self.lazy_loading = lazy

#         logger.info(f"🔄 External storage enabled: {self.memory_storage_path}")
#         logger.info(f"   Compression: {compress}, Lazy loading: {lazy}")

#     def disable_external_storage(self):
#         """Disable external storage and return to integrated mode"""
#         # Ensure memory is loaded before disabling external storage
#         self._ensure_memory_loaded()

#         self.external_storage = False
#         self.lazy_loading = False
#         self._memory_loaded = True

#         logger.info("🔄 External storage disabled, using integrated mode")


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module for text and vision features"""

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.text_proj = BitNetLinear(text_dim, hidden_dim)
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BitNetAttention(
                dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        batch_size, seq_len = text_features.shape[:2]

        # Validate input dimensions
        if text_features.size(-1) != self.text_dim:
            raise ValueError(f"Text features dimension {text_features.size(-1)} doesn't match expected {self.text_dim}")
        if vision_features.size(-1) != self.vision_dim:
            raise ValueError(f"Vision features dimension {vision_features.size(-1)} doesn't match expected {self.vision_dim}")

        # Project to common dimension
        # [batch_size, seq_len, hidden_dim]
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Cross-attention fusion
        fused = text_proj
        attention_weights = {}

        for i, (attn_layer, norm_layer) in enumerate(zip(self.cross_attention_layers, self.layer_norms)):
            # Text-to-vision cross-attention
            attn_output, attn_weights = attn_layer(
                query=fused,
                key=vision_proj,
                value=vision_proj
            )

            # Residual connection and normalization
            fused = norm_layer(fused + attn_output)
            attention_weights[f'layer_{i}'] = attn_weights

        # Output projection
        output = self.output_proj(fused)

        return output, attention_weights


class VisionEncoder(nn.Module):
    """Quantized Vision Encoder for DiNOv2 features"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2
    ):
        super().__init__()

        # Quantized layers
        self.layers = nn.ModuleList([
            BitNetLinear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, output_dim)

        # Activation and normalization
        self.activation = nn.GELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, input_dim] - DiNOv2 features

        Returns:
            encoded_features: [batch_size, output_dim]
        """
        # Handle potential extra dimensions
        if vision_features.dim() > 2:
            # Flatten any extra dimensions except batch
            original_shape = vision_features.shape
            vision_features = vision_features.view(original_shape[0], -1)

            # Ensure we have the expected input dimension
            if vision_features.size(-1) != self.layers[0].in_features:
                # Take only the first input_dim features if we have more
                if vision_features.size(-1) > self.layers[0].in_features:
                    vision_features = vision_features[:, :self.layers[0].in_features]
                else:
                    raise ValueError(f"Vision features dimension {vision_features.size(-1)} is smaller than expected {self.layers[0].in_features}")

        x = vision_features

        for layer, norm in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Output projection
        output = self.output_proj(x)

        return output


class BitMarModel(nn.Module):
    """
    BitMar: BitNet-quantized Vision-Language Episodic Memory Transformer
    Combines 1.58-bit quantization, DiNOv2 vision features, and Larimar episodic memory
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Loss balancing parameters
        self.cross_modal_loss_weight = config.get('cross_modal_loss_weight', 0.1)
        self.text_loss_weight = config.get('text_loss_weight', 1.0)
        self.vision_loss_weight = config.get('vision_loss_weight', 0.1)
        # self.memory_loss_weight = config.get('memory_loss_weight', 0.05)

        # Dynamic loss scaling
        self.adaptive_loss_scaling = config.get('adaptive_loss_scaling', True)
        self.loss_scale_temperature = config.get('loss_scale_temperature', 0.07)

        # Encoder freezing parameters
        self.freeze_text_encoder_steps = config.get('freeze_text_encoder_steps', 0)
        self.freeze_vision_encoder_steps = config.get('freeze_vision_encoder_steps', 0)
        self.current_step = 0

        # BitNet text encoder/decoder
        self.text_encoder = BitNetTextEncoder(
            vocab_size=config['vocab_size'],
            dim=config['text_encoder_dim'],
            num_layers=config['text_encoder_layers'],
            num_heads=config['text_encoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        self.text_decoder = BitNetTextDecoder(
            vocab_size=config['vocab_size'],
            dim=config['text_decoder_dim'],
            num_layers=config['text_decoder_layers'],
            num_heads=config['text_decoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        # Vision processing with BitNet quantization
        self.vision_encoder = VisionEncoder(
            input_dim=config['vision_encoder_dim'],
            hidden_dim=config['vision_hidden_size'],
            output_dim=config['vision_latent_size']
        )

        # Cross-modal fusion with BitNet
        self.fusion = CrossModalFusion(
            text_dim=config['text_encoder_dim'],
            vision_dim=config['vision_latent_size'],
            hidden_dim=config['fusion_hidden_size'],
            num_heads=config['fusion_num_heads'],
            num_layers=config['fusion_num_layers']
        )

        # # Episodic memory with BitNet quantization
        # self.memory = EpisodicMemory(
        #     memory_size=config['memory_size'],
        #     episode_dim=config['episode_dim'],
        #     alpha=config['memory_alpha'],
        #     direct_writing=config['direct_writing']
        # )

        # # Additional BitNet projection layers
        # self.text_to_episode = BitNetLinear(
        #     config['text_encoder_dim'],
        #     config['episode_dim']
        # )
        
        # self.vision_to_episode = BitNetLinear(
        #     config['vision_latent_size'],
        #     config['episode_dim']
        # )
        
        # self.memory_to_decoder = BitNetLinear(
        #     config['episode_dim'],
        #     config['fusion_hidden_size']
        # )

        # Projection to decoder dimension
        self.decoder_input_proj = BitNetLinear(
            config['fusion_hidden_size'],
            config['text_decoder_dim']
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode text using BitNet encoder"""
        text_features, attention_patterns = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        return text_features, attention_patterns

    def encode_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Encode vision features using quantized vision encoder"""
        vision_latent = self.vision_encoder(
            vision_features)  # [batch_size, vision_latent_size]
        return vision_latent

    # def create_episode(
    #     self,
    #     text_features: torch.Tensor,
    #     vision_latent: torch.Tensor,
    #     attention_weights: Dict[str, torch.Tensor]
    # ) -> torch.Tensor:
    #     """Create multimodal episode for memory storage"""
    #     # Pool text features (mean pooling)
    #     # [batch_size, text_encoder_dim]
    #     text_pooled = text_features.mean(dim=1)

    #     # Project both text and vision to episode dimension
    #     text_projected = self.text_to_episode(text_pooled)
    #     vision_projected = self.vision_to_episode(vision_latent)

    #     # Combine text and vision features (both now have episode_dim)
    #     episode = text_projected + vision_projected

    #     return episode

    def compute_cross_modal_contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss similar to CLIP
        """
        batch_size = text_features.shape[0]

        # Handle dimension mismatch between text and vision features
        text_dim = text_features.shape[-1]
        vision_dim = vision_features.shape[-1]

        if text_dim != vision_dim:
            # Project to smaller dimension to maintain compatibility
            target_dim = min(text_dim, vision_dim)

            if text_dim > vision_dim:
                # Project text features to vision dimension
                text_features = text_features[:, :target_dim]
            else:
                # Project vision features to text dimension
                vision_features = vision_features[:, :target_dim]

        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(text_features, vision_features.T) / temperature

        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss for both directions
        text_to_vision_loss = F.cross_entropy(logits, labels)
        vision_to_text_loss = F.cross_entropy(logits.T, labels)

        return (text_to_vision_loss + vision_to_text_loss) / 2

    def compute_vision_reconstruction_loss(
        self,
        original_vision: torch.Tensor,
        reconstructed_vision: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vision reconstruction loss to prevent vision encoder collapse
        """
        return F.mse_loss(reconstructed_vision, original_vision)

    # def compute_memory_consistency_loss(
    #     self,
    #     # episode: torch.Tensor,
    #     retrieved_memory: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute memory consistency loss to encourage meaningful memory usage
    #     """
    #     # L2 regularization on memory difference
    #     memory_diff = episode - retrieved_memory
    #     return torch.mean(torch.norm(memory_diff, dim=-1))

    def compute_balanced_loss(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor] = None,
        # memory_loss: Optional[torch.Tensor] = None,
        step: int = 0,
        adaptive_controller=None  # NEW: Adaptive training controller
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced multi-objective loss with adaptive scaling
        """
        losses = {'decoder_loss': decoder_loss, 'cross_modal_loss': cross_modal_loss}

        if vision_loss is not None:
            losses['vision_loss'] = vision_loss
        # if memory_loss is not None:
        #     losses['memory_loss'] = memory_loss

        if self.adaptive_loss_scaling:
            # Adaptive scaling based on loss magnitudes
            with torch.no_grad():
                # Compute relative loss scales
                decoder_scale = decoder_loss.detach()
                cross_modal_scale = cross_modal_loss.detach()

                # Prevent division by zero
                if decoder_scale > 1e-8:
                    adaptive_cross_modal_weight = (decoder_scale / cross_modal_scale.clamp(min=1e-8)) * self.cross_modal_loss_weight
                else:
                    adaptive_cross_modal_weight = self.cross_modal_loss_weight

                # Clamp adaptive weights
                adaptive_cross_modal_weight = torch.clamp(adaptive_cross_modal_weight, 0.01, 1.0)
        else:
            adaptive_cross_modal_weight = self.cross_modal_loss_weight

        # Apply loss scheduling (increase cross-modal importance over time)
        cross_modal_schedule = min(1.0, step / 50000)  # Ramp up over 50k steps
        scheduled_cross_modal_weight = adaptive_cross_modal_weight * cross_modal_schedule

        # Compute weighted total loss
        total_loss = (
            self.text_loss_weight * decoder_loss +
            scheduled_cross_modal_weight * cross_modal_loss
        )

        if vision_loss is not None:
            total_loss += self.vision_loss_weight * vision_loss
        # if memory_loss is not None:
        #     total_loss += self.memory_loss_weight * memory_loss

        losses.update({
            'total_loss': total_loss,
            'cross_modal_weight': scheduled_cross_modal_weight,
            'adaptive_weight': adaptive_cross_modal_weight if self.adaptive_loss_scaling else torch.tensor(0.0)
        })

        return losses

    def apply_encoder_freezing(self, step: int):
        """
        Apply temporary encoder freezing based on training step
        """
        self.current_step = step

        # Freeze text encoder if within freezing window
        freeze_text = step < self.freeze_text_encoder_steps
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze_text

        # Freeze vision encoder if within freezing window
        freeze_vision = step < self.freeze_vision_encoder_steps
        for param in self.vision_encoder.parameters():
            param.requires_grad = not freeze_vision

        return {
            'text_encoder_frozen': freeze_text,
            'vision_encoder_frozen': freeze_vision
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "train",
        step: int = 0,
        has_vision: Optional[torch.Tensor] = None,  # NEW: Indicates which samples have real vision
        adaptive_controller=None  # NEW: Adaptive training controller
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BitMar model with mixed vision/text batch support
        
        Args:
            has_vision: Boolean tensor [batch_size] indicating which samples have real vision features
        """
        batch_size, seq_len = input_ids.shape

        # Validate input tensor dimensions early
        expected_vision_dim = self.config['vision_encoder_dim']
        if vision_features.dim() != 2 or vision_features.size(-1) != expected_vision_dim:
            raise ValueError(f"Vision features shape {vision_features.shape} doesn't match expected [batch_size, {expected_vision_dim}]")
        
        if input_ids.size(0) != vision_features.size(0):
            raise ValueError(f"Batch size mismatch: input_ids {input_ids.size(0)} vs vision_features {vision_features.size(0)}")

        # Default has_vision to all True if not provided (backward compatibility)
        if has_vision is None:
            has_vision = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

        # Apply adaptive encoder freezing if controller is provided
        freezing_status = {}
        if mode == "train" and adaptive_controller is not None:
            freezing_status = self.apply_adaptive_encoder_freezing(adaptive_controller)
        elif mode == "train":
            # Fallback to step-based freezing
            freezing_status = self.apply_encoder_freezing(step)

        # Encode text (always available)
        text_features, text_attention = self.encode_text(input_ids, attention_mask)
        
        # Encode vision (with masking for text-only samples)
        vision_latent = self.encode_vision(vision_features)
        
        # Mask vision features for text-only samples
        vision_mask = has_vision.float().unsqueeze(-1)  # [batch_size, 1]
        vision_latent_masked = vision_latent * vision_mask

        # Cross-modal fusion (will handle masked vision features)
        fused_features, cross_attention = self.fusion(text_features, vision_latent_masked)

        # Update adaptive controller with cross-modal similarity if available
        if mode == "train" and adaptive_controller is not None and has_vision.any():
            try:
                # Compute cross-modal similarity for samples with vision
                vision_indices = has_vision.nonzero(as_tuple=True)[0]
                if len(vision_indices) > 0:
                    # Get features for samples with vision
                    text_with_vision = text_features[vision_indices]  # [n_vision, seq_len, dim]
                    vision_with_vision = vision_latent_masked[vision_indices]  # [n_vision, dim]
                    
                    # Pool text features
                    text_pooled = text_with_vision.mean(dim=1)  # [n_vision, dim]
                    
                    # Compute cosine similarity
                    similarity = F.cosine_similarity(text_pooled, vision_with_vision, dim=1)
                    avg_similarity = similarity.mean().item()
                    
                    # Update adaptive controller
                    adaptive_controller.update_similarity(avg_similarity, step)
            except Exception as e:
                logger.warning(f"Failed to update adaptive controller: {e}")

        # # Create episodes (different handling for vision vs text-only)
        # episode = self.create_episode_mixed(
        #     text_features, vision_latent_masked, cross_attention, has_vision
        # )

        # # Episodic memory interaction
        # if mode == "train":
        #     retrieved_memory, memory_attention = self.memory(episode, mode="read_write")
        # else:
        #     retrieved_memory, memory_attention = self.memory(episode, mode="read")

        # # Prepare decoder input
        # memory_context = self.memory_to_decoder(retrieved_memory)
        # memory_context_expanded = memory_context.unsqueeze(1).expand(-1, seq_len, -1)
        # fused_with_memory = fused_features + memory_context_expanded
        # decoder_input = self.decoder_input_proj(fused_with_memory)
        
        # Directly use fused features for decoder (bypass memory)
        decoder_input = self.decoder_input_proj(fused_features)
        

        # Generate text using BitNet decoder
        decoder_outputs = self.text_decoder(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            labels=labels
        )

        # Compute losses if in training mode
        if mode == "train" and labels is not None:
            # Primary decoder loss
            decoder_loss = decoder_outputs['loss']

            # Cross-modal contrastive loss (only for samples with vision)
            cross_modal_loss = torch.tensor(0.0, device=input_ids.device)
            if has_vision.any():
                # Only compute cross-modal loss for samples with vision
                vision_indices = has_vision.nonzero(as_tuple=True)[0]
                if len(vision_indices) > 0:
                    text_pooled = text_features[vision_indices].mean(dim=1)
                    vision_for_loss = vision_latent[vision_indices]
                    cross_modal_loss = self.compute_cross_modal_contrastive_loss(
                        text_pooled, vision_for_loss, temperature=self.loss_scale_temperature
                    )

            # Optional additional losses
            vision_loss = None
            if hasattr(self, 'vision_reconstruction') and self.config.get('use_vision_reconstruction', False):
                if has_vision.any():
                    vision_indices = has_vision.nonzero(as_tuple=True)[0]
                    reconstructed_vision = self.vision_reconstruction(vision_latent[vision_indices])
                    vision_loss = self.compute_vision_reconstruction_loss(
                        vision_features[vision_indices], reconstructed_vision
                    )

            # memory_loss = None
            # if self.config.get('use_memory_consistency_loss', True):
            #     memory_loss = self.compute_memory_consistency_loss(episode, retrieved_memory)

            # Compute balanced loss with adaptive controller support
            loss_dict = self.compute_balanced_loss_mixed(
                decoder_loss, cross_modal_loss, vision_loss, 
                # memory_loss, 
                step, adaptive_controller, has_vision
            )

            final_loss = loss_dict['total_loss']
        else:
            final_loss = decoder_outputs['loss'] if 'loss' in decoder_outputs else None
            loss_dict = {}

        result = {
            'loss': final_loss,
            'logits': decoder_outputs['logits'],
            'text_features': text_features,
            'vision_latent': vision_latent_masked,  # Return masked version
            'fused_features': fused_features,
            # 'episode': episode,
            # 'retrieved_memory': retrieved_memory,
            'cross_attention': cross_attention,
            # 'memory_attention': memory_attention,
            'text_attention': text_attention,
            'decoder_attention': decoder_outputs.get('attention_patterns', None),
            # 'memory_usage': self.memory.memory_usage.clone(),
            'has_vision': has_vision,  # Return for downstream analysis
        }

        # Add loss breakdown and freezing status
        if mode == "train":
            result.update(loss_dict)
            result.update(freezing_status)

        return result

    def apply_adaptive_encoder_freezing(self, adaptive_controller):
        """
        Apply encoder freezing based on adaptive controller decisions
        """
        if adaptive_controller is None:
            return {'text_encoder_frozen': False, 'vision_encoder_frozen': False}
        
        freeze_states = adaptive_controller.get_encoder_freeze_states()
        
        # Freeze/unfreeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze_states['freeze_text_encoder']
        
        # Freeze/unfreeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = not freeze_states['freeze_vision_encoder']
        
        return {
            'text_encoder_frozen': freeze_states['freeze_text_encoder'],
            'vision_encoder_frozen': freeze_states['freeze_vision_encoder']
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """Generate text given input text and vision features"""
        self.eval()

        with torch.no_grad():
            # Encode inputs
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                mode="inference"
            )

            # Start with input sequence
            generated_ids = input_ids.clone()

            for _ in range(max_length - input_ids.size(1)):
                # Get next token logits
                # [batch_size, vocab_size]
                next_logits = outputs['logits'][:, -1, :]

                # Apply temperature
                next_logits = next_logits / temperature

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_token)
                ], dim=1)

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update outputs for next iteration
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    vision_features=vision_features,
                    mode="inference"
                )

        return {
            'generated_ids': generated_ids,
            'generated_text': self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
            'attention_patterns': outputs['cross_attention'],
            'memory_patterns': outputs['memory_attention']
        }

    # def create_episode_mixed(
    #     self,
    #     text_features: torch.Tensor,
    #     vision_latent: torch.Tensor,
    #     attention_weights: Dict[str, torch.Tensor],
    #     has_vision: torch.Tensor
    # ) -> torch.Tensor:
    #     """Create episodes with different handling for vision vs text-only samples"""
    #     batch_size = text_features.size(0)
        
    #     # Pool text features
    #     text_pooled = text_features.mean(dim=1)  # [batch_size, text_dim]
        
    #     # Project to episode dimension
    #     text_episode = self.text_to_episode(text_pooled)
    #     vision_episode = self.vision_to_episode(vision_latent)
        
    #     # For text-only samples, use only text features
    #     # For multimodal samples, combine text and vision
    #     episode = torch.zeros_like(text_episode)
        
    #     # Text-only samples (has_vision == False)
    #     text_only_mask = ~has_vision
    #     if text_only_mask.any():
    #         episode[text_only_mask] = text_episode[text_only_mask]
        
    #     # Multimodal samples (has_vision == True)
    #     multimodal_mask = has_vision
    #     if multimodal_mask.any():
    #         # Combine text and vision for multimodal samples
    #         combined = text_episode[multimodal_mask] + vision_episode[multimodal_mask]
    #         episode[multimodal_mask] = combined
            
    #     return episode

    def compute_balanced_loss_mixed(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor],
        # memory_loss: Optional[torch.Tensor],
        step: int,
        adaptive_controller=None,
        has_vision: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Compute balanced loss for mixed vision/text batches"""
        
        # Base loss weights from config
        text_weight = self.config.get('text_generation_loss_weight', 1.0)
        cross_modal_weight = self.config.get('cross_modal_loss_weight', 1.0)
        memory_weight = self.config.get('memory_regularization_weight', 0.1)
        
        # Adjust cross-modal weight based on vision availability
        if has_vision is not None:
            vision_ratio = has_vision.float().mean().item()
            # Scale cross-modal loss by the proportion of samples with vision
            cross_modal_weight = cross_modal_weight * vision_ratio
        
        # Apply adaptive loss rebalancing if controller is available
        if adaptive_controller is not None:
            controller_info = adaptive_controller.get_loss_multipliers()
            cross_modal_weight *= controller_info.get('cross_modal_weight_multiplier', 1.0)
        
        # Compute total loss
        total_loss = text_weight * decoder_loss
        
        if cross_modal_loss is not None and cross_modal_loss.item() > 0:
            total_loss = total_loss + cross_modal_weight * cross_modal_loss
        
        if vision_loss is not None:
            vision_weight = self.config.get('vision_reconstruction_weight', 0.1)
            total_loss = total_loss + vision_weight * vision_loss
            
        # if memory_loss is not None:
        #     total_loss = total_loss + memory_weight * memory_loss
        
        # Return loss breakdown
        loss_dict = {
            'total_loss': total_loss,
            'decoder_loss': decoder_loss,
            'cross_modal_loss': cross_modal_loss,
            'text_weight': text_weight,
            'cross_modal_weight': cross_modal_weight,
            'memory_weight': memory_weight
        }
        
        if vision_loss is not None:
            loss_dict['vision_loss'] = vision_loss
        # if memory_loss is not None:
        #     loss_dict['memory_loss'] = memory_loss
            
        return loss_dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def create_bitmar_model(config: Dict) -> BitMarModel:
    """Create BitMar model from configuration"""
    model = BitMarModel(config)

    # Print model statistics
    param_count = count_parameters(model)
    logger.info(
        f"BitMar Model created with {param_count['total_parameters']:,} total parameters")
    logger.info(
        f"Trainable parameters: {param_count['trainable_parameters']:,}")

    return model
