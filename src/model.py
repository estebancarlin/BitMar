"""
BitMar Model Architecture
BitNet-quantized Vision-Language Episodic Memory Transformer
Combines 1.58-bit quantization, DiNOv2 vision, and Larimar episodic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class EpisodicMemory(nn.Module):
    """Episodic Memory mechanism inspired by Larimar - Optimized for Tiny Edge Deployment"""

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        alpha: float = 0.1,
        direct_writing: bool = True,
        observation_noise_std: float = 1e-6,
        memory_compression: bool = True,
        memory_quantization: bool = True,
        memory_access_threshold: float = 0.1,
        memory_consolidation: bool = True
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.alpha = alpha
        self.direct_writing = direct_writing
        self.observation_noise_std = observation_noise_std
        self.memory_compression = memory_compression
        self.memory_quantization = memory_quantization
        self.memory_access_threshold = memory_access_threshold
        self.memory_consolidation = memory_consolidation

        # Memory storage - optimized for edge deployment
        self.register_buffer('memory', torch.zeros(memory_size, episode_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('memory_importance', torch.ones(memory_size))  # Importance scores for selective forgetting
        
        # Edge-specific buffers
        if memory_consolidation:
            self.register_buffer('consolidated_memory', torch.zeros(memory_size // 2, episode_dim))
            self.register_buffer('consolidation_threshold', torch.tensor(10.0))  # Usage threshold for consolidation

        # Memory access networks with reduced complexity for edge deployment
        if memory_compression:
            # Compressed access with smaller intermediate dimensions
            compressed_dim = episode_dim // 2
            self.query_compress = BitNetLinear(episode_dim, compressed_dim)
            self.key_compress = BitNetLinear(episode_dim, compressed_dim)
            self.value_expand = BitNetLinear(compressed_dim, episode_dim)
            
            self.query_net = BitNetLinear(compressed_dim, compressed_dim)
            self.key_net = BitNetLinear(compressed_dim, compressed_dim)
            self.value_net = BitNetLinear(compressed_dim, compressed_dim)
        else:
            self.query_net = BitNetLinear(episode_dim, episode_dim)
            self.key_net = BitNetLinear(episode_dim, episode_dim)
            self.value_net = BitNetLinear(episode_dim, episode_dim)

        # Selective forgetting mechanism
        self.forgetting_gate = BitNetLinear(episode_dim, 1)
        
        # Fast fact editing mechanism
        self.fact_update_gate = BitNetLinear(episode_dim * 2, episode_dim)
        
        # Memory consolidation step counter
        self.register_buffer('consolidation_step', torch.tensor(0))

    def selective_forgetting(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """Implement selective forgetting based on importance scores"""
        # Forget least important memories below threshold
        forget_mask = importance_scores < self.memory_access_threshold
        
        if forget_mask.any():
            # Gradually reduce importance of forgotten memories
            self.memory_importance[forget_mask] *= 0.9
            
            # Clear memory slots that are completely unimportant
            clear_mask = self.memory_importance < 0.01
            if clear_mask.any():
                self.memory[clear_mask] = 0
                self.memory_age[clear_mask] = 0
                self.memory_usage[clear_mask] = 0
                self.memory_importance[clear_mask] = 1.0  # Reset for reuse
        
        return forget_mask

    def fast_fact_editing(self, episode: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        """Fast fact editing without retraining - directly update memory slots"""
        if update_mask.any():
            # Find memory slots to update based on similarity
            similarities = torch.matmul(episode, self.memory.transpose(0, 1))
            _, most_similar_indices = similarities.max(dim=1)
            
            # Update memory with new facts
            for i, idx in enumerate(most_similar_indices):
                if update_mask[i]:
                    # Combine old and new information
                    old_memory = self.memory[idx]
                    combined = torch.cat([old_memory, episode[i]], dim=0)
                    updated_memory = self.fact_update_gate(combined)
                    
                    # Direct update (no retraining needed)
                    self.memory[idx] = updated_memory
                    self.memory_importance[idx] = min(2.0, self.memory_importance[idx] + 0.2)  # Increase importance
        
        return episode

    def memory_consolidation(self):
        """Consolidate frequently accessed memories for efficiency"""
        if not self.memory_consolidation:
            return
            
        self.consolidation_step += 1
        
        # Consolidate every 100 steps
        if self.consolidation_step % 100 == 0:
            # Find most frequently used memories
            high_usage_mask = self.memory_usage > self.consolidation_threshold
            
            if high_usage_mask.any():
                high_usage_indices = torch.where(high_usage_mask)[0]
                
                # Consolidate up to half of memory slots
                consolidation_slots = min(len(high_usage_indices), self.memory_size // 2)
                if consolidation_slots > 0:
                    # Average highly used memories for consolidation
                    consolidated_indices = high_usage_indices[:consolidation_slots]
                    self.consolidated_memory[:consolidation_slots] = self.memory[consolidated_indices].mean(dim=0, keepdim=True)

    def write_memory(self, episode: torch.Tensor, is_fact_update: bool = False) -> torch.Tensor:
        """Write episode to memory with edge optimizations"""
        batch_size = episode.size(0)

        # Fast fact editing check
        if is_fact_update:
            update_mask = torch.ones(batch_size, dtype=torch.bool, device=episode.device)
            episode = self.fast_fact_editing(episode, update_mask)

        if self.direct_writing:
            # Direct writing: find least recently used slots considering importance
            k = min(batch_size, self.memory_size)
            
            # Weight LRU by inverse importance (prefer to overwrite less important memories)
            weighted_age = self.memory_age / (self.memory_importance + 1e-8)
            _, lru_indices = weighted_age.topk(k, largest=False)

            # Handle batch size larger than memory
            if batch_size > self.memory_size:
                for i in range(0, batch_size, self.memory_size):
                    end_idx = min(i + self.memory_size, batch_size)
                    chunk_size = end_idx - i

                    # Get LRU indices for this chunk
                    _, chunk_lru_indices = weighted_age.topk(chunk_size, largest=False)

                    # Update memory slots
                    self.memory[chunk_lru_indices] = episode[i:end_idx].detach()
                    self.memory_age[chunk_lru_indices] = self.memory_age.max() + 1 + i
                    self.memory_usage[chunk_lru_indices] += 1
                    
                    # Update importance for new memories
                    self.memory_importance[chunk_lru_indices] = torch.clamp(
                        self.memory_importance[chunk_lru_indices] + 0.1, max=2.0
                    )
            else:
                # Normal case
                self.memory[lru_indices] = episode[:k].detach()
                self.memory_age[lru_indices] = self.memory_age.max() + 1
                self.memory_usage[lru_indices] += 1
                
                # Update importance
                self.memory_importance[lru_indices] = torch.clamp(
                    self.memory_importance[lru_indices] + 0.1, max=2.0
                )

        # Trigger consolidation and selective forgetting
        self.memory_consolidation()
        self.selective_forgetting(self.memory_importance)

        return episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using attention mechanism with edge optimizations"""
        batch_size = query.size(0)

        # Validate query dimensions
        if query.size(-1) != self.episode_dim:
            raise ValueError(f"Query dimension {query.size(-1)} doesn't match memory episode_dim {self.episode_dim}")

        # Memory compression for edge deployment
        if self.memory_compression:
            # Compress query and memory for efficient processing
            q_compressed = self.query_compress(query)
            memory_compressed = self.key_compress(self.memory)
            
            # Compute attention on compressed representations
            q = self.query_net(q_compressed)
            k = self.key_net(memory_compressed)
            v = self.value_net(memory_compressed)
            
            # Expand values back to full dimension
            v_expanded = self.value_expand(v)
        else:
            # Standard processing
            q = self.query_net(query)
            k = self.key_net(self.memory)
            v_expanded = self.value_net(self.memory)

        # Attention scores with importance weighting
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.size(-1))
        
        # Weight attention by memory importance for better retrieval
        importance_weights = self.memory_importance.unsqueeze(0).expand(batch_size, -1)
        attention_scores = attention_scores * importance_weights
        
        # Apply access threshold for sparse attention (edge optimization)
        if self.memory_access_threshold > 0:
            attention_mask = attention_scores < self.memory_access_threshold
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted memory retrieval with importance consideration
        retrieved = torch.matmul(attention_weights, v_expanded)

        # Update memory access statistics with importance decay
        access_counts = attention_weights.sum(0)
        self.memory_usage += access_counts.detach()
        
        # Boost importance of accessed memories (reinforcement learning principle)
        accessed_mask = access_counts > 0.01
        self.memory_importance[accessed_mask] = torch.clamp(
            self.memory_importance[accessed_mask] + 0.05, max=2.0
        )

        # Memory quantization for edge deployment
        if self.memory_quantization and not self.training:
            retrieved = self.quantize_for_edge(retrieved)

        return retrieved, attention_weights

    def quantize_for_edge(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor for edge deployment efficiency"""
        if not self.memory_quantization:
            return tensor
            
        # Simple 8-bit quantization for edge deployment
        tensor_min, tensor_max = tensor.min(), tensor.max()
        scale = (tensor_max - tensor_min) / 255.0
        
        if scale < 1e-8:
            return tensor
            
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255)
        dequantized = quantized * scale + tensor_min
        
        return dequantized

    def forward(self, episode: torch.Tensor, mode: str = "read_write", is_fact_update: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through episodic memory with edge optimizations"""
        if mode == "write":
            return self.write_memory(episode, is_fact_update), None
        elif mode == "read":
            return self.read_memory(episode)
        else:  # read_write
            # Write episode to memory
            self.write_memory(episode, is_fact_update)
            # Read from memory
            retrieved, attention_weights = self.read_memory(episode)
            return retrieved, attention_weights

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for monitoring"""
        return {
            'memory_utilization': (self.memory_usage > 0).float().mean().item(),
            'average_importance': self.memory_importance.mean().item(),
            'memory_diversity': torch.var(self.memory.flatten()).item(),
            'consolidation_step': self.consolidation_step.item(),
            'active_slots': (self.memory.norm(dim=1) > 1e-6).sum().item()
        }


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
        self.memory_loss_weight = config.get('memory_loss_weight', 0.05)

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

        # Episodic memory with BitNet quantization and edge optimizations
        self.memory = EpisodicMemory(
            memory_size=config['memory_size'],
            episode_dim=config['episode_dim'],
            alpha=config['memory_alpha'],
            direct_writing=config['direct_writing'],
            memory_compression=config.get('memory_compression', True),
            memory_quantization=config.get('memory_quantization', True),
            memory_access_threshold=config.get('memory_access_threshold', 0.1),
            memory_consolidation=config.get('memory_consolidation', True)
        )

        # Additional BitNet projection layers
        self.text_to_episode = BitNetLinear(
            config['text_encoder_dim'],
            config['episode_dim']
        )
        
        self.vision_to_episode = BitNetLinear(
            config['vision_latent_size'],
            config['episode_dim']
        )
        
        self.memory_to_decoder = BitNetLinear(
            config['episode_dim'],
            config['fusion_hidden_size']
        )

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

    def create_episode(
        self,
        text_features: torch.Tensor,
        vision_latent: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Create multimodal episode for memory storage"""
        # Pool text features (mean pooling)
        # [batch_size, text_encoder_dim]
        text_pooled = text_features.mean(dim=1)

        # Project both text and vision to episode dimension
        text_projected = self.text_to_episode(text_pooled)
        vision_projected = self.vision_to_episode(vision_latent)

        # Combine text and vision features (both now have episode_dim)
        episode = text_projected + vision_projected

        return episode

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

    def compute_memory_consistency_loss(
        self,
        episode: torch.Tensor,
        retrieved_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute memory consistency loss to encourage meaningful memory usage
        """
        # L2 regularization on memory difference
        memory_diff = episode - retrieved_memory
        return torch.mean(torch.norm(memory_diff, dim=-1))

    def compute_balanced_loss(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor] = None,
        memory_loss: Optional[torch.Tensor] = None,
        step: int = 0,
        adaptive_controller=None  # NEW: Adaptive training controller
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced multi-objective loss with adaptive scaling
        """
        losses = {'decoder_loss': decoder_loss, 'cross_modal_loss': cross_modal_loss}

        if vision_loss is not None:
            losses['vision_loss'] = vision_loss
        if memory_loss is not None:
            losses['memory_loss'] = memory_loss

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
        if memory_loss is not None:
            total_loss += self.memory_loss_weight * memory_loss

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

        # Create episodes (different handling for vision vs text-only)
        episode = self.create_episode_mixed(
            text_features, vision_latent_masked, cross_attention, has_vision
        )

        # Episodic memory interaction
        if mode == "train":
            retrieved_memory, memory_attention = self.memory(episode, mode="read_write")
        else:
            retrieved_memory, memory_attention = self.memory(episode, mode="read")

        # Prepare decoder input
        memory_context = self.memory_to_decoder(retrieved_memory)
        memory_context_expanded = memory_context.unsqueeze(1).expand(-1, seq_len, -1)
        fused_with_memory = fused_features + memory_context_expanded
        decoder_input = self.decoder_input_proj(fused_with_memory)

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

            memory_loss = None
            if self.config.get('use_memory_consistency_loss', True):
                memory_loss = self.compute_memory_consistency_loss(episode, retrieved_memory)

            # Compute balanced loss with adaptive controller support
            loss_dict = self.compute_balanced_loss_mixed(
                decoder_loss, cross_modal_loss, vision_loss, memory_loss, 
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
            'episode': episode,
            'retrieved_memory': retrieved_memory,
            'cross_attention': cross_attention,
            'memory_attention': memory_attention,
            'text_attention': text_attention,
            'decoder_attention': decoder_outputs.get('attention_patterns', None),
            'memory_usage': self.memory.memory_usage.clone(),
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

    def create_episode_mixed(
        self,
        text_features: torch.Tensor,
        vision_latent: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor],
        has_vision: torch.Tensor
    ) -> torch.Tensor:
        """Create episodes with different handling for vision vs text-only samples"""
        batch_size = text_features.size(0)
        
        # Pool text features
        text_pooled = text_features.mean(dim=1)  # [batch_size, text_dim]
        
        # Project to episode dimension
        text_episode = self.text_to_episode(text_pooled)
        vision_episode = self.vision_to_episode(vision_latent)
        
        # For text-only samples, use only text features
        # For multimodal samples, combine text and vision
        episode = torch.zeros_like(text_episode)
        
        # Text-only samples (has_vision == False)
        text_only_mask = ~has_vision
        if text_only_mask.any():
            episode[text_only_mask] = text_episode[text_only_mask]
        
        # Multimodal samples (has_vision == True)
        multimodal_mask = has_vision
        if multimodal_mask.any():
            # Combine text and vision for multimodal samples
            combined = text_episode[multimodal_mask] + vision_episode[multimodal_mask]
            episode[multimodal_mask] = combined
            
        return episode

    def compute_balanced_loss_mixed(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor],
        memory_loss: Optional[torch.Tensor],
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
            
        if memory_loss is not None:
            total_loss = total_loss + memory_weight * memory_loss
        
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
        if memory_loss is not None:
            loss_dict['memory_loss'] = memory_loss
            
        return loss_dict

    def fast_fact_edit(self, fact_text: str, new_value: str, vision_features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Fast fact editing without retraining - directly update episodic memory
        
        Args:
            fact_text: The fact to be updated (tokenized internally)
            new_value: The new value for the fact (tokenized internally) 
            vision_features: Optional vision features if the fact has visual component
            
        Returns:
            Dict with editing success metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize the fact and new value
            fact_tokens = self.tokenizer.encode(fact_text, return_tensors='pt', max_length=256, truncation=True)
            new_tokens = self.tokenizer.encode(new_value, return_tensors='pt', max_length=256, truncation=True)
            
            # Create attention mask
            fact_mask = torch.ones_like(fact_tokens)
            new_mask = torch.ones_like(new_tokens)
            
            # Handle vision features
            if vision_features is None:
                # Create dummy vision features for text-only facts
                vision_features = torch.zeros(1, self.config['vision_encoder_dim'], device=fact_tokens.device)
                has_vision = torch.tensor([False], device=fact_tokens.device)
            else:
                vision_features = vision_features.unsqueeze(0) if vision_features.dim() == 1 else vision_features
                has_vision = torch.tensor([True], device=fact_tokens.device)
            
            # Encode original fact
            original_result = self.forward(
                input_ids=fact_tokens,
                attention_mask=fact_mask,
                vision_features=vision_features,
                mode="eval",
                has_vision=has_vision
            )
            
            # Create episode for the new fact
            new_result = self.forward(
                input_ids=new_tokens,
                attention_mask=new_mask,
                vision_features=vision_features,
                mode="eval",
                has_vision=has_vision
            )
            
            # Perform fast fact editing in episodic memory
            edit_episode = new_result['episode']
            updated_memory, _ = self.memory(edit_episode, mode="write", is_fact_update=True)
            
            # Verify the edit by retrieving and checking similarity
            retrieved_memory, attention_weights = self.memory(edit_episode, mode="read")
            
            # Compute editing success metrics
            similarity_score = F.cosine_similarity(edit_episode, retrieved_memory, dim=1).mean().item()
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1).mean().item()
            
            return {
                'editing_success': similarity_score > 0.7,  # Threshold for successful edit
                'similarity_score': similarity_score,
                'attention_entropy': attention_entropy,
                'memory_utilization': self.memory.get_memory_stats()['memory_utilization']
            }

    def selective_forget(self, forget_texts: List[str], forget_threshold: float = 0.5) -> Dict[str, float]:
        """
        Selective forgetting - reduce importance of specified facts in episodic memory
        
        Args:
            forget_texts: List of texts/facts to selectively forget
            forget_threshold: Threshold below which memories are considered forgotten
            
        Returns:
            Dict with forgetting success metrics
        """
        self.eval()
        
        forgotten_count = 0
        total_processed = 0
        
        with torch.no_grad():
            for text in forget_texts:
                # Tokenize the text to forget
                tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=256, truncation=True)
                mask = torch.ones_like(tokens)
                
                # Create dummy vision features
                vision_features = torch.zeros(1, self.config['vision_encoder_dim'], device=tokens.device)
                has_vision = torch.tensor([False], device=tokens.device)
                
                # Get memory representation of the text
                result = self.forward(
                    input_ids=tokens,
                    attention_mask=mask,
                    vision_features=vision_features,
                    mode="eval",
                    has_vision=has_vision
                )
                
                episode = result['episode']
                
                # Find most similar memory slots
                similarities = torch.matmul(episode, self.memory.memory.transpose(0, 1))
                most_similar_indices = similarities.argmax(dim=1)
                
                # Reduce importance of similar memory slots
                for idx in most_similar_indices:
                    if similarities[0, idx] > forget_threshold:
                        self.memory.memory_importance[idx] *= 0.3  # Reduce importance significantly
                        if self.memory.memory_importance[idx] < 0.1:
                            # Mark for clearing
                            self.memory.memory[idx] = 0
                            self.memory.memory_age[idx] = 0
                            self.memory.memory_usage[idx] = 0
                            self.memory.memory_importance[idx] = 1.0
                            forgotten_count += 1
                        
                total_processed += 1
        
        return {
            'forgotten_count': forgotten_count,
            'total_processed': total_processed,
            'forgetting_ratio': forgotten_count / max(total_processed, 1),
            'memory_stats': self.memory.get_memory_stats()
        }

    def get_updated_knowledge_accuracy(self, test_facts: List[Tuple[str, str]], vision_features_list: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Test accuracy on updated knowledge - verify that edited facts are correctly retrieved
        
        Args:
            test_facts: List of (question, expected_answer) tuples
            vision_features_list: Optional list of vision features for each fact
            
        Returns:
            Dict with accuracy metrics
        """
        self.eval()
        
        correct_answers = 0
        total_questions = len(test_facts)
        
        with torch.no_grad():
            for i, (question, expected_answer) in enumerate(test_facts):
                # Tokenize question
                question_tokens = self.tokenizer.encode(question, return_tensors='pt', max_length=256, truncation=True)
                question_mask = torch.ones_like(question_tokens)
                
                # Handle vision features
                if vision_features_list and i < len(vision_features_list):
                    vision_features = vision_features_list[i].unsqueeze(0) if vision_features_list[i].dim() == 1 else vision_features_list[i]
                    has_vision = torch.tensor([True], device=question_tokens.device)
                else:
                    vision_features = torch.zeros(1, self.config['vision_encoder_dim'], device=question_tokens.device)
                    has_vision = torch.tensor([False], device=question_tokens.device)
                
                # Get model response through episodic memory
                result = self.forward(
                    input_ids=question_tokens,
                    attention_mask=question_mask,
                    vision_features=vision_features,
                    mode="eval",
                    has_vision=has_vision
                )
                
                # Generate answer (simplified - in practice would use beam search or sampling)
                logits = result['logits']
                predicted_tokens = torch.argmax(logits, dim=-1)
                predicted_answer = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
                
                # Simple string matching for accuracy (could be improved with semantic similarity)
                if expected_answer.lower() in predicted_answer.lower():
                    correct_answers += 1
        
        accuracy = correct_answers / max(total_questions, 1)
        
        return {
            'accuracy': accuracy,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'memory_stats': self.memory.get_memory_stats()
        }


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
