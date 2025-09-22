"""
Attention Analysis Module for BitMar
Analyzes attention patterns, head importance, and cross-modal interactions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Comprehensive attention analysis for BitMar model"""

    def __init__(self, model: nn.Module, tokenizer, save_dir: str = "attention_analysis"):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Storage for attention patterns
        self.attention_patterns = defaultdict(list)
        self.head_importance_scores = defaultdict(list)
        self.cross_modal_patterns = []
        self.memory_access_patterns = []

    def extract_attention_patterns(
        self,
        batch: Dict[str, torch.Tensor],
        save_patterns: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from model forward pass"""
        self.model.eval()

        with torch.no_grad():
            # Forward pass with attention tracking
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                vision_features=batch['vision_features']
            )

            # Extract attention patterns
            patterns = {
                'cross_attention': outputs.get('cross_attention', {}),
                # 'memory_attention': outputs.get('memory_attention'),
                'text_latent': outputs.get('text_latent'),
                'vision_latent': outputs.get('vision_latent'),
                # 'episode': outputs.get('episode'),
                # 'memory_usage': outputs.get('memory_usage')
            }

            if save_patterns:
                self._store_patterns(patterns, batch)

            return patterns

    def _store_patterns(self, patterns: Dict, batch: Dict):
        """Store attention patterns for analysis"""
        # Store cross-modal attention
        if patterns['cross_attention']:
            for layer_name, attention in patterns['cross_attention'].items():
                self.attention_patterns[f'cross_modal_{layer_name}'].append(
                    attention.cpu().numpy()
                )

        # # Store memory attention
        # if patterns['memory_attention'] is not None:
        #     self.memory_access_patterns.append(
        #         patterns['memory_attention'].cpu().numpy()
        #     )

        # Store episode information
        self.cross_modal_patterns.append({
            'text_latent': patterns['text_latent'].cpu().numpy(),
            'vision_latent': patterns['vision_latent'].cpu().numpy(),
            # 'episode': patterns['episode'].cpu().numpy(),
            'captions': batch.get('caption', [])
        })

    def analyze_head_importance(
        self,
        dataloader,
        num_batches: int = 50,
        method: str = "gradient"
    ) -> Dict[str, np.ndarray]:
        """
        Analyze importance of attention heads using gradient-based or variance-based methods

        Args:
            dataloader: DataLoader for analysis
            num_batches: Number of batches to analyze
            method: "gradient" or "variance"

        Returns:
            Dictionary of head importance scores
        """
        logger.info(
            f"Analyzing attention head importance using {method} method...")

        head_scores = defaultdict(list)

        if method == "gradient":
            head_scores = self._gradient_based_importance(
                dataloader, num_batches)
        elif method == "variance":
            head_scores = self._variance_based_importance(
                dataloader, num_batches)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Average scores across batches
        avg_scores = {}
        for layer_name, scores_list in head_scores.items():
            avg_scores[layer_name] = np.mean(scores_list, axis=0)

        # Save results
        self._save_head_importance(avg_scores)

        return avg_scores

    def _gradient_based_importance(self, dataloader, num_batches: int) -> Dict[str, List]:
        """Compute head importance based on gradient magnitude"""
        self.model.train()  # Enable gradients
        head_scores = defaultdict(list)

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(
                        next(self.model.parameters()).device)

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Extract gradients from attention heads
            self._extract_gradient_scores(head_scores)

            # Clear gradients
            self.model.zero_grad()

        return head_scores

    def _variance_based_importance(self, dataloader, num_batches: int) -> Dict[str, List]:
        """Compute head importance based on attention variance"""
        self.model.eval()
        head_scores = defaultdict(list)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(
                            next(self.model.parameters()).device)

                # Extract attention patterns
                patterns = self.extract_attention_patterns(
                    batch, save_patterns=False)

                # Compute variance scores
                self._compute_variance_scores(patterns, head_scores)

        return head_scores

    def _extract_gradient_scores(self, head_scores: Dict[str, List]):
        """Extract gradient-based importance scores"""
        # This is a simplified version - in practice, you'd need to hook into
        # specific attention layers to get per-head gradients

        for name, param in self.model.named_parameters():
            if 'attention' in name and param.grad is not None:
                # Compute gradient magnitude as importance score
                grad_magnitude = param.grad.abs().mean().item()
                head_scores[name].append(grad_magnitude)

    def _compute_variance_scores(self, patterns: Dict, head_scores: Dict[str, List]):
        """Compute variance-based importance scores"""
        # Cross-modal attention variance
        if patterns['cross_attention']:
            for layer_name, attention in patterns['cross_attention'].items():
                # Compute attention variance across heads
                variance = torch.var(attention, dim=-1).mean().item()
                head_scores[f'cross_modal_{layer_name}'].append(variance)

        # # Memory attention variance
        # if patterns['memory_attention'] is not None:
        #     variance = torch.var(
        #         patterns['memory_attention'], dim=-1).mean().item()
        #     head_scores['memory_attention'].append(variance)

    def analyze_cross_modal_alignment(self) -> Dict[str, float]:
        """Analyze alignment between text and vision features"""
        if not self.cross_modal_patterns:
            logger.warning("No cross-modal patterns stored for analysis")
            return {}

        alignments = {
            'cosine_similarity': [],
            'euclidean_distance': [],
            'mutual_information': []
        }

        for pattern in self.cross_modal_patterns:
            text_features = pattern['text_latent']
            vision_features = pattern['vision_latent']

            # Pool text features (mean over sequence)
            # [batch_size, feature_dim]
            text_pooled = np.mean(text_features, axis=1)

            # Compute similarities
            for i in range(len(text_pooled)):
                # Cosine similarity
                cos_sim = np.dot(text_pooled[i], vision_features[i]) / (
                    np.linalg.norm(text_pooled[i]) *
                    np.linalg.norm(vision_features[i])
                )
                alignments['cosine_similarity'].append(cos_sim)

                # Euclidean distance
                eucl_dist = np.linalg.norm(text_pooled[i] - vision_features[i])
                alignments['euclidean_distance'].append(eucl_dist)

        # Compute average alignments
        avg_alignments = {
            key: np.mean(values) for key, values in alignments.items()
            if values
        }

        return avg_alignments

    # def analyze_memory_usage(self) -> Dict[str, Union[float, np.ndarray]]:
    #     """Analyze episodic memory usage patterns"""
    #     if not self.memory_access_patterns:
    #         logger.warning("No memory access patterns stored for analysis")
    #         return {}

    #     # Concatenate all memory access patterns
    #     all_patterns = np.concatenate(self.memory_access_patterns, axis=0)

    #     # Compute statistics
    #     memory_stats = {
    #         'mean_access_pattern': np.mean(all_patterns, axis=0),
    #         'std_access_pattern': np.std(all_patterns, axis=0),
    #         'entropy': -np.sum(all_patterns * np.log(all_patterns + 1e-8), axis=-1).mean(),
    #         'max_usage_slot': np.argmax(np.mean(all_patterns, axis=0)),
    #         'usage_uniformity': 1.0 - np.std(np.mean(all_patterns, axis=0)) / np.mean(all_patterns)
    #     }

    #     return memory_stats

    def identify_important_heads(
        self,
        head_scores: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> Dict[str, List[int]]:
        """Identify most important attention heads"""
        important_heads = {}

        for layer_name, scores in head_scores.items():
            if len(scores.shape) > 0:
                # Get top-k head indices
                top_indices = np.argsort(scores)[-top_k:][::-1]
                important_heads[layer_name] = top_indices.tolist()
            else:
                important_heads[layer_name] = [0]  # Single score

        return important_heads

    def visualize_attention_patterns(
        self,
        patterns: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        save_plots: bool = True
    ):
        """Create visualizations of attention patterns"""
        # Cross-modal attention visualization
        if patterns['cross_attention']:
            self._plot_cross_modal_attention(
                patterns['cross_attention'],
                sample_idx,
                save_plots
            )

        # # Memory attention visualization
        # if patterns['memory_attention'] is not None:
        #     self._plot_memory_attention(
        #         patterns['memory_attention'],
        #         sample_idx,
        #         save_plots
        #     )

    def _plot_cross_modal_attention(
        self,
        cross_attention: Dict[str, torch.Tensor],
        sample_idx: int,
        save_plots: bool
    ):
        """Plot cross-modal attention patterns"""
        for layer_name, attention in cross_attention.items():
            # attention shape: [batch_size, num_heads, seq_len, 1] (text to vision)
            # [num_heads, seq_len, 1]
            attn_data = attention[sample_idx].cpu().numpy()

            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                # [1, seq_len] -> [seq_len, num_heads]
                attn_data.squeeze(-1).T,
                cmap='Blues',
                cbar=True,
                xticklabels=[f'Head_{i}' for i in range(attn_data.shape[0])],
                yticklabels=False
            )
            plt.title(f'Cross-Modal Attention - {layer_name}')
            plt.xlabel('Attention Heads')
            plt.ylabel('Text Tokens')

            if save_plots:
                plt.savefig(
                    self.save_dir / f'cross_modal_{layer_name}_sample_{sample_idx}.png')
            plt.close()

    # def _plot_memory_attention(
    #     self,
    #     memory_attention: torch.Tensor,
    #     sample_idx: int,
    #     save_plots: bool
    # ):
    #     """Plot memory attention patterns"""
    #     # memory_attention shape: [batch_size, memory_size]
    #     attn_data = memory_attention[sample_idx].cpu().numpy()

    #     plt.figure(figsize=(15, 4))
    #     plt.bar(range(len(attn_data)), attn_data)
    #     plt.title('Episodic Memory Access Pattern')
    #     plt.xlabel('Memory Slot')
    #     plt.ylabel('Attention Weight')
    #     plt.grid(True, alpha=0.3)

    #     if save_plots:
    #         plt.savefig(self.save_dir /
    #                     f'memory_attention_sample_{sample_idx}.png')
    #     plt.close()

    def _save_head_importance(self, head_scores: Dict[str, np.ndarray]):
        """Save head importance scores to file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_scores = {}
        for layer_name, scores in head_scores.items():
            if isinstance(scores, np.ndarray):
                serializable_scores[layer_name] = scores.tolist()
            else:
                serializable_scores[layer_name] = scores

        # Save to JSON
        with open(self.save_dir / 'head_importance_scores.json', 'w') as f:
            json.dump(serializable_scores, f, indent=2)

        # Save as numpy arrays
        np.savez(
            self.save_dir / 'head_importance_scores.npz',
            **head_scores
        )

        logger.info(f"Head importance scores saved to {self.save_dir}")

    def generate_report(self) -> Dict:
        """Generate comprehensive attention analysis report"""
        logger.info("Generating attention analysis report...")

        report = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'cross_modal_alignment': self.analyze_cross_modal_alignment(),
            # 'memory_usage': self.analyze_memory_usage(),
            'attention_patterns_count': {
                layer: len(patterns) for layer, patterns in self.attention_patterns.items()
            }
        }

        # Save report
        with open(self.save_dir / 'attention_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Attention analysis report saved to {self.save_dir}")

        return report

    def save_top_heads(
        self,
        head_scores: Dict[str, np.ndarray],
        top_k: int = 50,
        output_file: str = "top_attention_heads.npy"
    ):
        """Save top attention heads for use in lo-fit style training"""
        important_heads = self.identify_important_heads(head_scores, top_k)

        # Convert to format similar to lo-fit: [(layer, head), ...]
        top_heads_list = []

        for layer_name, head_indices in important_heads.items():
            for head_idx in head_indices:
                # Extract layer number from layer name if possible
                try:
                    layer_num = int(layer_name.split('_')[-1])
                except:
                    layer_num = 0  # Default layer number

                top_heads_list.append([layer_num, head_idx])

        # Sort by importance (assuming order in important_heads reflects importance)
        top_heads_array = np.array(top_heads_list[:top_k])

        # Save in lo-fit compatible format
        output_path = self.save_dir / output_file
        np.save(output_path, top_heads_array)

        logger.info(f"Top {top_k} attention heads saved to {output_path}")
        logger.info(f"Format: each row is [layer_index, head_index]")

        return top_heads_array


def analyze_model_attention(
    model: nn.Module,
    dataloader,
    tokenizer,
    config: Dict,
    num_analysis_batches: int = 100
) -> AttentionAnalyzer:
    """Complete attention analysis pipeline"""
    logger.info("Starting comprehensive attention analysis...")

    # Create analyzer
    analyzer = AttentionAnalyzer(
        model=model,
        tokenizer=tokenizer,
        save_dir=config.get('attention_dir', 'attention_analysis')
    )

    # Extract attention patterns from multiple batches
    for i, batch in enumerate(dataloader):
        if i >= num_analysis_batches:
            break

        # Move batch to model device
        device = next(model.parameters()).device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        # Extract patterns
        patterns = analyzer.extract_attention_patterns(batch)

        # Visualize first few samples
        if i < 5:
            analyzer.visualize_attention_patterns(patterns, sample_idx=0)

        if i % 20 == 0:
            logger.info(f"Processed {i+1}/{num_analysis_batches} batches")

    # Analyze head importance
    head_scores = analyzer.analyze_head_importance(dataloader, num_batches=50)

    # Save top heads for potential lo-fit style training
    analyzer.save_top_heads(head_scores, top_k=50)

    # Generate comprehensive report
    report = analyzer.generate_report()

    logger.info("Attention analysis completed!")
    logger.info(f"Results saved to: {analyzer.save_dir}")

    return analyzer
