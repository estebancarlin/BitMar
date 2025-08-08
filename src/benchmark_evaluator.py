"""
Comprehensive Benchmark Evaluator for BitMar
Integrates with tinyBenchmarks (https://github.com/felipemaiapolo/tinyBenchmarks)
and WildChat for TinyLLM evaluation, specialized for tiny models with episodic memory assessment
"""

import torch
import json
import time
import requests
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from tqdm import tqdm
import random
import re
from collections import defaultdict
import pandas as pd

# Try to import tinyBenchmarks
try:
    import tinyBenchmarks as tb
    TINY_BENCHMARKS_AVAILABLE = True
except ImportError:
    TINY_BENCHMARKS_AVAILABLE = False
    tb = None

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    """Comprehensive benchmark evaluation for tiny models with episodic memory"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 tokenizer,
                 device: torch.device,
                 config: Dict,
                 save_dir: str = "./benchmark_results"):
        """Initialize benchmark evaluator"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config.get('evaluation', {}).get('benchmark_evaluations', {})
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark configuration
        self.enabled = self.config.get('enabled', False)
        self.max_samples = self.config.get('max_samples_per_task', 1000)
        self.few_shot_examples = self.config.get('few_shot_examples', 5)
        self.use_episodic_analysis = self.config.get('use_episodic_analysis', True)
        
        # Results storage
        self.benchmark_results = defaultdict(dict)
        self.episodic_analysis_results = defaultdict(dict)
        
        logger.info(f"🎯 Benchmark Evaluator initialized")
        logger.info(f"   • Enabled: {self.enabled}")
        logger.info(f"   • Max samples per task: {self.max_samples}")
        logger.info(f"   • Few-shot examples: {self.few_shot_examples}")
        logger.info(f"   • Episodic analysis: {self.use_episodic_analysis}")
        logger.info(f"   • tinyBenchmarks available: {TINY_BENCHMARKS_AVAILABLE}")
        
        # Available tinyBenchmarks from HuggingFace community
        self.tiny_benchmarks = [
            'mmlu',        # tinyMMLU
            'truthfulqa',  # tinyTruthfulQA  
            'gsm8k',       # tinyGSM8K
            'winogrande',  # tinyWinogrande
            'arc',         # tinyARC
            'hellaswag',   # tinyHellaSwag
            'alpaca'       # tinyAlpacaEval
        ]
    
    def evaluate_tiny_benchmarks(self) -> Dict[str, Any]:
        """Evaluate using the official tinyBenchmarks package"""
        if not self.config.get('evaluate_tiny_benchmarks', True):
            return {'skipped': True, 'reason': 'disabled_in_config'}
        
        if not TINY_BENCHMARKS_AVAILABLE:
            logger.warning("⚠️  tinyBenchmarks package not available. Install with: pip install git+https://github.com/felipemaiapolo/tinyBenchmarks")
            return {'error': 'tinyBenchmarks package not available'}
        
        logger.info("📊 Evaluating using official tinyBenchmarks...")
        
        results = {
            'benchmarks': {},
            'overall_scores': {},
            'episodic_memory_analysis': {}
        }
        
        # Evaluate each tiny benchmark
        for benchmark in self.tiny_benchmarks:
            if not self.config.get(f'evaluate_tiny_{benchmark}', True):
                logger.info(f"Skipping {benchmark} (disabled in config)")
                continue
                
            logger.info(f"🔍 Evaluating tiny{benchmark.upper()}...")
            
            try:
                # Load tiny dataset from HuggingFace
                dataset_name = f"tinyBenchmarks/tiny{benchmark.upper()}"
                dataset = load_dataset(dataset_name, split="test")
                
                # Get model predictions
                predictions, episodic_activations = self._get_tiny_benchmark_predictions(
                    dataset, benchmark
                )
                
                # Evaluate using tinyBenchmarks IRT methods
                if tb is not None:
                    benchmark_results = tb.evaluate(predictions, benchmark)
                else:
                    # Fallback evaluation when tinyBenchmarks is not available
                    benchmark_results = {benchmark: {'irt': 0.0, 'pirt': 0.0, 'gpirt': 0.0}}
                
                # Store results
                results['benchmarks'][benchmark] = {
                    'dataset_size': len(dataset),
                    'predictions': predictions.tolist(),
                    'irt_score': benchmark_results[benchmark]['irt'],
                    'pirt_score': benchmark_results[benchmark]['pirt'], 
                    'gpirt_score': benchmark_results[benchmark]['gpirt'],
                    'episodic_activations': episodic_activations
                }
                
                # Analyze episodic memory impact if available
                if episodic_activations and self.use_episodic_analysis:
                    episodic_analysis = self._analyze_episodic_memory_for_tiny_benchmark(
                        predictions, episodic_activations, benchmark
                    )
                    results['benchmarks'][benchmark]['episodic_analysis'] = episodic_analysis
                
                logger.info(f"✅ tiny{benchmark.upper()} - IRT: {benchmark_results[benchmark]['irt']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate tiny{benchmark.upper()}: {e}")
                results['benchmarks'][benchmark] = {'error': str(e)}
        
        # Calculate overall performance
        valid_benchmarks = [b for b in results['benchmarks'] if 'irt_score' in results['benchmarks'][b]]
        if valid_benchmarks:
            results['overall_scores']['average_irt'] = np.mean([
                results['benchmarks'][b]['irt_score'] for b in valid_benchmarks
            ])
            results['overall_scores']['average_pirt'] = np.mean([
                results['benchmarks'][b]['pirt_score'] for b in valid_benchmarks  
            ])
            results['overall_scores']['average_gpirt'] = np.mean([
                results['benchmarks'][b]['gpirt_score'] for b in valid_benchmarks
            ])
            results['overall_scores']['benchmarks_completed'] = len(valid_benchmarks)
        
        logger.info(f"✅ tinyBenchmarks evaluation completed")
        return results
    
    def evaluate_wildchat_for_tinyllm(self) -> Dict[str, Any]:
        """Evaluate on WildChat for TinyLLM evaluation benchmark"""
        if not self.config.get('evaluate_wildchat_tinyllm', True):
            return {'skipped': True, 'reason': 'disabled_in_config'}
        
        logger.info("💬 Evaluating WildChat for TinyLLM benchmark...")
        
        try:
            # Load WildChat dataset for TinyLLM evaluation
            # This might be available through specific TinyLLM evaluation suites
            wildchat_data = self._load_wildchat_for_tinyllm()
            
            results = {
                'total_conversations': len(wildchat_data),
                'conversation_quality': {},
                'episodic_memory_utilization': {},
                'knowledge_grounding': {},
                'instruction_following': {}
            }
            
            conversation_scores = []
            knowledge_scores = []
            instruction_scores = []
            episodic_activations = []
            
            for i, conversation in enumerate(tqdm(wildchat_data, desc="WildChat-TinyLLM")):
                # Extract conversation data
                user_input = conversation.get('user_input', '')
                expected_response = conversation.get('expected_response', '')
                conversation_type = conversation.get('type', 'general')
                
                # Generate response with episodic memory tracking
                generated_response, episodic_activation = self._generate_tinyllm_response(user_input)
                
                # Evaluate different aspects for TinyLLM
                conversation_score = self._evaluate_tinyllm_conversation_quality(
                    user_input, generated_response, expected_response
                )
                conversation_scores.append(conversation_score)
                
                # Evaluate knowledge grounding
                if self._is_knowledge_intensive(user_input):
                    knowledge_score = self._evaluate_knowledge_grounding(
                        user_input, generated_response, expected_response
                    )
                    knowledge_scores.append(knowledge_score)
                
                # Evaluate instruction following
                if self._is_instruction_following_task(user_input):
                    instruction_score = self._evaluate_instruction_following(
                        user_input, generated_response, expected_response
                    )
                    instruction_scores.append(instruction_score)
                
                # Track episodic memory usage
                if episodic_activation is not None:
                    episodic_activations.append({
                        'activation': episodic_activation,
                        'conversation_score': conversation_score,
                        'conversation_type': conversation_type,
                        'user_input_length': len(user_input.split()),
                        'response_length': len(generated_response.split())
                    })
            
            # Calculate final metrics
            results['conversation_quality'] = {
                'average_score': np.mean(conversation_scores),
                'std_score': np.std(conversation_scores),
                'score_distribution': self._calculate_score_distribution(conversation_scores)
            }
            
            if knowledge_scores:
                results['knowledge_grounding'] = {
                    'average_score': np.mean(knowledge_scores),
                    'knowledge_tasks_count': len(knowledge_scores)
                }
            
            if instruction_scores:
                results['instruction_following'] = {
                    'average_score': np.mean(instruction_scores),
                    'instruction_tasks_count': len(instruction_scores)
                }
            
            # Analyze episodic memory utilization
            if episodic_activations and self.use_episodic_analysis:
                results['episodic_memory_utilization'] = self._analyze_episodic_for_conversations(
                    episodic_activations
                )
            
            logger.info(f"✅ WildChat-TinyLLM evaluation completed: {results['conversation_quality']['average_score']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ WildChat-TinyLLM evaluation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_wildchat_50m(self) -> Dict[str, Any]:
        """Evaluate on WildChat-50M subset for conversational abilities"""
        if not self.config.get('evaluate_wildchat', True):
            return {'skipped': True, 'reason': 'disabled_in_config'}
        
        logger.info("💬 Evaluating WildChat-50M subset...")
        
        try:
            # Load WildChat dataset (using a manageable subset)
            # Note: You may need to adjust this based on actual dataset availability
            wildchat_data = self._load_wildchat_subset()
            
            results = {
                'total_conversations': len(wildchat_data),
                'response_quality': {},
                'episodic_memory_utilization': {},
                'conversation_coherence': 0.0,
                'knowledge_recall_accuracy': 0.0
            }
            
            quality_scores = []
            episodic_activations = []
            knowledge_recall_scores = []
            
            for i, conversation in enumerate(tqdm(wildchat_data, desc="WildChat")):
                # Extract conversation turns
                user_input = conversation.get('user_input', '')
                expected_response = conversation.get('assistant_response', '')
                
                # Generate response with episodic memory tracking
                generated_response, episodic_activation = self._generate_wildchat_response(user_input)
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(
                    user_input, generated_response, expected_response
                )
                quality_scores.append(quality_score)
                
                # Track episodic memory usage
                if episodic_activation is not None:
                    episodic_activations.append({
                        'activation': episodic_activation,
                        'quality_score': quality_score,
                        'conversation_type': self._classify_conversation_type(user_input)
                    })
                
                # Test knowledge recall if applicable
                if self._is_knowledge_question(user_input):
                    recall_score = self._evaluate_knowledge_recall(
                        user_input, generated_response, expected_response
                    )
                    knowledge_recall_scores.append(recall_score)
            
            # Calculate final metrics
            results['response_quality']['average_score'] = np.mean(quality_scores)
            results['response_quality']['distribution'] = {
                'excellent': sum(1 for s in quality_scores if s >= 0.8),
                'good': sum(1 for s in quality_scores if 0.6 <= s < 0.8),
                'fair': sum(1 for s in quality_scores if 0.4 <= s < 0.6),
                'poor': sum(1 for s in quality_scores if s < 0.4)
            }
            
            if knowledge_recall_scores:
                results['knowledge_recall_accuracy'] = np.mean(knowledge_recall_scores)
            
            # Analyze episodic memory utilization
            if episodic_activations and self.use_episodic_analysis:
                results['episodic_memory_utilization'] = self._analyze_episodic_memory_impact(
                    episodic_activations, 'wildchat'
                )
            
            logger.info(f"✅ WildChat evaluation completed: {results['response_quality']['average_score']:.3f} quality")
            return results
            
        except Exception as e:
            logger.error(f"❌ WildChat evaluation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_episodic_memory_benchmarks(self) -> Dict[str, Any]:
        """Specialized benchmarks for episodic memory evaluation"""
        if not self.config.get('evaluate_episodic_benchmarks', True):
            return {'skipped': True, 'reason': 'disabled_in_config'}
        
        logger.info("🧠 Evaluating episodic memory specialized benchmarks...")
        
        results = {
            'fact_learning_speed': {},
            'knowledge_interference': {},
            'memory_consolidation': {},
            'cross_modal_episodic': {}
        }
        
        try:
            # Test 1: Fact Learning Speed
            results['fact_learning_speed'] = self._test_fact_learning_speed()
            
            # Test 2: Knowledge Interference Resistance
            results['knowledge_interference'] = self._test_knowledge_interference()
            
            # Test 3: Memory Consolidation
            results['memory_consolidation'] = self._test_memory_consolidation()
            
            # Test 4: Cross-modal Episodic Learning
            results['cross_modal_episodic'] = self._test_cross_modal_episodic()
            
            logger.info("✅ Episodic memory benchmarks completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Episodic memory benchmarks failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_benchmark_evaluation(self, step: int, epoch: int, wandb_logger=None) -> Dict[str, Any]:
        """Run all benchmark evaluations"""
        if not self.enabled:
            return {'disabled': True}
        
        logger.info(f"🚀 Running comprehensive benchmark evaluation at step {step}, epoch {epoch}")
        
        results = {
            'step': step,
            'epoch': epoch,
            'timestamp': time.time(),
            'benchmarks': {}
        }
        
        # Run tinyBenchmarks (official package)
        if self.config.get('evaluate_tiny_benchmarks', True):
            results['benchmarks']['tiny_benchmarks'] = self.evaluate_tiny_benchmarks()
        
        # Run WildChat for TinyLLM
        if self.config.get('evaluate_wildchat_tinyllm', True):
            results['benchmarks']['wildchat_tinyllm'] = self.evaluate_wildchat_for_tinyllm()
        
        # Run legacy evaluations if enabled
        if self.config.get('evaluate_tiny_mmlu', False):
            results['benchmarks']['tiny_mmlu'] = self.evaluate_tiny_mmlu()
        
        if self.config.get('evaluate_wildchat', False):
            results['benchmarks']['wildchat_50m'] = self.evaluate_wildchat_50m()
        
        # Run episodic memory benchmarks
        if self.config.get('evaluate_episodic_benchmarks', True):
            results['benchmarks']['episodic_memory'] = self.evaluate_episodic_memory_benchmarks()
        
        # Calculate overall benchmark score
        results['overall_benchmark_score'] = self._calculate_overall_score(results['benchmarks'])
        
        # Log to wandb if available
        if wandb_logger:
            self._log_benchmark_results_to_wandb(results, step, epoch, wandb_logger)
        
        # Save results
        self._save_benchmark_results(results)
        
        # Generate summary report
        summary = self._generate_benchmark_summary(results)
        logger.info(f"📊 Benchmark Summary:\n{summary}")
        
        return results
    
    def _format_mmlu_prompt(self, question: str, choices: List[str]) -> str:
        """Format MMLU question as multiple choice prompt"""
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "\nAnswer:"
    # Helper methods for tinyBenchmarks evaluation
    def _generate_tinybenchmarks_predictions(self, dataset_name: str, dataset) -> List[str]:
        """Generate predictions for tinyBenchmarks dataset"""
        predictions = []
        episodic_activations = []
        
        for i, example in enumerate(tqdm(dataset, desc=f"Generating {dataset_name} predictions")):
            # Extract question and format for model
            if dataset_name == 'tinyMMLU':
                question = example['question']
                choices = example['choices']
                prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            elif dataset_name == 'tinyTruthfulQA':
                question = example['question']
                prompt = f"Question: {question}\nAnswer:"
            elif dataset_name == 'tinyGSM8K':
                question = example['question']
                prompt = f"Question: {question}\nAnswer:"
            elif dataset_name == 'tinyWinogrande':
                sentence = example['sentence']
                option1 = example['option1']
                option2 = example['option2']
                prompt = f"Sentence: {sentence}\nOption 1: {option1}\nOption 2: {option2}\nAnswer:"
            elif dataset_name == 'tinyARC':
                question = example['question']
                choices = example['choices']['text']
                prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            elif dataset_name == 'tinyHellaSwag':
                ctx = example['ctx']
                endings = example['endings']
                prompt = f"Context: {ctx}\nPossible endings: {', '.join(endings)}\nMost likely ending:"
            elif dataset_name == 'tinyAlpacaEval':
                instruction = example['instruction']
                prompt = f"Instruction: {instruction}\nResponse:"
            else:
                # Generic fallback
                prompt = str(example.get('question', example.get('input', str(example))))
            
            # Generate prediction with episodic memory tracking
            prediction, episodic_activation = self._generate_with_episodic_tracking(prompt)
            predictions.append(prediction.strip())
            
            if episodic_activation is not None:
                episodic_activations.append({
                    'dataset': dataset_name,
                    'example_idx': i,
                    'activation': episodic_activation,
                    'prompt_length': len(prompt.split()),
                    'prediction_length': len(prediction.split())
                })
        
        # Store episodic data for analysis
        if episodic_activations and self.use_episodic_analysis:
            self._store_episodic_data(dataset_name, episodic_activations)
        
        return predictions
    
    def _generate_with_episodic_tracking(self, prompt: str) -> Tuple[str, Optional[float]]:
        """Generate prediction while tracking episodic memory activation"""
        if not hasattr(self.model, 'generate'):
            return "Model doesn't support generation", None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate with tracking
            episodic_activation = None
            if hasattr(self.model, 'episodic_memory') and self.use_episodic_analysis:
                # Track episodic memory activation during generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Extract episodic activation if available
                if hasattr(self.model, 'last_episodic_activation'):
                    episodic_activation = float(self.model.last_episodic_activation.mean().cpu())
            else:
                # Standard generation without episodic tracking
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            # Decode prediction
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[0][input_length:] if hasattr(outputs, 'sequences') else outputs[0][input_length:]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return prediction, episodic_activation
            
        except Exception as e:
            logger.warning(f"Generation failed for prompt: {e}")
            return "Generation failed", None
    
    def _analyze_episodic_for_tinybenchmarks(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze episodic memory effectiveness across tinyBenchmarks datasets"""
        if not self.use_episodic_analysis:
            return {'disabled': True}
        
        episodic_analysis = {
            'cross_dataset_correlation': {},
            'activation_patterns': {},
            'performance_correlation': {}
        }
        
        # Collect all episodic data
        all_activations = []
        all_scores = []
        dataset_patterns = {}
        
        for dataset_name, results in dataset_results.items():
            if isinstance(results, dict) and 'episodic_data' in results:
                episodic_data = results['episodic_data']
                dataset_score = results.get('score', 0.0)
                
                activations = [entry['activation'] for entry in episodic_data if entry['activation'] is not None]
                if activations:
                    all_activations.extend(activations)
                    all_scores.extend([dataset_score] * len(activations))
                    
                    dataset_patterns[dataset_name] = {
                        'mean_activation': np.mean(activations),
                        'std_activation': np.std(activations),
                        'score': dataset_score,
                        'activation_range': [np.min(activations), np.max(activations)]
                    }
        
        if all_activations:
            # Calculate cross-dataset correlations
            episodic_analysis['cross_dataset_correlation'] = {
                'activation_score_correlation': np.corrcoef(all_activations, all_scores)[0, 1] if len(all_activations) > 1 else 0.0,
                'total_samples': len(all_activations)
            }
            
            # Analyze activation patterns
            episodic_analysis['activation_patterns'] = {
                'overall_mean': np.mean(all_activations),
                'overall_std': np.std(all_activations),
                'per_dataset': dataset_patterns
            }
            
            # Performance correlation analysis
            if len(dataset_patterns) > 1:
                dataset_means = [patterns['mean_activation'] for patterns in dataset_patterns.values()]
                dataset_scores = [patterns['score'] for patterns in dataset_patterns.values()]
                
                episodic_analysis['performance_correlation'] = {
                    'dataset_level_correlation': np.corrcoef(dataset_means, dataset_scores)[0, 1] if len(dataset_means) > 1 else 0.0,
                    'high_activation_datasets': [name for name, patterns in dataset_patterns.items() if patterns['mean_activation'] > np.mean(dataset_means)],
                    'high_performance_datasets': [name for name, patterns in dataset_patterns.items() if patterns['score'] > np.mean(dataset_scores)]
                }
        
        return episodic_analysis
    
    def _store_episodic_data(self, dataset_name: str, episodic_data: List[Dict[str, Any]]):
        """Store episodic memory data for later analysis"""
        if not hasattr(self, '_episodic_store'):
            self._episodic_store = {}
        
        self._episodic_store[dataset_name] = episodic_data

    def _load_wildchat_for_tinyllm(self) -> List[Dict[str, Any]]:
        """Load WildChat dataset optimized for TinyLLM evaluation"""
        try:
            # Try to load a subset of WildChat optimized for TinyLLM evaluation
            # This could be from a specific subset or preprocessed version
            wildchat_data = []
            
            # For now, create sample conversations for TinyLLM evaluation
            sample_conversations = [
                {
                    'user_input': 'What is the capital of France?',
                    'expected_response': 'The capital of France is Paris.',
                    'type': 'factual_knowledge'
                },
                {
                    'user_input': 'How do you make a paper airplane?',
                    'expected_response': 'To make a paper airplane, fold a piece of paper in half lengthwise...',
                    'type': 'instructional'
                },
                {
                    'user_input': 'Why is the sky blue?',
                    'expected_response': 'The sky appears blue due to Rayleigh scattering...',
                    'type': 'scientific_explanation'
                }
            ] * (self.max_samples // 3)
            
            return sample_conversations[:self.max_samples]
            
        except Exception as e:
            logger.warning(f"Failed to load WildChat for TinyLLM: {e}")
            return []
    
    def _generate_tinyllm_response(self, user_input: str) -> Tuple[str, Optional[float]]:
        """Generate response for TinyLLM conversation evaluation"""
        prompt = f"User: {user_input}\nAssistant:"
        return self._generate_with_episodic_tracking(prompt)
    
    def _evaluate_tinyllm_conversation_quality(self, user_input: str, generated_response: str, expected_response: str) -> float:
        """Evaluate conversation quality for TinyLLM"""
        # Simple heuristic evaluation (in practice, could use more sophisticated metrics)
        if not generated_response or generated_response == "Generation failed":
            return 0.0
        
        # Basic quality metrics
        response_relevance = self._calculate_response_relevance(user_input, generated_response)
        response_coherence = self._calculate_response_coherence(generated_response)
        response_helpfulness = self._calculate_response_helpfulness(user_input, generated_response)
        
        # Weight the different aspects
        quality_score = (response_relevance * 0.4 + response_coherence * 0.3 + response_helpfulness * 0.3)
        return min(1.0, max(0.0, quality_score))
    
    def _is_knowledge_intensive(self, user_input: str) -> bool:
        """Check if the input requires knowledge-intensive reasoning"""
        knowledge_keywords = ['what is', 'who is', 'when did', 'where is', 'how does', 'explain', 'define']
        return any(keyword in user_input.lower() for keyword in knowledge_keywords)
    
    def _is_instruction_following_task(self, user_input: str) -> bool:
        """Check if the input is an instruction-following task"""
        instruction_keywords = ['how to', 'please', 'can you', 'write', 'create', 'make', 'generate']
        return any(keyword in user_input.lower() for keyword in instruction_keywords)
    
    def _evaluate_knowledge_grounding(self, user_input: str, generated_response: str, expected_response: str) -> float:
        """Evaluate knowledge grounding accuracy"""
        # Simple keyword overlap metric (could be improved with semantic similarity)
        if not generated_response or not expected_response:
            return 0.0
        
        generated_words = set(generated_response.lower().split())
        expected_words = set(expected_response.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(generated_words.intersection(expected_words))
        return overlap / len(expected_words)
    
    def _evaluate_instruction_following(self, user_input: str, generated_response: str, expected_response: str) -> float:
        """Evaluate instruction following quality"""
        # Check if response addresses the instruction
        if not generated_response:
            return 0.0
        
        # Basic heuristics for instruction following
        response_length_appropriate = 20 <= len(generated_response.split()) <= 200
        response_not_empty = len(generated_response.strip()) > 0
        response_relevant = self._calculate_response_relevance(user_input, generated_response)
        
        # Combine metrics
        instruction_score = (
            (0.3 if response_length_appropriate else 0.0) +
            (0.2 if response_not_empty else 0.0) +
            (0.5 * response_relevant)
        )
        
        return min(1.0, instruction_score)
    
    def _calculate_response_relevance(self, user_input: str, response: str) -> float:
        """Calculate relevance between user input and response"""
        if not response or not user_input:
            return 0.0
        
        # Simple word overlap metric
        input_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        if not input_words:
            return 0.0
        
        overlap = len(input_words.intersection(response_words))
        return min(1.0, overlap / len(input_words))
    
    def _calculate_response_coherence(self, response: str) -> float:
        """Calculate response coherence"""
        if not response:
            return 0.0
        
        # Basic coherence heuristics
        sentences = response.split('.')
        has_multiple_sentences = len(sentences) > 1
        reasonable_length = 10 <= len(response.split()) <= 100
        no_repetition = len(set(response.split())) / len(response.split()) > 0.5 if response.split() else 0
        
        coherence_score = (
            (0.3 if has_multiple_sentences else 0.2) +
            (0.4 if reasonable_length else 0.0) +
            (0.3 * no_repetition)
        )
        
        return min(1.0, coherence_score)
    
    def _calculate_response_helpfulness(self, user_input: str, response: str) -> float:
        """Calculate response helpfulness"""
        if not response:
            return 0.0
        
        # Basic helpfulness metrics
        provides_information = len(response.split()) > 5
        addresses_question = self._calculate_response_relevance(user_input, response) > 0.3
        not_too_verbose = len(response.split()) < 150
        
        helpfulness_score = (
            (0.4 if provides_information else 0.0) +
            (0.4 if addresses_question else 0.0) +
            (0.2 if not_too_verbose else 0.0)
        )
        
        return min(1.0, helpfulness_score)
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Calculate score distribution statistics"""
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        return {
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75))
        }
    
    def _analyze_episodic_for_conversations(self, episodic_activations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze episodic memory patterns in conversation evaluation"""
        if not episodic_activations:
            return {'no_data': True}
        
        activations = [entry['activation'] for entry in episodic_activations]
        scores = [entry['conversation_score'] for entry in episodic_activations]
        
        analysis = {
            'activation_stats': {
                'mean': np.mean(activations),
                'std': np.std(activations),
                'min': np.min(activations),
                'max': np.max(activations)
            },
            'correlation_with_quality': np.corrcoef(activations, scores)[0, 1] if len(activations) > 1 else 0.0,
            'high_activation_performance': np.mean([entry['conversation_score'] for entry in episodic_activations if entry['activation'] > np.mean(activations)]) if activations else 0.0,
            'low_activation_performance': np.mean([entry['conversation_score'] for entry in episodic_activations if entry['activation'] <= np.mean(activations)]) if activations else 0.0
        }
        
        return analysis
        """Get model prediction while tracking episodic memory activation"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Add dummy vision features for cross-modal model
            batch_size = inputs['input_ids'].size(0)
            vision_features = torch.randn(batch_size, 768, device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    vision_features=vision_features
                )
                
                # Get prediction
                logits = outputs.logits[:, -1, :]  # Last token logits
                predicted_token_id = torch.argmax(logits, dim=-1)
                prediction = self.tokenizer.decode(predicted_token_id[0])
                
                # Extract episodic memory activation if available
                episodic_activation = None
                if hasattr(outputs, 'memory_outputs') and outputs.memory_outputs is not None:
                    episodic_activation = outputs.memory_outputs.mean().item()
                elif hasattr(self.model, 'memory') and hasattr(self.model.memory, 'last_attention_weights'):
                    episodic_activation = self.model.memory.last_attention_weights.mean().item()
            
            return prediction, episodic_activation
            
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return "", None
        finally:
            self.model.train()
    
    def _evaluate_mmlu_answer(self, prediction: str, correct_answer: int, choices: List[str]) -> bool:
        """Evaluate MMLU answer prediction"""
        # Clean and extract answer
        prediction = prediction.strip().upper()
        
        # Try to extract A, B, C, D from prediction
        if len(prediction) == 1 and prediction in 'ABCD':
            predicted_idx = ord(prediction) - ord('A')
        else:
            # Try to find A, B, C, D in the response
            for i, letter in enumerate('ABCD'):
                if letter in prediction:
                    predicted_idx = i
                    break
            else:
                # If no clear answer, guess randomly or return False
                return False
        
        return predicted_idx == correct_answer
    
    def _evaluate_helm_task(self, task_name: str) -> Dict[str, Any]:
        """Evaluate a specific HELM task"""
        try:
            # Load task dataset (this would need to be adapted based on actual dataset availability)
            if task_name == 'boolq':
                dataset = load_dataset("boolq", split="validation")
            elif task_name == 'piqa':
                dataset = load_dataset("piqa", split="validation")
            elif task_name == 'hellaswag':
                dataset = load_dataset("hellaswag", split="validation")
            elif task_name == 'winogrande':
                dataset = load_dataset("winogrande", "winogrande_debiased", split="validation")
            elif task_name == 'arc_easy':
                dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
            elif task_name == 'openbookqa':
                dataset = load_dataset("openbookqa", split="test")
            else:
                return {'error': f'Unknown task: {task_name}'}
            
            # Sample subset
            sample_size = min(self.max_samples, len(dataset))
            indices = random.sample(range(len(dataset)), sample_size)
            task_data = dataset.select(indices)
            
            correct_predictions = 0
            episodic_activations = []
            
            for example in tqdm(task_data, desc=f"HELM-{task_name}"):
                prompt = self._format_helm_prompt(example, task_name)
                prediction, episodic_activation = self._get_model_prediction_with_memory(prompt)
                
                is_correct = self._evaluate_helm_answer(prediction, example, task_name)
                if is_correct:
                    correct_predictions += 1
                
                if episodic_activation is not None:
                    episodic_activations.append({
                        'activation': episodic_activation,
                        'correct': is_correct
                    })
            
            accuracy = correct_predictions / sample_size
            
            result = {
                'accuracy': accuracy,
                'total_samples': sample_size,
                'correct_predictions': correct_predictions
            }
            
            if episodic_activations and self.use_episodic_analysis:
                result['episodic_analysis'] = self._analyze_episodic_memory_impact(
                    episodic_activations, task_name
                )
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _format_helm_prompt(self, example: Dict, task_name: str) -> str:
        """Format HELM task prompt"""
        if task_name == 'boolq':
            return f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer (True/False):"
        elif task_name == 'piqa':
            return f"Goal: {example['goal']}\nSolution 1: {example['sol1']}\nSolution 2: {example['sol2']}\nWhich solution is better (1 or 2):"
        elif task_name == 'hellaswag':
            return f"Context: {example['ctx']}\nEnding:"
        elif task_name == 'winogrande':
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            return f"Sentence: {sentence}\nOption 1: {option1}\nOption 2: {option2}\nAnswer (1 or 2):"
        elif task_name == 'arc_easy':
            question = example['question']
            choices = example['choices']
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices['text']):
                prompt += f"{choices['label'][i]}. {choice}\n"
            prompt += "Answer:"
            return prompt
        elif task_name == 'openbookqa':
            question = example['question_stem']
            choices = example['choices']
            prompt = f"Question: {question}\n"
            for choice in choices['text']:
                prompt += f"{choice}\n"
            prompt += "Answer:"
            return prompt
        else:
            return str(example)
    
    def _evaluate_helm_answer(self, prediction: str, example: Dict, task_name: str) -> bool:
        """Evaluate HELM task answer"""
        prediction = prediction.strip().lower()
        
        if task_name == 'boolq':
            correct = example['answer']
            return ('true' in prediction and correct) or ('false' in prediction and not correct)
        elif task_name == 'piqa':
            correct = example['label']
            return str(correct) in prediction or ('1' in prediction and correct == 0) or ('2' in prediction and correct == 1)
        elif task_name == 'winogrande':
            correct = example['answer']
            return str(correct) in prediction
        elif task_name in ['arc_easy', 'openbookqa']:
            correct_label = example['answerKey'] if 'answerKey' in example else example['answer']
            return correct_label.lower() in prediction.lower()
        else:
            return False
    
    def _load_wildchat_subset(self) -> List[Dict]:
        """Load WildChat subset for evaluation"""
        # This is a placeholder - you would implement actual WildChat data loading
        # For now, create synthetic conversational data
        synthetic_conversations = [
            {
                'user_input': 'What is the capital of France?',
                'assistant_response': 'The capital of France is Paris.',
                'conversation_type': 'factual_question'
            },
            {
                'user_input': 'Explain quantum computing in simple terms.',
                'assistant_response': 'Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.',
                'conversation_type': 'explanation'
            },
            {
                'user_input': 'How do I bake a chocolate cake?',
                'assistant_response': 'To bake a chocolate cake, you\'ll need flour, sugar, cocoa powder, eggs, butter, and baking powder. Mix the dry ingredients, then combine with wet ingredients, and bake at 350°F for about 30 minutes.',
                'conversation_type': 'instruction'
            }
            # Add more synthetic examples or load real WildChat data
        ]
        
        # Repeat to create a larger test set
        return synthetic_conversations * (self.max_samples // len(synthetic_conversations) + 1)[:self.max_samples]
    
    def _generate_wildchat_response(self, user_input: str) -> Tuple[str, Optional[float]]:
        """Generate response to WildChat input"""
        try:
            # Create prompt
            prompt = f"User: {user_input}\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Add vision features
            batch_size = inputs['input_ids'].size(0)
            vision_features = torch.randn(batch_size, 768, device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                # Generate response
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    vision_features=vision_features,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(prompt):].strip()
                
                # Get episodic memory activation
                episodic_activation = None
                if hasattr(self.model, 'memory') and hasattr(self.model.memory, 'last_attention_weights'):
                    episodic_activation = self.model.memory.last_attention_weights.mean().item()
            
            return response, episodic_activation
            
        except Exception as e:
            logger.warning(f"Response generation failed: {e}")
            return "", None
        finally:
            self.model.train()
    
    def _evaluate_response_quality(self, user_input: str, generated: str, expected: str) -> float:
        """Evaluate response quality (simplified metric)"""
        # Simple quality metrics - you could use more sophisticated evaluation
        if not generated:
            return 0.0
        
        # Length appropriateness (not too short, not too long)
        length_score = min(1.0, len(generated.split()) / 20.0)
        length_score = min(length_score, 2.0 - length_score)  # Penalty for too long
        
        # Relevance (simple keyword overlap)
        user_words = set(user_input.lower().split())
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        relevance_score = len(user_words & generated_words) / max(len(user_words), 1)
        similarity_score = len(expected_words & generated_words) / max(len(expected_words), 1)
        
        # Combined score
        return (length_score * 0.3 + relevance_score * 0.3 + similarity_score * 0.4)
    
    def _classify_conversation_type(self, user_input: str) -> str:
        """Classify conversation type"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['what', 'who', 'where', 'when', 'which']):
            return 'factual_question'
        elif any(word in input_lower for word in ['how', 'explain', 'describe']):
            return 'explanation'
        elif any(word in input_lower for word in ['help', 'can you', 'please']):
            return 'assistance'
        else:
            return 'general'
    
    def _is_knowledge_question(self, user_input: str) -> bool:
        """Check if input is a knowledge question"""
        knowledge_indicators = ['what is', 'who is', 'where is', 'when did', 'capital of', 'explain']
        return any(indicator in user_input.lower() for indicator in knowledge_indicators)
    
    def _evaluate_knowledge_recall(self, user_input: str, generated: str, expected: str) -> float:
        """Evaluate knowledge recall accuracy"""
        # Simple keyword-based evaluation
        expected_words = set(expected.lower().split())
        generated_words = set(generated.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words & generated_words)
        return overlap / len(expected_words)
    
    def _analyze_episodic_memory_impact(self, activations: List[Dict], task_type: str) -> Dict[str, Any]:
        """Analyze episodic memory impact on performance"""
        if not activations:
            return {'no_data': True}
        
        # Separate correct and incorrect predictions
        correct_activations = [a['activation'] for a in activations if a['correct']]
        incorrect_activations = [a['activation'] for a in activations if not a['correct']]
        
        analysis = {
            'total_samples': len(activations),
            'correct_samples': len(correct_activations),
            'incorrect_samples': len(incorrect_activations),
            'average_activation': np.mean([a['activation'] for a in activations]),
            'activation_variance': np.var([a['activation'] for a in activations])
        }
        
        if correct_activations and incorrect_activations:
            analysis['correct_avg_activation'] = np.mean(correct_activations)
            analysis['incorrect_avg_activation'] = np.mean(incorrect_activations)
            analysis['activation_difference'] = analysis['correct_avg_activation'] - analysis['incorrect_avg_activation']
            
            # Correlation between activation and correctness
            activations_list = [a['activation'] for a in activations]
            correctness_list = [1 if a['correct'] else 0 for a in activations]
            correlation = np.corrcoef(activations_list, correctness_list)[0, 1]
            analysis['activation_correctness_correlation'] = correlation
        
        return analysis
    
    def _test_fact_learning_speed(self) -> Dict[str, Any]:
        """Test how quickly the model learns new facts"""
        facts = [
            "The Zorbex planet has three purple moons named Alpha, Beta, and Gamma.",
            "Professor Quinzel invented the hyperdrive engine in 2387.",
            "The rare mineral Quantium-X is found only in the Andromeda sector.",
            "The Battle of Stellar Ridge occurred on stardate 4523.7.",
            "Captain Vex discovered the wormhole to dimension Theta-9."
        ]
        
        results = {'facts_tested': len(facts), 'learning_scores': []}
        
        for fact in facts:
            # Present fact to model
            prompt = f"Remember this fact: {fact}\nNow, what did you just learn?"
            response, episodic_activation = self._get_model_prediction_with_memory(prompt)
            
            # Score based on fact recall
            learning_score = self._score_fact_recall(fact, response)
            results['learning_scores'].append({
                'fact': fact,
                'learning_score': learning_score,
                'episodic_activation': episodic_activation
            })
        
        results['average_learning_score'] = np.mean([s['learning_score'] for s in results['learning_scores']])
        return results
    
    def _test_knowledge_interference(self) -> Dict[str, Any]:
        """Test resistance to knowledge interference"""
        # This would test catastrophic forgetting resistance
        return {'test': 'knowledge_interference', 'status': 'placeholder'}
    
    def _test_memory_consolidation(self) -> Dict[str, Any]:
        """Test memory consolidation effectiveness"""
        # This would test how well the model consolidates episodic memories
        return {'test': 'memory_consolidation', 'status': 'placeholder'}
    
    def _test_cross_modal_episodic(self) -> Dict[str, Any]:
        """Test cross-modal episodic learning"""
        # This would test how episodic memory works across text and vision
        return {'test': 'cross_modal_episodic', 'status': 'placeholder'}
    
    def _score_fact_recall(self, original_fact: str, response: str) -> float:
        """Score how well a fact was recalled"""
        original_words = set(original_fact.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(original_words & response_words)
        return overlap / len(original_words) if original_words else 0.0
    
    def _calculate_overall_score(self, benchmarks: Dict[str, Any]) -> float:
        """Calculate overall benchmark score"""
        scores = []
        
        for benchmark_name, results in benchmarks.items():
            if 'error' in results or 'skipped' in results:
                continue
            
            if benchmark_name == 'tiny_mmlu' and 'overall_accuracy' in results:
                scores.append(results['overall_accuracy'])
            elif benchmark_name == 'tiny_helm' and 'average_score' in results:
                scores.append(results['average_score'])
            elif benchmark_name == 'wildchat_50m' and 'response_quality' in results:
                scores.append(results['response_quality']['average_score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        try:
            timestamp = int(time.time())
            results_file = self.save_dir / f"benchmark_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Also save as latest
            latest_file = self.save_dir / "latest_benchmark_results.json"
            with open(latest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📁 Benchmark results saved to {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {e}")
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable benchmark summary"""
        summary_lines = [
            "🎯 BENCHMARK EVALUATION SUMMARY",
            "=" * 50
        ]
        
        benchmarks = results.get('benchmarks', {})
        
        # tinyMMLU results
        if 'tiny_mmlu' in benchmarks and 'overall_accuracy' in benchmarks['tiny_mmlu']:
            mmlu_result = benchmarks['tiny_mmlu']
            summary_lines.extend([
                f"📚 tinyMMLU: {mmlu_result['overall_accuracy']:.3f} accuracy",
                f"   • Samples tested: {mmlu_result['total_samples']}"
            ])
        
        # tinyHELM results
        if 'tiny_helm' in benchmarks and 'average_score' in benchmarks['tiny_helm']:
            helm_result = benchmarks['tiny_helm']
            summary_lines.extend([
                f"⚡ tinyHELM: {helm_result['average_score']:.3f} average score",
                f"   • Tasks completed: {len(helm_result['tasks'])}"
            ])
        
        # WildChat results
        if 'wildchat_50m' in benchmarks and 'response_quality' in benchmarks['wildchat_50m']:
            wildchat_result = benchmarks['wildchat_50m']
            summary_lines.extend([
                f"💬 WildChat: {wildchat_result['response_quality']['average_score']:.3f} quality",
                f"   • Conversations: {wildchat_result['total_conversations']}"
            ])
        
        # Overall score
        overall_score = results.get('overall_benchmark_score', 0.0)
        summary_lines.extend([
            "",
            f"🏆 Overall Benchmark Score: {overall_score:.3f}",
            "=" * 50
        ])
        
        return "\n".join(summary_lines)
    
    def _log_benchmark_results_to_wandb(self, results: Dict, step: int, epoch: int, wandb_logger):
        """Log benchmark evaluation results to wandb"""
        try:
            import wandb
            
            wandb_metrics = {}
            benchmarks = results.get('benchmarks', {})
            
            # tinyBenchmarks (official package) results
            if 'tiny_benchmarks' in benchmarks:
                tb_results = benchmarks['tiny_benchmarks']
                if 'benchmarks' in tb_results:
                    for benchmark, scores in tb_results['benchmarks'].items():
                        if isinstance(scores, dict):
                            for metric, value in scores.items():
                                if isinstance(value, (int, float)) and metric != 'predictions':
                                    wandb_metrics[f"Benchmarks/tinyBenchmarks/{benchmark}/{metric}"] = value
                
                # Overall tinyBenchmarks scores
                if 'overall_scores' in tb_results:
                    for metric, value in tb_results['overall_scores'].items():
                        if isinstance(value, (int, float)):
                            wandb_metrics[f"Benchmarks/tinyBenchmarks/Overall/{metric}"] = value
            
            # WildChat TinyLLM results
            if 'wildchat_tinyllm' in benchmarks:
                wc_results = benchmarks['wildchat_tinyllm']
                for metric, value in wc_results.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"Benchmarks/WildChat_TinyLLM/{metric}"] = value
            
            # Legacy benchmark results
            if 'tiny_mmlu' in benchmarks:
                mmlu_results = benchmarks['tiny_mmlu']
                for metric, value in mmlu_results.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"Benchmarks/Legacy_tinyMMLU/{metric}"] = value
            
            if 'wildchat_50m' in benchmarks:
                wc50_results = benchmarks['wildchat_50m']
                for metric, value in wc50_results.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"Benchmarks/Legacy_WildChat/{metric}"] = value
                    elif isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                wandb_metrics[f"Benchmarks/Legacy_WildChat/{metric}_{sub_metric}"] = sub_value
            
            # Episodic memory benchmark results
            if 'episodic_memory' in benchmarks:
                em_results = benchmarks['episodic_memory']
                for category, metrics in em_results.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                wandb_metrics[f"Benchmarks/Episodic_Memory/{category}/{metric}"] = value
            
            # Overall benchmark score
            if 'overall_benchmark_score' in results:
                wandb_metrics[f"Benchmarks/Overall/benchmark_score"] = results['overall_benchmark_score']
            
            # Metadata
            wandb_metrics[f"Benchmarks/Meta/evaluation_step"] = step
            wandb_metrics[f"Benchmarks/Meta/evaluation_epoch"] = epoch
            
            # Log all metrics
            if wandb_metrics:
                wandb.log(wandb_metrics)
                logger.info(f"✅ Logged {len(wandb_metrics)} benchmark metrics to wandb for step {step}")
            else:
                logger.warning(f"⚠️  No benchmark metrics found to log for step {step}")
                
        except Exception as e:
            logger.error(f"Failed to log benchmark results to wandb: {e}")
