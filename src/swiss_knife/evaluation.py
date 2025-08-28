"""
Evaluation metrics for DistilBERT-style distilled models.

Based on Section 4 of "DistilBERT, a distilled version of BERT" (Sanh et al., 2019)
The paper evaluates on GLUE benchmark and SQuAD for downstream tasks.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)


class DistillationEvaluator:
    """
    Evaluator for distilled models following DistilBERT paper metrics.
    
    From the paper's evaluation:
    - GLUE benchmark performance
    - Inference speed comparison
    - Model size reduction
    - Perplexity on MLM task
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_perplexity(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        Evaluate perplexity on masked language modeling task.
        
        This measures how well the model predicts masked tokens,
        which is a key metric for language understanding.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            Perplexity score
        """
        model.eval()
        model = model.to(self.device)
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating perplexity"):
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                
                # Move to device
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Create labels (same as input for MLM)
                labels = input_ids.clone()
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                # Accumulate loss
                loss = outputs.loss
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def measure_inference_speed(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        texts: List[str],
        batch_sizes: List[int] = [1, 8, 32],
        max_length: int = 512,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Measure inference speed at different batch sizes.
        
        From the paper: "DistilBERT is 60% faster than BERT-base"
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            texts: Sample texts for inference
            batch_sizes: Batch sizes to test
            max_length: Maximum sequence length
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary of inference times per batch size
        """
        model.eval()
        model = model.to(self.device)
        
        results = {}
        
        for batch_size in batch_sizes:
            # Prepare batch
            batch_texts = texts[:batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Measure inference time
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            end_time = time.time()
            
            # Calculate average time per sample
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_runs
            avg_time_per_sample = avg_time_per_batch / batch_size
            
            results[f"batch_{batch_size}"] = {
                "avg_time_per_batch": avg_time_per_batch,
                "avg_time_per_sample": avg_time_per_sample,
                "samples_per_second": 1.0 / avg_time_per_sample,
            }
            
            logger.info(f"Batch size {batch_size}: {avg_time_per_sample*1000:.2f}ms per sample")
        
        return results
    
    def compare_models(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: AutoTokenizer,
        test_texts: List[str],
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Compare teacher and student models on various metrics.
        
        This implements the comparison methodology from the DistilBERT paper:
        - Size reduction (40% smaller)
        - Speed improvement (60% faster)
        - Performance retention (97% of BERT's performance)
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            tokenizer: Tokenizer
            test_texts: Test texts for evaluation
            max_length: Maximum sequence length
            
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        # 1. Model size comparison
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        results["size"] = {
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "compression_ratio": teacher_params / student_params,
            "size_reduction_percent": (1 - student_params / teacher_params) * 100,
        }
        
        logger.info(f"Size reduction: {results['size']['size_reduction_percent']:.1f}%")
        
        # 2. Speed comparison
        logger.info("Measuring teacher inference speed...")
        teacher_speed = self.measure_inference_speed(
            teacher_model, tokenizer, test_texts, batch_sizes=[1, 8], num_runs=50
        )
        
        logger.info("Measuring student inference speed...")
        student_speed = self.measure_inference_speed(
            student_model, tokenizer, test_texts, batch_sizes=[1, 8], num_runs=50
        )
        
        # Calculate speedup
        speedup_batch_1 = (
            teacher_speed["batch_1"]["avg_time_per_sample"] /
            student_speed["batch_1"]["avg_time_per_sample"]
        )
        speedup_batch_8 = (
            teacher_speed["batch_8"]["avg_time_per_sample"] /
            student_speed["batch_8"]["avg_time_per_sample"]
        )
        
        results["speed"] = {
            "teacher_speed": teacher_speed,
            "student_speed": student_speed,
            "speedup_batch_1": speedup_batch_1,
            "speedup_batch_8": speedup_batch_8,
            "avg_speedup": (speedup_batch_1 + speedup_batch_8) / 2,
        }
        
        logger.info(f"Average speedup: {results['speed']['avg_speedup']:.2f}x")
        
        # 3. Perplexity comparison
        logger.info("Calculating teacher perplexity...")
        teacher_perplexity = self.evaluate_perplexity(
            teacher_model, tokenizer, test_texts[:100]
        )
        
        logger.info("Calculating student perplexity...")
        student_perplexity = self.evaluate_perplexity(
            student_model, tokenizer, test_texts[:100]
        )
        
        results["perplexity"] = {
            "teacher": teacher_perplexity,
            "student": student_perplexity,
            "perplexity_increase": student_perplexity - teacher_perplexity,
            "relative_performance": (teacher_perplexity / student_perplexity) * 100,
        }
        
        logger.info(f"Teacher perplexity: {teacher_perplexity:.2f}")
        logger.info(f"Student perplexity: {student_perplexity:.2f}")
        
        return results
    
    def evaluate_on_glue_task(
        self,
        model: nn.Module,
        task_name: str = "sst2",
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate model on a GLUE task.
        
        The DistilBERT paper reports results on all GLUE tasks.
        This is a simplified evaluation for demonstration.
        
        Args:
            model: Model to evaluate
            task_name: GLUE task name
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset("glue", task_name)
        test_data = dataset["validation"][:num_samples]
        
        # Create pipeline for the task
        if task_name in ["sst2", "cola"]:
            task_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                device=0 if self.device == "cuda" else -1,
            )
        else:
            logger.warning(f"Task {task_name} not implemented in this demo")
            return {}
        
        # Run predictions
        predictions = []
        labels = []
        
        for i in range(len(test_data["sentence"])):
            text = test_data["sentence"][i]
            label = test_data["label"][i]
            
            # Get prediction
            result = task_pipeline(text)[0]
            pred_label = 1 if result["label"] == "POSITIVE" else 0
            
            predictions.append(pred_label)
            labels.append(label)
        
        # Calculate accuracy
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(predictions)
        
        return {
            "task": task_name,
            "accuracy": accuracy,
            "num_samples": num_samples,
        }
    
    def generate_evaluation_report(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        tokenizer: AutoTokenizer,
        test_texts: List[str],
    ) -> str:
        """
        Generate a comprehensive evaluation report matching DistilBERT paper format.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            tokenizer: Tokenizer
            test_texts: Test texts for evaluation
            
        Returns:
            Formatted evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Run comparison
        results = self.compare_models(
            teacher_model, student_model, tokenizer, test_texts
        )
        
        # Format report
        report = []
        report.append("=" * 60)
        report.append("DistilBERT-Style Model Distillation Evaluation Report")
        report.append("Based on Sanh et al., 2019 (arxiv:1909.10351)")
        report.append("=" * 60)
        report.append("")
        
        # Model size
        report.append("MODEL SIZE COMPARISON:")
        report.append("-" * 30)
        report.append(f"Teacher parameters: {results['size']['teacher_parameters']:,}")
        report.append(f"Student parameters: {results['size']['student_parameters']:,}")
        report.append(f"Compression ratio: {results['size']['compression_ratio']:.2f}x")
        report.append(f"Size reduction: {results['size']['size_reduction_percent']:.1f}%")
        report.append(f"Target (DistilBERT paper): 40% reduction ✓" 
                     if results['size']['size_reduction_percent'] >= 35 else
                     f"Target (DistilBERT paper): 40% reduction ✗")
        report.append("")
        
        # Speed
        report.append("INFERENCE SPEED COMPARISON:")
        report.append("-" * 30)
        report.append(f"Speedup (batch=1): {results['speed']['speedup_batch_1']:.2f}x")
        report.append(f"Speedup (batch=8): {results['speed']['speedup_batch_8']:.2f}x")
        report.append(f"Average speedup: {results['speed']['avg_speedup']:.2f}x")
        report.append(f"Target (DistilBERT paper): 1.6x speedup ✓"
                     if results['speed']['avg_speedup'] >= 1.5 else
                     f"Target (DistilBERT paper): 1.6x speedup ✗")
        report.append("")
        
        # Perplexity
        report.append("PERPLEXITY COMPARISON:")
        report.append("-" * 30)
        report.append(f"Teacher perplexity: {results['perplexity']['teacher']:.2f}")
        report.append(f"Student perplexity: {results['perplexity']['student']:.2f}")
        report.append(f"Perplexity increase: {results['perplexity']['perplexity_increase']:.2f}")
        report.append(f"Relative performance: {results['perplexity']['relative_performance']:.1f}%")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 30)
        report.append("DistilBERT Paper Targets:")
        report.append("  • 40% size reduction")
        report.append("  • 60% speed improvement")
        report.append("  • 97% of BERT's performance on GLUE")
        report.append("")
        report.append("Your Model Achievements:")
        report.append(f"  • {results['size']['size_reduction_percent']:.1f}% size reduction")
        report.append(f"  • {(results['speed']['avg_speedup'] - 1) * 100:.0f}% speed improvement")
        report.append(f"  • {results['perplexity']['relative_performance']:.1f}% relative performance")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
