"""
Main DistilBERT-style knowledge distillation trainer.

Based on "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
Sanh et al., 2019 (https://arxiv.org/pdf/1909.10351)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from typing import Optional, Dict, Any, List, Union
import logging
from tqdm import tqdm
import os
from pathlib import Path

from .losses import TripleLoss
from .initialization import (
    initialize_student_from_teacher,
    compare_model_sizes,
)

logger = logging.getLogger(__name__)


class DistilBERTDistiller:
    """
    Knowledge distillation trainer following DistilBERT methodology.
    
    Key features from the paper:
    - Triple loss: Lce (distillation) + Lmlm (MLM) + Lcos (cosine)
    - Student initialized from teacher (every other layer)
    - Training on same data as teacher
    - Temperature T=3 for distillation
    """
    
    def __init__(
        self,
        teacher_model_name: str,
        student_model: Optional[nn.Module] = None,
        temperature: float = 3.0,
        alpha: float = 0.5,
        beta: float = 2.0,
        gamma: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Initialize DistilBERT distiller.
        
        Args:
            teacher_model_name: Name/path of teacher model
            student_model: Optional pre-initialized student model
            temperature: Distillation temperature (paper uses 3.0)
            alpha: Weight for distillation loss
            beta: Weight for MLM loss
            gamma: Weight for cosine loss
            device: Device to use for training
        """
        self.teacher_model_name = teacher_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load teacher model
        logger.info(f"Loading teacher model: {teacher_model_name}")
        self.teacher = AutoModelForMaskedLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.teacher.eval()  # Teacher always in eval mode
        
        # Initialize or load student model
        if student_model is None:
            logger.info("Initializing student from teacher (DistilBERT strategy)")
            self.student = initialize_student_from_teacher(
                self.teacher,
                reduction_factor=2,  # Take every other layer
            )
        else:
            self.student = student_model
        self.student = self.student.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize loss function
        self.loss_fn = TripleLoss(
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        
        # Log model comparison
        comparison = compare_model_sizes(self.teacher, self.student)
        logger.info(f"Model comparison:")
        logger.info(f"  Teacher params: {comparison['teacher_parameters']:,}")
        logger.info(f"  Student params: {comparison['student_parameters']:,}")
        logger.info(f"  Compression ratio: {comparison['compression_ratio']:.2f}x")
        logger.info(f"  Size reduction: {comparison['size_reduction_percent']:.1f}%")
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 5e-4,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        eval_dataloader: Optional[DataLoader] = None,
        save_steps: int = 1000,
        save_dir: str = "./distilled_model",
        log_steps: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the student model using DistilBERT methodology.
        
        From the paper:
        - "We train DistilBERT on the same corpus as the original BERT model"
        - "We use a batch size of 4K examples per batch"
        - "The student is trained with the Adam optimizer with a learning rate of 5e-4"
        
        Args:
            train_dataloader: Training data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate (paper uses 5e-4)
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            eval_dataloader: Optional evaluation data loader
            save_steps: Save checkpoint every N steps
            save_dir: Directory to save checkpoints
            log_steps: Log metrics every N steps
            
        Returns:
            Dictionary of training metrics
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.student.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "distillation_loss": [],
            "mlm_loss": [],
            "cosine_loss": [],
            "eval_loss": [],
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        
        global_step = 0
        self.student.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {
                "distillation_loss": 0.0,
                "mlm_loss": 0.0,
                "cosine_loss": 0.0,
            }
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=False,
            )
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get teacher outputs (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
                
                # Get student outputs
                student_outputs = self.student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                
                # Calculate triple loss
                # Use last hidden states for cosine loss
                loss, loss_components = self.loss_fn(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    student_hidden=student_outputs.hidden_states[-1],
                    teacher_hidden=teacher_outputs.hidden_states[-1],
                    mlm_labels=batch.get("labels", batch["input_ids"]),
                    attention_mask=batch["attention_mask"],
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update metrics
                epoch_loss += loss.item() * gradient_accumulation_steps
                for key in epoch_metrics:
                    epoch_metrics[key] += loss_components.get(key, 0)
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        max_grad_norm,
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log metrics
                    if global_step % log_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        metrics["train_loss"].append(avg_loss)
                        for key in epoch_metrics:
                            metrics[key].append(
                                epoch_metrics[key] / (step + 1)
                            )
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self._save_checkpoint(
                            save_dir,
                            global_step,
                            optimizer,
                            scheduler,
                            metrics,
                        )
                    
                    # Evaluation
                    if eval_dataloader and global_step % save_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        metrics["eval_loss"].append(eval_loss)
                        logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                        self.student.train()
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Average loss: {avg_epoch_loss:.4f}")
            for key, value in epoch_metrics.items():
                avg_value = value / len(train_dataloader)
                logger.info(f"  Average {key}: {avg_value:.4f}")
        
        # Save final model
        final_path = os.path.join(save_dir, "final")
        logger.info(f"Saving final model to {final_path}")
        self.student.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        return metrics
    
    def evaluate(
        self,
        eval_dataloader: DataLoader,
    ) -> float:
        """
        Evaluate the student model.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Average evaluation loss
        """
        self.student.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model outputs
                teacher_outputs = self.teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                
                student_outputs = self.student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                
                # Calculate loss
                loss, _ = self.loss_fn(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    student_hidden=student_outputs.hidden_states[-1],
                    teacher_hidden=teacher_outputs.hidden_states[-1],
                    mlm_labels=batch.get("labels", batch["input_ids"]),
                    attention_mask=batch["attention_mask"],
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.student.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(
        self,
        save_dir: str,
        step: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, List[float]],
    ) -> None:
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.student.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Checkpoint saved at step {step}")
    
    def distill_on_dataset(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        max_length: int = 512,
        mlm_probability: float = 0.15,
        batch_size: int = 32,
        num_epochs: int = 3,
        **training_kwargs,
    ) -> Dict[str, List[float]]:
        """
        Convenience method to distill on a HuggingFace dataset.
        
        This implements the training setup described in the DistilBERT paper:
        - Training on the same data as BERT (Wikipedia + BookCorpus)
        - Using masked language modeling
        - Batch size and training hyperparameters from the paper
        
        Args:
            dataset_name: Name of HuggingFace dataset
            dataset_config: Dataset configuration
            max_length: Maximum sequence length
            mlm_probability: Probability of masking tokens (paper uses 15%)
            batch_size: Batch size (paper uses 4K, adjust based on GPU)
            num_epochs: Number of epochs
            **training_kwargs: Additional arguments for train()
            
        Returns:
            Training metrics
        """
        from datasets import load_dataset
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        
        # Create data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            tokenized_dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        
        eval_dataloader = None
        if "validation" in tokenized_dataset:
            eval_dataloader = DataLoader(
                tokenized_dataset["validation"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
        
        # Train the model
        return self.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_epochs=num_epochs,
            **training_kwargs,
        )
