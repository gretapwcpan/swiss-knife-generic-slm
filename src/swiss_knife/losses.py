"""
Loss functions for DistilBERT-style knowledge distillation.

Based on Section 3 of "DistilBERT, a distilled version of BERT" (Sanh et al., 2019)
The paper uses a triple loss: Lce + Lmlm + Lcos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation loss (Lce) from Hinton et al., 2015.
    
    As described in DistilBERT paper Section 3:
    "a distillation loss over the soft target probabilities of the teacher"
    
    References:
        - Hinton et al., 2015: "Distilling the Knowledge in a Neural Network"
        - Sanh et al., 2019: "DistilBERT" (Section 3, Equation for Lce)
    """
    
    def __init__(self, temperature: float = 3.0):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions.
                        DistilBERT paper uses T=3 based on preliminary experiments.
        """
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, 
                student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher logits.
        
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            mask: Attention mask to ignore padding tokens [batch_size, seq_len]
            
        Returns:
            Distillation loss value
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence
        loss = self.kl_div(student_log_probs, teacher_probs)
        
        # Scale by T^2 as per Hinton et al., 2015
        loss = loss * (self.temperature ** 2)
        
        # Apply mask if provided
        if mask is not None:
            # Reshape for proper masking
            batch_size, seq_len = mask.shape
            mask = mask.view(batch_size, seq_len, 1)
            loss = loss * mask.float()
            loss = loss.sum() / mask.sum()
            
        return loss


class MaskedLanguageModelingLoss(nn.Module):
    """
    Masked Language Modeling loss (Lmlm) as used in DistilBERT.
    
    From the paper: "we also trained DistilBERT on the same corpus 
    used to train the teacher model, using the masked language modeling loss"
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self,
                student_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute MLM loss for student predictions.
        
        Args:
            student_logits: Student model predictions [batch_size, seq_len, vocab_size]
            labels: Ground truth labels with -100 for non-masked tokens [batch_size, seq_len]
            
        Returns:
            MLM loss value
        """
        # Reshape for loss computation
        vocab_size = student_logits.size(-1)
        student_logits = student_logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        return self.criterion(student_logits, labels)


class CosineEmbeddingLoss(nn.Module):
    """
    Cosine embedding loss (Lcos) between student and teacher hidden states.
    
    From the paper: "a cosine embedding loss (Lcos) which will tend to align 
    the directions of the student and teacher hidden states vectors"
    """
    
    def __init__(self):
        super().__init__()
        self.cos_loss = nn.CosineEmbeddingLoss(reduction='mean')
        
    def forward(self,
                student_hidden: torch.Tensor,
                teacher_hidden: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cosine similarity loss between hidden states.
        
        Args:
            student_hidden: Student hidden states [batch_size, seq_len, hidden_dim]
            teacher_hidden: Teacher hidden states [batch_size, seq_len, hidden_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Cosine embedding loss value
        """
        # Ensure same dimensions (project if needed)
        if student_hidden.size(-1) != teacher_hidden.size(-1):
            # This shouldn't happen if student is properly initialized
            logger.warning(f"Hidden dimensions mismatch: student={student_hidden.size(-1)}, "
                         f"teacher={teacher_hidden.size(-1)}")
            return torch.tensor(0.0, device=student_hidden.device)
        
        # Flatten for cosine loss
        batch_size, seq_len, hidden_dim = student_hidden.shape
        student_flat = student_hidden.view(-1, hidden_dim)
        teacher_flat = teacher_hidden.view(-1, hidden_dim)
        
        # Target is always 1 (we want similarity)
        target = torch.ones(student_flat.size(0), device=student_hidden.device)
        
        # Compute cosine loss
        loss = self.cos_loss(student_flat, teacher_flat, target)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat.float()
            loss = loss.sum() / mask_flat.sum()
            
        return loss


class TripleLoss(nn.Module):
    """
    Combined triple loss as described in DistilBERT paper.
    
    From Section 3: "We use a linear combination of the distillation loss Lce 
    with the supervised training loss, in our case the masked language modeling 
    loss Lmlm. We also add a cosine embedding loss (Lcos)"
    
    Total loss = α * Lce + β * Lmlm + γ * Lcos
    """
    
    def __init__(self,
                 temperature: float = 3.0,
                 alpha: float = 0.5,
                 beta: float = 2.0,
                 gamma: float = 1.0):
        """
        Initialize triple loss.
        
        Args:
            temperature: Temperature for distillation (paper uses 3.0)
            alpha: Weight for distillation loss
            beta: Weight for MLM loss  
            gamma: Weight for cosine loss
            
        Note: The paper doesn't specify exact weights, these are reasonable defaults.
        """
        super().__init__()
        self.distillation_loss = DistillationLoss(temperature)
        self.mlm_loss = MaskedLanguageModelingLoss()
        self.cosine_loss = CosineEmbeddingLoss()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_hidden: torch.Tensor,
                teacher_hidden: torch.Tensor,
                mlm_labels: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute the combined triple loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            student_hidden: Student hidden states
            teacher_hidden: Teacher hidden states
            mlm_labels: Labels for MLM task
            attention_mask: Attention mask
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Compute individual losses
        l_ce = self.distillation_loss(student_logits, teacher_logits, attention_mask)
        l_mlm = self.mlm_loss(student_logits, mlm_labels)
        l_cos = self.cosine_loss(student_hidden, teacher_hidden, attention_mask)
        
        # Combine with weights
        total_loss = self.alpha * l_ce + self.beta * l_mlm + self.gamma * l_cos
        
        # Return total and components for logging
        components = {
            'distillation_loss': l_ce.item(),
            'mlm_loss': l_mlm.item(),
            'cosine_loss': l_cos.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, components
