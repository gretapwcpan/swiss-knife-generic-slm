"""
Student model initialization strategies based on DistilBERT paper.

From Section 3: "we initialize the student from the teacher by taking 
one layer out of two from the teacher's network"
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoConfig,
    BertModel,
    BertForMaskedLM,
    RobertaModel,
    RobertaForMaskedLM,
)
from typing import Optional, List, Union, Dict, Any
import logging
import copy

logger = logging.getLogger(__name__)


def initialize_student_from_teacher(
    teacher_model: Union[str, nn.Module],
    student_layers: Optional[List[int]] = None,
    reduction_factor: int = 2,
    model_type: str = "bert",
) -> nn.Module:
    """
    Initialize a student model from a teacher model following DistilBERT strategy.
    
    From the paper: "The student is initialized from the teacher by taking 
    one layer out of two from the teacher's network."
    
    Args:
        teacher_model: Either a model name/path or loaded model instance
        student_layers: Specific layer indices to keep. If None, uses reduction_factor
        reduction_factor: Factor by which to reduce layers (default=2 for DistilBERT)
        model_type: Type of model ("bert", "roberta", etc.)
        
    Returns:
        Initialized student model
    """
    # Load teacher model if string provided
    if isinstance(teacher_model, str):
        logger.info(f"Loading teacher model from: {teacher_model}")
        if "roberta" in teacher_model.lower():
            teacher = AutoModelForMaskedLM.from_pretrained(teacher_model)
            model_type = "roberta"
        else:
            teacher = AutoModelForMaskedLM.from_pretrained(teacher_model)
            model_type = "bert"
    else:
        teacher = teacher_model
    
    # Get teacher configuration
    teacher_config = teacher.config
    
    # Create student configuration
    student_config = copy.deepcopy(teacher_config)
    
    # Determine which layers to keep
    if hasattr(teacher_config, 'num_hidden_layers'):
        num_teacher_layers = teacher_config.num_hidden_layers
    else:
        # Try to infer from model structure
        if hasattr(teacher, 'bert'):
            num_teacher_layers = len(teacher.bert.encoder.layer)
        elif hasattr(teacher, 'roberta'):
            num_teacher_layers = len(teacher.roberta.encoder.layer)
        else:
            raise ValueError("Cannot determine number of teacher layers")
    
    if student_layers is None:
        # Default DistilBERT strategy: take every other layer
        # Paper: "taking one layer out of two"
        student_layers = list(range(0, num_teacher_layers, reduction_factor))
        logger.info(f"Using layers {student_layers} from teacher (every {reduction_factor} layers)")
    
    # Update student config
    student_config.num_hidden_layers = len(student_layers)
    
    # Remove token type embeddings as per DistilBERT
    # Paper: "we remove the token-type embeddings"
    if hasattr(student_config, 'type_vocab_size'):
        student_config.type_vocab_size = 0
    
    # Create student model with updated config
    logger.info(f"Creating student with {len(student_layers)} layers")
    
    if model_type == "roberta":
        student = RobertaForMaskedLM(student_config)
    else:
        student = BertForMaskedLM(student_config)
    
    # Copy weights from teacher to student
    _copy_weights_to_student(teacher, student, student_layers, model_type)
    
    logger.info(f"Student initialized with {count_parameters(student):,} parameters "
                f"(teacher has {count_parameters(teacher):,})")
    
    return student


def _copy_weights_to_student(
    teacher: nn.Module,
    student: nn.Module,
    student_layers: List[int],
    model_type: str = "bert"
) -> None:
    """
    Copy weights from teacher to student model.
    
    Strategy from DistilBERT:
    1. Copy embeddings (except token type)
    2. Copy selected transformer layers
    3. Copy prediction head
    """
    # Get the base model (bert/roberta)
    if model_type == "roberta":
        teacher_base = teacher.roberta
        student_base = student.roberta
    else:
        teacher_base = teacher.bert if hasattr(teacher, 'bert') else teacher.base_model
        student_base = student.bert if hasattr(student, 'bert') else student.base_model
    
    # 1. Copy embeddings
    logger.info("Copying embeddings...")
    
    # Word embeddings
    student_base.embeddings.word_embeddings.weight.data.copy_(
        teacher_base.embeddings.word_embeddings.weight.data
    )
    
    # Position embeddings
    student_base.embeddings.position_embeddings.weight.data.copy_(
        teacher_base.embeddings.position_embeddings.weight.data
    )
    
    # Layer norm
    student_base.embeddings.LayerNorm.weight.data.copy_(
        teacher_base.embeddings.LayerNorm.weight.data
    )
    student_base.embeddings.LayerNorm.bias.data.copy_(
        teacher_base.embeddings.LayerNorm.bias.data
    )
    
    # Note: We skip token_type_embeddings as per DistilBERT paper
    
    # 2. Copy transformer layers
    logger.info(f"Copying layers {student_layers}...")
    
    for student_idx, teacher_idx in enumerate(student_layers):
        logger.debug(f"Copying teacher layer {teacher_idx} to student layer {student_idx}")
        
        teacher_layer = teacher_base.encoder.layer[teacher_idx]
        student_layer = student_base.encoder.layer[student_idx]
        
        # Copy all parameters from teacher layer to student layer
        student_layer.load_state_dict(teacher_layer.state_dict())
    
    # 3. Copy pooler if exists
    if hasattr(teacher_base, 'pooler') and hasattr(student_base, 'pooler'):
        logger.info("Copying pooler...")
        student_base.pooler.dense.weight.data.copy_(
            teacher_base.pooler.dense.weight.data
        )
        student_base.pooler.dense.bias.data.copy_(
            teacher_base.pooler.dense.bias.data
        )
    
    # 4. Copy prediction head (MLM head)
    logger.info("Copying MLM head...")
    
    # Copy the LM head
    if hasattr(teacher, 'cls'):
        student.cls.predictions.transform.dense.weight.data.copy_(
            teacher.cls.predictions.transform.dense.weight.data
        )
        student.cls.predictions.transform.dense.bias.data.copy_(
            teacher.cls.predictions.transform.dense.bias.data
        )
        
        student.cls.predictions.transform.LayerNorm.weight.data.copy_(
            teacher.cls.predictions.transform.LayerNorm.weight.data
        )
        student.cls.predictions.transform.LayerNorm.bias.data.copy_(
            teacher.cls.predictions.transform.LayerNorm.bias.data
        )
        
        student.cls.predictions.decoder.weight.data.copy_(
            teacher.cls.predictions.decoder.weight.data
        )
        if hasattr(teacher.cls.predictions.decoder, 'bias'):
            student.cls.predictions.decoder.bias.data.copy_(
                teacher.cls.predictions.decoder.bias.data
            )
    elif hasattr(teacher, 'lm_head'):
        # For RoBERTa-style models
        student.lm_head.dense.weight.data.copy_(
            teacher.lm_head.dense.weight.data
        )
        student.lm_head.dense.bias.data.copy_(
            teacher.lm_head.dense.bias.data
        )
        
        student.lm_head.layer_norm.weight.data.copy_(
            teacher.lm_head.layer_norm.weight.data
        )
        student.lm_head.layer_norm.bias.data.copy_(
            teacher.lm_head.layer_norm.bias.data
        )
        
        student.lm_head.decoder.weight.data.copy_(
            teacher.lm_head.decoder.weight.data
        )
        if hasattr(teacher.lm_head.decoder, 'bias'):
            student.lm_head.decoder.bias.data.copy_(
                teacher.lm_head.decoder.bias.data
            )


def create_distilbert_from_bert(
    bert_model_name: str,
    save_path: Optional[str] = None
) -> nn.Module:
    """
    Create a DistilBERT model from a BERT model following the paper's exact approach.
    
    This is a convenience function that implements the exact DistilBERT initialization:
    - Takes layers 0, 2, 4, 6, 8, 10 from a 12-layer BERT
    - Removes token type embeddings
    - Initializes from teacher weights
    
    Args:
        bert_model_name: Name/path of BERT model to distill from
        save_path: Optional path to save the initialized student
        
    Returns:
        DistilBERT-style student model
    """
    logger.info(f"Creating DistilBERT from {bert_model_name}")
    
    # For BERT-base (12 layers), take every other layer
    # This gives us layers [0, 2, 4, 6, 8, 10] = 6 layers total
    student = initialize_student_from_teacher(
        teacher_model=bert_model_name,
        student_layers=[0, 2, 4, 6, 8, 10],
        model_type="bert"
    )
    
    if save_path:
        logger.info(f"Saving initialized student to {save_path}")
        student.save_pretrained(save_path)
    
    return student


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_model_sizes(teacher: nn.Module, student: nn.Module) -> Dict[str, Any]:
    """
    Compare teacher and student model sizes.
    
    Returns statistics matching those reported in the DistilBERT paper.
    """
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    
    return {
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
        "compression_ratio": teacher_params / student_params,
        "size_reduction_percent": (1 - student_params / teacher_params) * 100,
        "speedup_estimate": teacher_params / student_params * 0.6,  # Conservative estimate
    }
