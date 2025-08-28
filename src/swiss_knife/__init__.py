"""
Swiss Knife Generic SLM - DistilBERT-based Knowledge Distillation Framework

Based on "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
Sanh et al., 2019 (https://arxiv.org/pdf/1909.10351)
"""

from .distillation import DistilBERTDistiller
from .losses import DistillationLoss, CosineEmbeddingLoss, TripleLoss
from .initialization import initialize_student_from_teacher, create_distilbert_from_bert
from .evaluation import DistillationEvaluator
from .data_preparation import DatasetPreparator, prepare_custom_dataset, create_quick_demo_dataset
from .utils import setup_logging, get_device

__version__ = "0.2.0"
__all__ = [
    "DistilBERTDistiller",
    "DistillationLoss",
    "CosineEmbeddingLoss",
    "TripleLoss",
    "initialize_student_from_teacher",
    "create_distilbert_from_bert",
    "DistillationEvaluator",
    "DatasetPreparator",
    "prepare_custom_dataset",
    "create_quick_demo_dataset",
    "setup_logging",
    "get_device",
]
