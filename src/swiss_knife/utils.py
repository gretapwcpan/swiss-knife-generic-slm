"""
Utility functions for DistilBERT-style knowledge distillation.
"""

import torch
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_device() -> str:
    """
    Get the best available device (cuda, mps, or cpu).
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_model_size(num_params: int) -> str:
    """
    Format model size in human-readable format.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "110M", "1.5B")
    """
    if num_params < 1e6:
        return f"{num_params:,}"
    elif num_params < 1e9:
        return f"{num_params/1e6:.1f}M"
    else:
        return f"{num_params/1e9:.2f}B"
