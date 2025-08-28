# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Swiss Knife Generic SLM is a comprehensive toolkit for reducing large language models (LLMs) to smaller language models (SLMs) through various compression techniques including quantization, pruning, and knowledge distillation.

## Project Structure

```
src/swiss_knife/
├── __init__.py              # Main package exports
├── cli.py                   # Command-line interface
├── utils.py                 # Utilities and configuration classes
├── evaluation.py            # Model evaluation and comparison tools
└── compression/
    ├── __init__.py         # Compression module exports
    ├── quantization.py     # Model quantization (4-bit, 8-bit, dynamic, static)
    ├── pruning.py          # Weight and structural pruning techniques
    └── distillation.py     # Knowledge distillation implementation
```

## Development Commands

### Installation and Setup
- `pip install -e .` - Install package in development mode
- `pip install -e ".[dev]"` - Install with development dependencies
- `pip install -e ".[dev,onnx]"` - Install with optional dependencies

### Code Quality
- `black src/` - Format code with Black
- `isort src/` - Sort imports
- `mypy src/` - Type checking
- `pytest tests/` - Run test suite
- `pytest --cov=swiss_knife --cov-report=term-missing` - Run tests with coverage

### CLI Usage
- `swiss-knife quantize <model> --method 4bit --output-dir ./output` - Quantize model
- `swiss-knife prune <model> --sparsity 0.5 --output-dir ./output` - Prune model
- `swiss-knife evaluate <model> --eval-data data.txt` - Evaluate model
- `swiss-knife compare <original> <compressed>` - Compare models
- `swiss-knife create-configs` - Generate sample configuration files
- `swiss-knife device-info` - Show available compute devices

## Key Architecture Components

### Compression Techniques
1. **Quantization**: Reduces model precision (FP32 → FP16 → INT8 → INT4) using BitsAndBytes
2. **Pruning**: Removes weights/neurons using magnitude-based, structured, or gradual methods  
3. **Knowledge Distillation**: Trains smaller student models to mimic larger teacher models
4. **Layer Reduction**: Removes entire transformer layers to reduce model size

### Configuration System
- Uses `CompressionConfig` dataclass for pipeline configuration
- Supports JSON and YAML configuration files
- Sample configurations available via `swiss-knife create-configs`

### Evaluation Framework
- Comprehensive metrics: perplexity, inference time, memory usage, model size
- Model comparison utilities for before/after analysis
- Generation quality evaluation with BLEU scores
- Throughput benchmarking across batch sizes

## Dependencies

### Core Dependencies
- `torch>=2.0.0` - PyTorch for model operations
- `transformers>=4.30.0` - HuggingFace models and utilities
- `bitsandbytes>=0.41.0` - Efficient quantization
- `accelerate>=0.20.0` - Model loading and training acceleration
- `datasets>=2.14.0` - Dataset handling
- `optimum>=1.12.0` - Model optimization

### Development Dependencies
- `pytest>=7.4.0` - Testing framework
- `black>=23.7.0` - Code formatting
- `mypy>=1.5.0` - Type checking
- `isort>=5.12.0` - Import sorting

## Common Workflows

1. **Model Quantization**: Load model → Apply quantization → Evaluate performance → Save compressed model
2. **Model Pruning**: Load model → Calculate importance scores → Remove weights/layers → Fine-tune → Evaluate
3. **Knowledge Distillation**: Load teacher/student models → Prepare training data → Train with distillation loss → Evaluate student
4. **Evaluation Pipeline**: Load models → Run comprehensive evaluation → Generate comparison reports

## Notes for Development

- All compression methods support both HuggingFace model names and local model paths
- Evaluation tools work with any compressed model format
- CLI provides high-level interface; Python API offers fine-grained control
- Configuration files enable reproducible compression pipelines
- Device auto-detection supports CUDA, MPS, and CPU execution