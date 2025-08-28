# AWS SageMaker Deployment Guide for DistilBERT Training

Based on the computational requirements from the DistilBERT paper (Sanh et al., 2019) and practical experience with knowledge distillation.

## 📊 Recommended SageMaker Instances

### For Production Training (Full Wikipedia + BookCorpus)

| Instance Type | GPUs | GPU Memory | Use Case | Cost/Hour* | Training Time Estimate |
|--------------|------|------------|----------|------------|----------------------|
| **ml.p3.8xlarge** | 4x V100 | 64 GB | **RECOMMENDED** - Full training | ~$12.24 | 3-5 days |
| **ml.p3.16xlarge** | 8x V100 | 128 GB | Large-scale parallel training | ~$24.48 | 2-3 days |
| **ml.p4d.24xlarge** | 8x A100 | 320 GB | Fastest training (premium) | ~$32.77 | 1-2 days |

### For Development/Testing

| Instance Type | GPUs | GPU Memory | Use Case | Cost/Hour* | Training Time Estimate |
|--------------|------|------------|----------|------------|----------------------|
| **ml.g4dn.xlarge** | 1x T4 | 16 GB | Small experiments | ~$0.736 | Not recommended for full training |
| **ml.g4dn.12xlarge** | 4x T4 | 64 GB | Medium experiments | ~$3.912 | 7-10 days |
| **ml.g5.4xlarge** | 1x A10G | 24 GB | Development/testing | ~$1.624 | 10-14 days |
| **ml.g5.12xlarge** | 4x A10G | 96 GB | Moderate training | ~$5.672 | 5-7 days |

*Prices are approximate and vary by region (US East as reference)

## 🎯 Recommended Configuration

### **Best Value: ml.p3.8xlarge**

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='./src',
    role=role,
    instance_type='ml.p3.8xlarge',  # 4x V100 GPUs
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'batch_size': 32,  # Per GPU
        'gradient_accumulation_steps': 32,  # Effective batch = 32*4*32 = 4096
        'num_epochs': 3,
        'learning_rate': 5e-4,
        'temperature': 3.0,
        'max_length': 512,
        'mlm_probability': 0.15,
    },
    max_run=432000,  # 5 days max
)
```

## 💾 Memory Requirements

### Model Sizes (FP16 Training)

| Model | Parameters | Memory (Training) | Memory (Inference) |
|-------|------------|------------------|-------------------|
| BERT-base (Teacher) | 110M | ~6 GB | ~1.5 GB |
| DistilBERT (Student) | 66M | ~4 GB | ~0.9 GB |
| Both Models + Optimizer | - | ~12-14 GB | - |

### Batch Size Recommendations

| GPU Memory | Max Batch Size | Gradient Accumulation | Effective Batch |
|------------|---------------|----------------------|-----------------|
| 16 GB (T4) | 8 | 512 | 4,096 |
| 24 GB (A10G) | 16 | 256 | 4,096 |
| 32 GB (V100) | 32 | 128 | 4,096 |
| 40 GB (A100) | 48 | 85 | 4,080 |

## 🚀 Complete SageMaker Training Script

### `sagemaker_train.py`

```python
import argparse
import os
import torch
from transformers import AutoTokenizer
from swiss_knife import (
    DistilBERTDistiller,
    DatasetPreparator,
    create_distilbert_from_bert,
)

def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
    # Prepare datasets
    preparator = DatasetPreparator(tokenizer)
    
    # Load data based on mode
    if args.use_demo_data:
        from swiss_knife import create_quick_demo_dataset
        train_loader, val_loader = create_quick_demo_dataset(
            tokenizer, num_samples=1000
        )
    else:
        # Full Wikipedia + BookCorpus
        train_dataset, val_dataset = preparator.prepare_distillation_dataset(
            use_wikipedia=True,
            use_bookcorpus=True,
            max_samples_per_dataset=args.max_samples,
            validation_split=0.05,
        )
        
        # Tokenize
        train_dataset = preparator.tokenize_dataset(train_dataset)
        val_dataset = preparator.tokenize_dataset(val_dataset)
        
        # Create loaders
        train_loader, val_loader = preparator.create_data_loaders(
            train_dataset,
            val_dataset,
            batch_size=args.batch_size,
            mlm_probability=args.mlm_probability,
        )
    
    # Initialize student
    if args.student_checkpoint:
        from transformers import AutoModelForMaskedLM
        student = AutoModelForMaskedLM.from_pretrained(args.student_checkpoint)
    else:
        student = create_distilbert_from_bert(args.teacher_model)
    
    # Initialize distiller
    distiller = DistilBERTDistiller(
        teacher_model_name=args.teacher_model,
        student_model=student,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        device=device,
    )
    
    # Train
    metrics = distiller.train(
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_dir=args.output_dir,
        save_steps=args.save_steps,
    )
    
    print("Training completed!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--teacher_model", default="bert-base-uncased")
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--output_dir", default="/opt/ml/model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    
    # Data arguments
    parser.add_argument("--use_demo_data", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    train(args)
```

## 📦 SageMaker Notebook Setup

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = get_execution_role()
sess = sagemaker.Session()

# Upload code to S3
code_location = sess.upload_data(
    path='./src',
    key_prefix='distilbert-training/code'
)

# Configure training job
estimator = PyTorch(
    entry_point='sagemaker_train.py',
    source_dir='./src',
    role=role,
    instance_type='ml.p3.8xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'teacher_model': 'bert-base-uncased',
        'batch_size': 32,
        'gradient_accumulation_steps': 32,
        'num_epochs': 3,
        'learning_rate': 5e-4,
        'temperature': 3.0,
        'use_demo_data': False,  # Set True for testing
    },
    max_run=432000,  # 5 days
    use_spot_instances=True,  # Save 70% on costs
    max_wait=432000,
)

# Start training
estimator.fit()
```

## 💰 Cost Optimization Tips

### 1. **Use Spot Instances** (Save 70%)
```python
estimator = PyTorch(
    # ... other parameters ...
    use_spot_instances=True,
    max_wait=432000,  # Max wait time for spot
)
```

### 2. **Use Gradient Accumulation**
Instead of large batch sizes requiring more GPUs:
```python
# Effective batch size = 4096 (paper default)
# batch_size * num_gpus * gradient_accumulation = 4096
# 32 * 4 * 32 = 4096 (for ml.p3.8xlarge)
```

### 3. **Start with Smaller Datasets**
```python
hyperparameters={
    'use_demo_data': True,  # For initial testing
    'max_samples': 10000,   # Limit dataset size
}
```

## 📈 Monitoring Training

### CloudWatch Metrics
```python
from sagemaker.analytics import TrainingJobAnalytics

metrics_dataframe = TrainingJobAnalytics(
    training_job_name=estimator.latest_training_job.name
).dataframe()

# Plot training metrics
metrics_dataframe.plot(
    x='timestamp', 
    y=['train_loss', 'eval_loss'],
    title='Training Progress'
)
```

### Real-time Logs
```python
estimator.logs()  # Stream logs during training
```

## 🔧 Multi-GPU Training

For instances with multiple GPUs (ml.p3.8xlarge has 4 GPUs):

```python
# The code automatically uses DataParallel
# Effective batch size = batch_size * num_gpus
# For ml.p3.8xlarge: 32 * 4 = 128 per step
```

## 📊 Expected Training Times

| Dataset Size | ml.g5.4xlarge | ml.p3.8xlarge | ml.p4d.24xlarge |
|-------------|---------------|---------------|-----------------|
| Demo (1K samples) | 30 min | 10 min | 5 min |
| Small (100K samples) | 12 hours | 4 hours | 2 hours |
| Medium (1M samples) | 5 days | 2 days | 1 day |
| Full (Wikipedia+Books) | 14 days | 4 days | 2 days |

## 🚨 Important Considerations

1. **Storage**: Ensure sufficient EBS volume (at least 100GB for full dataset)
2. **Checkpointing**: Save checkpoints frequently (every 1000 steps)
3. **Monitoring**: Set up CloudWatch alarms for GPU utilization
4. **Budget**: Set spending limits in AWS Budgets
5. **Region**: Use regions with better GPU availability (us-east-1, us-west-2)

## 📝 Paper Reference

The DistilBERT paper trained on 8 V100 GPUs with:
- Batch size: 4K examples
- Training time: ~90 hours
- Dataset: Wikipedia + BookCorpus

Our ml.p3.8xlarge recommendation (4 V100s) will take approximately twice as long but costs less than using 8 GPUs.
