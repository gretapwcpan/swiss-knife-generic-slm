"""
Command-line interface for DistilBERT-style knowledge distillation.

Based on "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
Sanh et al., 2019 (https://arxiv.org/pdf/1909.10351)
"""

import click
import torch
from pathlib import Path
import json
from typing import Optional

from .distillation import DistilBERTDistiller
from .initialization import (
    create_distilbert_from_bert,
    initialize_student_from_teacher,
    compare_model_sizes,
)
from .evaluation import DistillationEvaluator
from .utils import setup_logging, get_device


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str):
    """
    Swiss Knife Generic SLM - DistilBERT-based Knowledge Distillation
    
    Implementation of the DistilBERT paper (Sanh et al., 2019):
    https://arxiv.org/pdf/1909.10351
    """
    setup_logging(log_level)


@cli.command()
@click.argument("teacher_model")
@click.option("--output-dir", default="./distilled_model", help="Output directory")
@click.option("--layers", help="Comma-separated layer indices to keep (e.g., '0,2,4,6,8,10')")
@click.option("--reduction-factor", default=2, help="Layer reduction factor (default: 2 = every other layer)")
def initialize(teacher_model: str, output_dir: str, layers: Optional[str], reduction_factor: int):
    """
    Initialize a student model from a teacher model following DistilBERT strategy.
    
    From the paper: "The student is initialized from the teacher by taking 
    one layer out of two from the teacher's network."
    
    Example:
        swiss-knife initialize bert-base-uncased --output-dir ./my-distilbert
    """
    click.echo(f"Initializing student from teacher: {teacher_model}")
    
    # Parse layer indices if provided
    student_layers = None
    if layers:
        student_layers = [int(x.strip()) for x in layers.split(",")]
        click.echo(f"Using specified layers: {student_layers}")
    
    # Initialize student
    if "bert" in teacher_model.lower() and not layers:
        # Use the exact DistilBERT configuration
        click.echo("Creating DistilBERT-style model (layers 0,2,4,6,8,10)")
        student = create_distilbert_from_bert(teacher_model, output_dir)
    else:
        # General initialization
        student = initialize_student_from_teacher(
            teacher_model,
            student_layers=student_layers,
            reduction_factor=reduction_factor,
        )
        student.save_pretrained(output_dir)
    
    # Show model comparison
    from transformers import AutoModelForMaskedLM
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_model)
    comparison = compare_model_sizes(teacher, student)
    
    click.echo("\nModel Comparison:")
    click.echo(f"  Teacher parameters: {comparison['teacher_parameters']:,}")
    click.echo(f"  Student parameters: {comparison['student_parameters']:,}")
    click.echo(f"  Compression ratio: {comparison['compression_ratio']:.2f}x")
    click.echo(f"  Size reduction: {comparison['size_reduction_percent']:.1f}%")
    click.echo(f"\nStudent model saved to: {output_dir}")


@cli.command()
@click.argument("teacher_model")
@click.option("--student-model", help="Pre-initialized student model path")
@click.option("--dataset", default="wikitext", help="HuggingFace dataset name")
@click.option("--dataset-config", default="wikitext-103-raw-v1", help="Dataset configuration")
@click.option("--output-dir", default="./distilled_model", help="Output directory")
@click.option("--num-epochs", default=3, help="Number of training epochs")
@click.option("--batch-size", default=32, help="Training batch size")
@click.option("--learning-rate", default=5e-4, help="Learning rate (paper uses 5e-4)")
@click.option("--temperature", default=3.0, help="Distillation temperature (paper uses 3.0)")
@click.option("--alpha", default=0.5, help="Weight for distillation loss")
@click.option("--beta", default=2.0, help="Weight for MLM loss")
@click.option("--gamma", default=1.0, help="Weight for cosine loss")
def distill(
    teacher_model: str,
    student_model: Optional[str],
    dataset: str,
    dataset_config: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    Train a student model using DistilBERT triple loss methodology.
    
    From the paper: "We use a linear combination of the distillation loss Lce 
    with the supervised training loss Lmlm. We also add a cosine embedding loss Lcos"
    
    Example:
        swiss-knife distill bert-base-uncased --output-dir ./my-distilbert
    """
    click.echo("Starting DistilBERT-style knowledge distillation")
    click.echo(f"Teacher: {teacher_model}")
    click.echo(f"Dataset: {dataset}/{dataset_config}")
    click.echo(f"Temperature: {temperature} (paper default: 3.0)")
    click.echo(f"Loss weights: α={alpha}, β={beta}, γ={gamma}")
    
    # Load student if provided
    student = None
    if student_model:
        from transformers import AutoModelForMaskedLM
        click.echo(f"Loading pre-initialized student: {student_model}")
        student = AutoModelForMaskedLM.from_pretrained(student_model)
    
    # Initialize distiller
    distiller = DistilBERTDistiller(
        teacher_model_name=teacher_model,
        student_model=student,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    
    # Train on dataset
    click.echo(f"\nStarting training for {num_epochs} epochs...")
    metrics = distiller.distill_on_dataset(
        dataset_name=dataset,
        dataset_config=dataset_config,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=output_dir,
    )
    
    # Save metrics
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    click.echo(f"\nTraining completed!")
    click.echo(f"Model saved to: {output_dir}")
    click.echo(f"Metrics saved to: {metrics_file}")


@cli.command()
@click.argument("teacher_model")
@click.argument("student_model")
@click.option("--num-samples", default=100, help="Number of samples for evaluation")
def evaluate(teacher_model: str, student_model: str, num_samples: int):
    """
    Evaluate and compare teacher and student models.
    
    Metrics from the DistilBERT paper:
    - 40% size reduction
    - 60% speed improvement  
    - 97% of BERT's performance
    
    Example:
        swiss-knife evaluate bert-base-uncased ./my-distilbert
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    click.echo("Loading models for evaluation...")
    
    # Load models
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_model)
    student = AutoModelForMaskedLM.from_pretrained(student_model)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    
    # Create evaluator
    evaluator = DistillationEvaluator()
    
    # Generate test texts
    test_texts = [
        "The capital of France is [MASK].",
        "Machine learning is a subset of [MASK] intelligence.",
        "The quick brown [MASK] jumps over the lazy dog.",
    ] * (num_samples // 3)
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        teacher, student, tokenizer, test_texts[:num_samples]
    )
    
    click.echo("\n" + report)


@cli.command()
@click.argument("model_path")
@click.option("--task", default="sst2", help="GLUE task name")
@click.option("--num-samples", default=100, help="Number of samples to evaluate")
def benchmark(model_path: str, task: str, num_samples: int):
    """
    Benchmark a model on GLUE tasks.
    
    The DistilBERT paper evaluates on all GLUE tasks to show
    97% performance retention compared to BERT.
    
    Example:
        swiss-knife benchmark ./my-distilbert --task sst2
    """
    from transformers import AutoModelForSequenceClassification
    
    click.echo(f"Benchmarking {model_path} on {task}")
    
    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except:
        click.echo(f"Note: Loading as MLM model. For GLUE tasks, fine-tuning is required.")
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    
    # Evaluate
    evaluator = DistillationEvaluator()
    results = evaluator.evaluate_on_glue_task(model, task, num_samples)
    
    if results:
        click.echo(f"\nResults on {results['task']}:")
        click.echo(f"  Accuracy: {results['accuracy']:.2%}")
        click.echo(f"  Samples evaluated: {results['num_samples']}")
    else:
        click.echo(f"Task {task} evaluation not available in this demo")


@cli.command()
def paper_info():
    """
    Display information about the DistilBERT paper and methodology.
    """
    info = """
    ============================================================
    DistilBERT: A Distilled Version of BERT
    ============================================================
    
    Paper: https://arxiv.org/pdf/1909.10351
    Authors: Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf
    Year: 2019
    
    KEY CONTRIBUTIONS:
    ------------------
    1. Triple Loss Function:
       - Distillation loss (Lce): KL divergence between teacher/student
       - Masked LM loss (Lmlm): Standard BERT MLM objective
       - Cosine embedding loss (Lcos): Align hidden state directions
    
    2. Student Initialization:
       - Take every other layer from teacher (layers 0,2,4,6,8,10)
       - Remove token type embeddings
       - Initialize from teacher weights
    
    3. Training Setup:
       - Same data as BERT (Wikipedia + BookCorpus)
       - Temperature T=3 for distillation
       - Batch size: 4K examples
       - Learning rate: 5e-4
    
    RESULTS:
    --------
    - 40% smaller than BERT-base
    - 60% faster inference
    - 97% of BERT's performance on GLUE
    - 95% of BERT's performance on SQuAD
    
    APPLICATIONS:
    -------------
    - Edge deployment
    - Real-time inference
    - Resource-constrained environments
    - Mobile applications
    
    CITATION:
    ---------
    @article{sanh2019distilbert,
      title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
      author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
      journal={arXiv preprint arXiv:1910.01108},
      year={2019}
    }
    ============================================================
    """
    click.echo(info)


@cli.command()
def quick_start():
    """
    Show quick start examples for common use cases.
    """
    examples = """
    ============================================================
    QUICK START GUIDE
    ============================================================
    
    1. Create a DistilBERT from BERT-base:
    ---------------------------------------
    swiss-knife initialize bert-base-uncased --output-dir ./my-distilbert
    
    2. Distill with custom teacher:
    --------------------------------
    swiss-knife distill bert-large-uncased \\
        --output-dir ./distilled-large \\
        --num-epochs 3 \\
        --batch-size 16
    
    3. Evaluate distilled model:
    -----------------------------
    swiss-knife evaluate bert-base-uncased ./my-distilbert
    
    4. Benchmark on GLUE:
    ---------------------
    swiss-knife benchmark ./my-distilbert --task sst2
    
    5. Custom layer selection:
    ---------------------------
    swiss-knife initialize bert-base-uncased \\
        --layers "0,1,3,5,7,9,11" \\
        --output-dir ./custom-distilbert
    
    TIPS:
    -----
    - Use GPU for faster training (automatically detected)
    - Adjust batch size based on GPU memory
    - Temperature=3.0 works well (paper default)
    - Monitor all three loss components during training
    
    ============================================================
    """
    click.echo(examples)


if __name__ == "__main__":
    cli()
