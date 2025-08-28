"""
Dataset preparation for DistilBERT-style knowledge distillation.

Based on Section 3 of "DistilBERT, a distilled version of BERT" (Sanh et al., 2019):
"DistilBERT is trained on the same corpus as the original BERT model: 
a concatenation of English Wikipedia and Toronto Book Corpus"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from typing import Optional, List, Dict, Any, Union, Tuple
import logging
from pathlib import Path
import json
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """
    Custom dataset for knowledge distillation following DistilBERT paper.
    
    From the paper: "We use a concatenation of English Wikipedia and 
    Toronto Book Corpus for training, the same data as BERT."
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        """
        Initialize distillation dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length (paper uses 512)
            mlm_probability: Masking probability for MLM (paper uses 15%)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Flatten tensors
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return encoding


class DatasetPreparator:
    """
    Prepare datasets for DistilBERT training following the paper's methodology.
    
    The paper uses:
    - English Wikipedia
    - Toronto Book Corpus (BookCorpus)
    - Same preprocessing as BERT
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize dataset preparator.
        
        Args:
            tokenizer: Tokenizer to use for preprocessing
        """
        self.tokenizer = tokenizer
        
    def prepare_wikipedia(
        self,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> HFDataset:
        """
        Prepare Wikipedia dataset as mentioned in the paper.
        
        From the paper: "a concatenation of English Wikipedia..."
        
        Args:
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples (for testing)
            
        Returns:
            Processed Wikipedia dataset
        """
        logger.info("Loading Wikipedia dataset (as per DistilBERT paper)...")
        
        # Load Wikipedia dataset
        # Note: Using wikipedia dataset from HuggingFace as proxy
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            cache_dir=cache_dir,
        )
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Process Wikipedia articles
        def process_wikipedia(examples):
            # Extract text content
            texts = examples["text"]
            
            # Clean and prepare texts
            processed_texts = []
            for text in texts:
                # Remove very short articles
                if len(text) > 100:
                    # Split into paragraphs for better training
                    paragraphs = text.split("\n\n")
                    for para in paragraphs:
                        if len(para) > 50:
                            processed_texts.append(para.strip())
            
            return {"text": processed_texts}
        
        # Process in batches
        dataset = dataset.map(
            process_wikipedia,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Processing Wikipedia",
        )
        
        logger.info(f"Wikipedia dataset prepared: {len(dataset)} samples")
        return dataset
    
    def prepare_bookcorpus(
        self,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> HFDataset:
        """
        Prepare BookCorpus dataset as mentioned in the paper.
        
        From the paper: "...and Toronto Book Corpus"
        
        Args:
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples (for testing)
            
        Returns:
            Processed BookCorpus dataset
        """
        logger.info("Loading BookCorpus dataset (as per DistilBERT paper)...")
        
        # Load BookCorpus dataset
        # Note: Original BookCorpus may not be available, using bookcorpusopen as alternative
        try:
            dataset = load_dataset(
                "bookcorpusopen",
                split="train",
                cache_dir=cache_dir,
            )
        except:
            logger.warning("BookCorpus not available, using alternative dataset")
            # Fallback to another book dataset
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="train",
                cache_dir=cache_dir,
            )
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Process book texts
        def process_books(examples):
            texts = examples.get("text", examples.get("title", []))
            
            # Clean and prepare texts
            processed_texts = []
            for text in texts:
                if isinstance(text, str) and len(text) > 50:
                    # Split very long texts
                    if len(text) > 1000:
                        # Split into chunks
                        chunks = [text[i:i+512] for i in range(0, len(text), 400)]
                        processed_texts.extend(chunks)
                    else:
                        processed_texts.append(text.strip())
            
            return {"text": processed_texts}
        
        # Process in batches
        dataset = dataset.map(
            process_books,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Processing BookCorpus",
        )
        
        logger.info(f"BookCorpus dataset prepared: {len(dataset)} samples")
        return dataset
    
    def prepare_distillation_dataset(
        self,
        use_wikipedia: bool = True,
        use_bookcorpus: bool = True,
        cache_dir: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None,
        validation_split: float = 0.05,
    ) -> Tuple[HFDataset, HFDataset]:
        """
        Prepare the complete dataset for DistilBERT training.
        
        From the paper: "DistilBERT is trained on the same corpus as the 
        original BERT model: a concatenation of English Wikipedia and 
        Toronto Book Corpus"
        
        Args:
            use_wikipedia: Whether to include Wikipedia
            use_bookcorpus: Whether to include BookCorpus
            cache_dir: Directory to cache datasets
            max_samples_per_dataset: Max samples per dataset (for testing)
            validation_split: Fraction of data for validation
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        datasets_to_concat = []
        
        # Load Wikipedia if requested
        if use_wikipedia:
            wiki_dataset = self.prepare_wikipedia(cache_dir, max_samples_per_dataset)
            datasets_to_concat.append(wiki_dataset)
        
        # Load BookCorpus if requested
        if use_bookcorpus:
            book_dataset = self.prepare_bookcorpus(cache_dir, max_samples_per_dataset)
            datasets_to_concat.append(book_dataset)
        
        # Concatenate datasets
        if len(datasets_to_concat) > 1:
            logger.info("Concatenating Wikipedia and BookCorpus (as per paper)...")
            full_dataset = concatenate_datasets(datasets_to_concat)
        else:
            full_dataset = datasets_to_concat[0]
        
        # Shuffle the dataset
        logger.info("Shuffling dataset...")
        full_dataset = full_dataset.shuffle(seed=42)
        
        # Split into train and validation
        if validation_split > 0:
            split = full_dataset.train_test_split(
                test_size=validation_split,
                seed=42,
            )
            train_dataset = split["train"]
            val_dataset = split["test"]
            
            logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
        else:
            train_dataset = full_dataset
            val_dataset = None
            logger.info(f"Dataset size: {len(train_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def tokenize_dataset(
        self,
        dataset: HFDataset,
        max_length: int = 512,
        preprocessing_num_workers: int = 4,
    ) -> HFDataset:
        """
        Tokenize dataset for training.
        
        Args:
            dataset: Dataset to tokenize
            max_length: Maximum sequence length (paper uses 512)
            preprocessing_num_workers: Number of workers for preprocessing
            
        Returns:
            Tokenized dataset
        """
        logger.info(f"Tokenizing dataset with max_length={max_length}...")
        
        def tokenize_function(examples):
            # Tokenize texts
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        
        # Tokenize in batches
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=["text"],
            desc="Tokenizing",
        )
        
        return tokenized_dataset
    
    def create_data_loaders(
        self,
        train_dataset: HFDataset,
        val_dataset: Optional[HFDataset] = None,
        batch_size: int = 32,
        mlm_probability: float = 0.15,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create data loaders for training.
        
        From the paper: "We use a batch size of 4K examples"
        (Note: Adjust batch_size based on your GPU memory)
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size for training
            mlm_probability: Masking probability (paper uses 15%)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )
        
        # Create train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        
        # Create validation loader if dataset provided
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
        
        logger.info(f"Data loaders created: batch_size={batch_size}, mlm_probability={mlm_probability}")
        
        return train_loader, val_loader


def prepare_custom_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    validation_split: float = 0.1,
    max_samples: Optional[int] = None,
) -> Tuple[HFDataset, HFDataset]:
    """
    Prepare a custom dataset from text files.
    
    Args:
        data_path: Path to data file or directory
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        validation_split: Validation split ratio
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading custom dataset from {data_path}")
    
    texts = []
    data_path = Path(data_path)
    
    # Load texts from file(s)
    if data_path.is_file():
        # Single file
        if data_path.suffix == ".json":
            with open(data_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = data
                elif isinstance(data, dict) and "texts" in data:
                    texts = data["texts"]
        else:
            # Text file
            with open(data_path, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        # Directory of files
        for file_path in data_path.glob("*.txt"):
            with open(file_path, "r") as f:
                texts.extend([line.strip() for line in f if line.strip()])
    
    # Limit samples if requested
    if max_samples and len(texts) > max_samples:
        texts = random.sample(texts, max_samples)
    
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Create HuggingFace dataset
    dataset = HFDataset.from_dict({"text": texts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    # Split into train and validation
    if validation_split > 0:
        split = tokenized_dataset.train_test_split(
            test_size=validation_split,
            seed=42,
        )
        return split["train"], split["test"]
    else:
        return tokenized_dataset, None


def create_quick_demo_dataset(
    tokenizer: AutoTokenizer,
    num_samples: int = 1000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create a quick demo dataset for testing the implementation.
    
    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info(f"Creating demo dataset with {num_samples} samples...")
    
    # Generate sample texts
    sample_texts = [
        "The capital of France is Paris, a beautiful city known for its culture.",
        "Machine learning is a subset of artificial intelligence focused on data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have revolutionized computer vision applications.",
        "The transformer architecture has become fundamental in modern NLP.",
        "BERT stands for Bidirectional Encoder Representations from Transformers.",
        "Knowledge distillation transfers knowledge from large models to smaller ones.",
        "The DistilBERT paper shows how to create efficient language models.",
    ]
    
    # Expand to requested number of samples
    expanded_texts = []
    for _ in range(num_samples // len(sample_texts) + 1):
        expanded_texts.extend(sample_texts)
    expanded_texts = expanded_texts[:num_samples]
    
    # Create dataset
    dataset = HFDataset.from_dict({"text": expanded_texts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Split into train and validation
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Create loaders
    train_loader = DataLoader(
        split["train"],
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    val_loader = DataLoader(
        split["test"],
        batch_size=8,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    logger.info(f"Demo dataset created: {len(split['train'])} train, {len(split['test'])} validation")
    
    return train_loader, val_loader
