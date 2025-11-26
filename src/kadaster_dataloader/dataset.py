import sys
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")


class LabelEncoder:
    """
    Helper to map raw sparse integer codes to dense indices (0..N) and back.
    Reserves index 0 for 'Unknown' codes.
    """

    def __init__(self, train_codes: List[int]):
        # Get unique codes and sort them for deterministic behavior
        self.unique_codes = sorted(list(set(train_codes)))

        # Reserve 0 for unknown
        self.unknown_idx = 0

        # Create mappings
        # We start real codes from index 1
        self.code2idx = {code: i + 1 for i, code in enumerate(self.unique_codes)}
        self.idx2code = {i + 1: code for i, code in enumerate(self.unique_codes)}

        # Add unknown mapping
        self.idx2code[self.unknown_idx] = 0

    def __len__(self):
        # +1 for the unknown category
        return len(self.unique_codes) + 1

    def encode(self, codes: List[int]) -> torch.Tensor:
        """Converts a list of raw codes into a Multi-Hot Tensor."""
        # Create a vector of zeros with length equal to total classes
        vector = torch.zeros(len(self), dtype=torch.float32)

        # Set indices to 1 for present codes
        for code in codes:
            if code in self.code2idx:
                vector[self.code2idx[code]] = 1.0
            else:
                # Map to unknown if not found
                vector[self.unknown_idx] = 1.0
        return vector

    def decode(self, vector: torch.Tensor) -> List[int]:
        """Converts a Multi-Hot Tensor/Probabilities back to raw codes."""
        # Get indices where value is high (assuming threshold 0.5 for logits)
        indices = (vector > 0.5).nonzero(as_tuple=True)[0]
        decoded = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx2code:
                decoded.append(self.idx2code[idx_val])
            else:
                # Should not happen if logic is correct, but safe fallback
                decoded.append(0)
        return decoded


class RechtsfeitDataset(Dataset):
    def __init__(self, hf_dataset, encoder: LabelEncoder):
        self.dataset = hf_dataset
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        raw_codes = item["rechtsfeitcodes"]

        # Transform raw codes to Multi-Hot Tensor
        label_tensor = self.encoder.encode(raw_codes)

        return text, label_tensor


class DatasetFactory:
    def __init__(self, file_path: str, batch_size: int = 32, split_ratio: float = 0.8):
        self.file_path = file_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.train_dataset: Optional[RechtsfeitDataset] = None
        self.test_dataset: Optional[RechtsfeitDataset] = None
        self.encoder: Optional[LabelEncoder] = None

        self._prepare_data()

    def _prepare_data(self):
        logger.info(f"Loading {self.file_path}...")
        # Load the full dataset
        full_dataset = load_dataset("json", data_files=self.file_path, split="train")

        # Split the dataset
        # We use a fixed seed for reproducibility
        split_ds = full_dataset.train_test_split(train_size=self.split_ratio, seed=42)
        train_data = split_ds["train"]
        test_data = split_ds["test"]

        # Build encoder based ONLY on training data
        logger.info("Building label index from training data...")
        train_codes = []
        for codes in train_data["rechtsfeitcodes"]:
            train_codes.extend(codes)

        self.encoder = LabelEncoder(train_codes)
        logger.info(f"Found {len(self.encoder)} unique categories (including Unknown).")

        # Create Dataset objects
        self.train_dataset = RechtsfeitDataset(train_data, self.encoder)
        self.test_dataset = RechtsfeitDataset(test_data, self.encoder)

    def get_dataset(self) -> Dict[str, RechtsfeitDataset]:
        if self.train_dataset is None or self.test_dataset is None:
            self._prepare_data()

        assert self.train_dataset is not None
        assert self.test_dataset is not None

        return {"train": self.train_dataset, "test": self.test_dataset}

    def get_loader(self) -> Dict[str, DataLoader]:
        datasets = self.get_dataset()

        train_loader = DataLoader(
            datasets["train"], batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        test_loader = DataLoader(
            datasets["test"], batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        return {"train": train_loader, "test": test_loader}


if __name__ == "__main__":
    logger.info("Starting dataset processing...")
    from pathlib import Path

    datadir = Path("data/raw")
    file_path = "ai_challenge_data_anonymized_19789.jsonl"
    path = datadir / file_path

    factory = DatasetFactory(str(path), batch_size=2, split_ratio=0.6)
    loaders = factory.get_loader()

    # Assert encoder is initialized for mypy
    assert factory.encoder is not None

    logger.info("\n--- Testing Train Batch ---")
    for batch_texts, batch_labels in loaders["train"]:
        logger.info(f"Batch text count: {len(batch_texts)}")
        logger.info(f"Batch labels shape: {batch_labels.shape}")
        logger.info(f"Sample restored codes: {factory.encoder.decode(batch_labels[0])}")
        break

    logger.info("\n--- Testing Test Batch ---")
    for batch_texts, batch_labels in loaders["test"]:
        logger.info(f"Batch text count: {len(batch_texts)}")
        logger.info(f"Batch labels shape: {batch_labels.shape}")
        logger.info(f"Sample restored codes: {factory.encoder.decode(batch_labels[0])}")

        # Check if we have any unknowns (index 0)
        if batch_labels[0][0] == 1.0:
            logger.info("-> Found an UNKNOWN code in this sample!")
        break
