import csv
import re
from typing import Dict, List

import torch
from loguru import logger


class RegexGenerator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.regexes: Dict[int, str] = {}
        self._generate_regexes()

    def _generate_regexes(self):
        """
        Parses the CSV and generates regex patterns for each code.
        """
        logger.info(f"Parsing {self.csv_path} for regex generation...")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            # The CSV uses ';' as delimiter based on user input
            reader = csv.reader(f, delimiter=";")

            # Skip header if present (Code;Waarde)
            header = next(reader, None)
            if header and header[0].lower() != "code":
                # If first row doesn't look like header, reset
                f.seek(0)

            for row in reader:
                if not row or len(row) < 2:
                    continue

                try:
                    code = int(row[0].strip())
                    description = row[1].strip()

                    # Generate regex from description
                    pattern = self._create_pattern(description)
                    self.regexes[code] = pattern

                except ValueError:
                    continue

        logger.info(f"Generated {len(self.regexes)} regex patterns.")

    def _create_pattern(self, description: str) -> str:
        """
        Converts a description into a flexible regex pattern.
        """
        # Escape special regex characters first
        escaped = re.escape(description)

        # Handle "enz." (etc.) - remove it or make it optional
        escaped = escaped.replace(r"enz\.", "")

        # Handle parentheses: make content within () optional
        # A simple approach: replace '\(' with '(?:' and '\)' with ')?'
        # This assumes balanced parens in the description.
        pattern = escaped.replace(r"\(", r"(?:").replace(r"\)", r")?")

        # Handle "art." and "BW" / "WVG"
        # Maybe make whitespace flexible?
        pattern = pattern.replace(r"\ ", r"\s+")

        # Case insensitive flag will be used in compilation
        return pattern


class RegexClassifier:
    def __init__(self, generator: RegexGenerator):
        self.patterns = {
            code: re.compile(pattern, re.IGNORECASE)
            for code, pattern in generator.regexes.items()
        }

    def predict(self, text: str) -> List[int]:
        """
        Returns a list of codes found in the text.
        """
        found_codes = []
        for code, pattern in self.patterns.items():
            if pattern.search(text):
                found_codes.append(code)
        return found_codes


class RegexVectorizer:
    def __init__(self, generator: RegexGenerator, label_encoder=None):
        """
        Args:
            generator: The RegexGenerator instance.
            label_encoder: Optional LabelEncoder. If provided, the vectorizer will
                           only use patterns for codes present in the encoder and
                           will align the output vector with the encoder's indices.
        """
        # Use the simple regexes (code -> pattern)
        self.patterns = {}

        if label_encoder:
            # Single Source of Truth: Use the encoder's codes and indices
            self.code_to_idx = label_encoder.code2idx
            self.output_dim = len(label_encoder)

            # Only compile patterns for codes that are in the encoder (and have a regex)
            for code_str, idx in self.code_to_idx.items():
                # Encoder keys are strings, generator keys might be ints
                try:
                    code_int = int(code_str)
                    if code_int in generator.regexes:
                        self.patterns[idx] = re.compile(
                            generator.regexes[code_int], re.IGNORECASE
                        )
                except ValueError:
                    continue
        else:
            # Fallback to using all regexes sorted by code
            for code, pattern in generator.regexes.items():
                self.patterns[code] = re.compile(pattern, re.IGNORECASE)

            self.sorted_codes = sorted(self.patterns.keys())
            self.code_to_idx = {code: i for i, code in enumerate(self.sorted_codes)}
            self.output_dim = len(self.sorted_codes)

            # Remap patterns to use the new index directly for faster lookup in forward
            self.patterns = {
                self.code_to_idx[code]: pat for code, pat in self.patterns.items()
            }

    def _match_text(self, text: str) -> List[int]:
        """Helper for parallel processing: returns list of indices that matched."""
        matches = []
        for idx, pattern in self.patterns.items():
            if pattern.search(text):
                matches.append(idx)
        return matches

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Transforms a list of texts into a binary tensor of shape (batch_size, num_classes).
        Values:
        0: No match
        1: Match
        """
        from joblib import Parallel, delayed

        batch_size = len(texts)
        features = torch.zeros((batch_size, self.output_dim), dtype=torch.float32)

        # Parallelize the matching process
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=-1)(delayed(self._match_text)(text) for text in texts)

        for i, matched_indices in enumerate(results):
            if matched_indices:
                features[i, matched_indices] = 1.0

        return features
