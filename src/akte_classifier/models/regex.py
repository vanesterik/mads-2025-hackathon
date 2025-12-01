import csv
import re
from typing import Dict, List

import torch
from loguru import logger


class RegexGenerator:
    """
    Based on a CSV file with code and description,
    this class automatically generate regex patterns for each code.
    Includes manual overrides for specific domain knowledge.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.regexes: Dict[int, str] = {}

        # --- MANUAL OVERRIDES ---
        self.manual_overrides = {
            # 606: "Overdracht" is legally called "Levering" in deeds
            606: r"(akte\s+van\s+)?(levering|eigendomsoverdracht)",
            # 545: Remove "stuk betreffende" (database metadata)
            545: r"kwalitatieve\s+verplichting",
            # 572: Remove "stuk betreffende", handle singular/plural
            572: r"erfdienstbaarhe(id|den)",
            # 537: Hypotheek (Stop matching "Doorhaling" or "Royement")
            537: r"(?<!doorhaling\s)(?<!royement\s)(akte\s+van\s+)?hypotheek(?!.*(doorhaling|royement|afstand))",
            # 538: Hypotheek doorhaling (Cancellation)
            538: r"(algeheel\s+)?(royement|afstand|doorhaling).*(hypotheek|recht)",
            # 517: Beslag (Stop matching "Doorhaling")
            517: r"(?<!doorhaling\s)(?<!opheffing\s)(proces-verbaal\s+van\s+)?beslag(?!.*(doorhaling|opheffing))",
            # 518: Beslag doorhaling
            518: r"(doorhaling|opheffing).*beslag",
            # 652: Koopovereenkomst 7:3 (Vormerkung)
            652: r"(koopovereenkomst|vormerkung).*(7:3|inschrijving)",
            # 543: Koopovereenkomst beëindiging
            543: r"(ontbinding|beëindiging|vernietiging).*(koopovereenkomst|koopcontract)",
            # 532: Aanvullende akte (Often called Rectificatie/Depot)
            532: r"(akte\s+van\s+)?(rectificatie|aanvulling|verbetering|depot)",
            # 564: Verbetering (Similar to 532, but sometimes distinct)
            564: r"(blad)?verbetering",
            # 644: Opstal Nutsvoorzieningen (Simplify)
            644: r"(vestiging\s+)?(zakelijk\s+)?recht\s+van\s+opstal.*(nut|kabel|leiding|netwerk|trafo)",
            # 516: Beperkt recht wijziging
            516: r"(wijziging|aanvulling).*(voorwaarden|bepalingen)",
            # 527: Erfpachtcanon wijziging
            527: r"(wijziging|herziening).*(canon|erfpacht)",
            # 616: Splitsing
            616: r"(akte\s+van\s+)?(hoofd)?splitsing.*appartementsrecht(en)?",
            # 671: Vermenging
            671: r"(akte\s+van\s+)?vermenging",
            # 581: Divorce/Partnership
            581: r"verdeling.*(echtscheiding|ontbinding|huwelijk|partnerschap)",
            # 579: Inheritance/Heirs
            579: r"verdeling.*(nalatenschap|erfgena(a)?m|overlijden)",
            # 580: General Division (Catch-all if keywords above are missing)
            580: r"(akte\s+van\s+)?verdeling",
        }
        # ------------------------

        self._generate_regexes()

    def _generate_regexes(self):
        """
        Parses the CSV and generates regex patterns for each code.
        """
        logger.info(f"Parsing {self.csv_path} for regex generation...")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            if header and header[0].lower() != "code":
                f.seek(0)

            for row in reader:
                if not row or len(row) < 2:
                    continue

                try:
                    code = int(row[0].strip())
                    description = row[1].strip()

                    # --- CHECK FOR OVERRIDE FIRST ---
                    if code in self.manual_overrides:
                        self.regexes[code] = self.manual_overrides[code]
                    else:
                        # Fallback to auto-generation
                        pattern = self._create_pattern(description)
                        self.regexes[code] = pattern
                    # --------------------------------

                except ValueError:
                    continue

        logger.info(f"Generated {len(self.regexes)} regex patterns.")

    def generate_hash(self) -> str:
        """
        Generates a hash so we can spot changes in the regex patterns.
        This helps with caching
        """
        import hashlib

        # Sort by code to ensure stability
        sorted_items = sorted(self.regexes.items())
        # Create a string representation
        repr_str = str(sorted_items)
        # Hash it
        return hashlib.md5(repr_str.encode()).hexdigest()[:8]

    def _create_pattern(self, description: str) -> str:
        """
        Converts a description into a flexible regex pattern.
        """
        # Handle "enz." (etc.) - remove it
        clean_desc = description.replace("enz.", "").strip()

        if "(" in clean_desc and ")" in clean_desc:
            # Split by parens
            parts = re.split(r"[\(\)]", clean_desc)
            # Filter empty and strip
            parts = [p.strip() for p in parts if p.strip()]

            pattern_parts = []
            for p in parts:
                escaped = re.escape(p)
                # Flexible whitespace
                escaped = escaped.replace(r"\ ", r"\s+")
                pattern_parts.append(escaped)

            # Combine with .* to match in order, allowing anything in between
            pattern = ".*".join(pattern_parts)
        else:
            # Standard handling
            escaped = re.escape(clean_desc)
            # Flexible whitespace
            pattern = escaped.replace(r"\ ", r"\s+")

        return pattern


class RegexClassifier:
    def __init__(self, generator: RegexGenerator):
        self.patterns = {
            code: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for code, pattern in generator.regexes.items()
        }

    def predict(self, text: str) -> List[int]:
        """
        Returns a list of codes found in the text.
        """
        matches = []
        for code, pattern in self.patterns.items():
            if pattern.search(text):
                matches.append(code)
        return matches


class RegexVectorizer:
    def __init__(self, generator: RegexGenerator, label_encoder=None):
        self.generator = generator
        self.hash = generator.generate_hash()

        """
        Args:
            generator: The RegexGenerator instance.
            label_encoder: Optional LabelEncoder to align outputs with model classes.
        """
        self.output_dim = 0
        self.code_to_idx = {}
        # Use the simple regexes (code -> pattern)
        self.patterns = {}

        if label_encoder is None:
            logger.error(
                "LabelEncoder is required for RegexVectorizer to align with model classes."
            )
            raise ValueError("LabelEncoder is required.")

        # Single Source of Truth: Use the encoder's codes and indices
        self.code_to_idx = label_encoder.code2idx
        self.output_dim = len(label_encoder)

        # Only compile patterns for codes that are in the encoder (and have a regex)
        for code_str, idx in self.code_to_idx.items():
            try:
                code_int = int(code_str)
                if code_int in generator.regexes:
                    self.patterns[idx] = re.compile(
                        generator.regexes[code_int], re.IGNORECASE | re.DOTALL
                    )
            except ValueError:
                continue

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