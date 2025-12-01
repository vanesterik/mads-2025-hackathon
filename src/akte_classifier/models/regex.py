import csv
import re
from typing import Dict, List, Set, Optional
from functools import lru_cache

import torch
from loguru import logger

try:
    import ahocorasick
    USE_AHO_CORASICK = True
    logger.info("Using Aho-Corasick for fast trigger matching")
except ImportError:
    USE_AHO_CORASICK = False
    logger.info("Aho-Corasick not available, using standard trigger matching")

class RegexGenerator:
    """
    Based on a CSV file with code and description,
    this class automatically generate regex patterns for each code.
    Includes manual overrides and TRIGGER WORDS for performance optimization.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.regexes: Dict[int, str] = {}
        self.triggers: Dict[int, List[str]] = {}

        # --- MANUAL REGEX OVERRIDES ---
        self.manual_overrides = {
            608: r"(?s)\b(schenking|gift|om\s+niet|(geen|zonder)\s+tegenprestatie)\b",
            606: r"(?s)\b(levering|vervreemding|eigendomsoverdracht|verkoop|wordt\s+geleverd|lever(en|t)?\s+.*?aanvaard(t|en)?)\b(?!.*om\s+niet)",
            591: r"(?s)verklaring.*waarde(loo)?s(heid)?",
            536: r"(?s)((huwelijkse|partnerschaps)\s*voorwaarden|koude\s+uitsluiting|beperkte\s+gemeenschap|verreken(beding|stelsel))",
            524: r"(?s)\b(cessie|cederen|overdracht\s+van\s+((een\s+)?vordering|hypotheek.*\bbank))\b",
            617: r"(?s)\bondersplitsing\b",
            620: r"(?s)(opheffing|doorhaling).*splitsing.*appartementsrecht",
            618: r"(?s)wijziging.*splitsing.*(indices|breukdelen|aandelen)",
            601: r"(?s)wijziging.*(splitsing|reglement|akte).*((v|vereniging)\s+v(an)?\s+e(igenaars|igenaaren)|modelreglement)",
            598: r"(?s)(juridische\s+)?splitsing.*(rechtspersoon|vennootschap|bv|nv|onderneming)",
            540: r"(?s)\b(kavelruil|ruilverkaveling)\b|((inbreng.*toedeling|toedeling.*inbreng).*(perceel|grond))",
            541: r"(?s)overeenkomst.*(inzake\s+)?kavelruil",
            556: r"(?s)(levering|overdracht).*onder\s+voorbehoud\s+van",
            611: r"(?s)(vestiging.*)?\brecht\s+van\s+vruchtgebruik\b",
            539: r"(?s)(wijziging|verhoging|verlaging|aanpassing).*(hypotheek|hoofdsom|rente)",
            545: r"(?s)(kwalitatieve\s+verplichting|art(ikel)?\.?\s*(6\s*:\s*252|252\s*,?\s*boek\s+6))",
            537: r"(?s)(?<!doorhaling\s)(?<!royement\s)\b(hypotheek(stelling)?|zekerheidstelling|ter\s+verzekering\s+van|verle(en|n)(t|en)?\s+.*?hypotheek)\b(?!.*(doorhaling|royement|afstand))",
            572: r"(?s)(?=(?:.*?\b(erfdienstbaarhe(id|den))\b)).*?\b(vestig(ing|en|t)|gevestigd|afstand\s+te\s+doen|aan(vaard(t|en)?|te\s+nemen))\b",
            538: r"(?s)(algeheel\s+)?(royement|afstand|doorhaling).*(hypotheek|recht)",
            517: r"(?s)(?<!doorhaling\s)(?<!opheffing\s)(proces-verbaal\s+van\s+)?beslag(?!.*(doorhaling|opheffing))",
            518: r"(?s)(doorhaling|opheffing).*beslag",
            652: r"(?s)(koopovereenkomst|vormerkung).*(7:3|inschrijving)",
            543: r"(?s)(ontbinding|beëindiging|vernietiging).*(koopovereenkomst|koopcontract)",
            532: r"(?s)(akte\s+van\s+)?(rectificatie|aanvulling|verbetering|depot)",
            564: r"(?s)(blad)?verbetering",
            644: r"(?s)(vestiging\s+)?(zakelijk\s+)?recht\s+van\s+opstal.*(nut|kabel|leiding|netwerk|trafo)",
            516: r"(?s)(wijziging|aanvulling).*(voorwaarden|bepalingen)",
            527: r"(?s)(wijziging|herziening).*(canon|erfpacht)",
            616: r"(?s)(akte\s+van\s+)?(hoofd)?splitsing.*appartementsrecht(en)?",
            671: r"(?s)(akte\s+van\s+)?vermenging",
            581: r"(?s)verdeling.*(echtscheiding|ontbinding|huwelijk|partnerschap)",
            579: r"(?s)verdeling.*(nalatenschap|erfgena(a)?m|overlijden)",
            580: r"(?s)(akte\s+van\s+)?verdeling",
        }

        self.manual_triggers = {
            608: ["schenking", "gift", "niet", "tegenprestatie"],
            591: ["waardeloos", "verklaring"],
            536: ["huwelijk", "partnerschap", "voorwaarden", "uitsluiting", "verreken"],
            524: ["cessie", "cederen", "overdracht"],
            618: ["wijziging", "indices", "breukdelen", "aandelen"],
            601: ["wijziging", "reglement", "vve", "vereniging"],
            617: ["ondersplitsing"],
            620: ["opheffing", "splitsing"],
            598: ["splitsing", "rechtspersoon", "vennootschap", "onderneming"],
            540: ["kavelruil", "inbreng", "ruilverkaveling"],
            541: ["overeenkomst", "kavelruil"],
            556: ["voorbehoud"],
            611: ["vruchtgebruik"],
            539: ["wijziging", "verhoging", "hypotheek"],
            606: ["levering", "vervreemding", "eigendom", "verkoop", "geleverd", "lever"],
            572: ["erfdienstbaarhe"],
            545: ["kwalitatieve", "252"],
            537: ["hypotheek", "zekerheid"],
            517: ["beslag"],
            652: ["koopovereenkomst", "vormerkung"],
            644: ["opstal"],
            616: ["splitsing"],
            538: ["royement", "afstand", "doorhaling"],
            518: ["doorhaling", "opheffing"],
            581: ["verdeling"],
            579: ["verdeling"],
            580: ["verdeling"],
        }

        self._generate_regexes()

    def _generate_regexes(self):
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

                    # Set Trigger (convert to lowercase)
                    if code in self.manual_triggers:
                        self.triggers[code] = [t.lower() for t in self.manual_triggers[code]]

                    # Set Regex
                    if code in self.manual_overrides:
                        self.regexes[code] = self.manual_overrides[code]
                    else:
                        self.regexes[code] = self._create_pattern(description)
                except ValueError:
                    continue
        logger.info(f"Generated {len(self.regexes)} regex patterns and {len(self.triggers)} trigger sets.")

    def generate_hash(self) -> str:
        import hashlib
        # Hash triggers too, as they affect output logic
        sorted_regex = sorted(self.regexes.items())
        sorted_triggers = sorted(self.triggers.items())
        repr_str = str(sorted_regex) + str(sorted_triggers)
        return hashlib.md5(repr_str.encode()).hexdigest()[:8]

    def _create_pattern(self, description: str) -> str:
        clean_desc = description.replace("enz.", "").strip()
        if "(" in clean_desc and ")" in clean_desc:
            parts = re.split(r"[\(\)]", clean_desc)
            parts = [p.strip() for p in parts if p.strip()]
            pattern_parts = []
            for p in parts:
                escaped = re.escape(p)
                escaped = escaped.replace(r"\ ", r"\s+")
                pattern_parts.append(escaped)
            pattern = ".*".join(pattern_parts)
        else:
            escaped = re.escape(clean_desc)
            pattern = escaped.replace(r"\ ", r"\s+")
        return pattern


class RegexClassifier:
    def __init__(self, generator: RegexGenerator):
        self.patterns = {
            code: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for code, pattern in generator.regexes.items()
        }

    def predict(self, text: str) -> List[int]:
        matches = []
        for code, pattern in self.patterns.items():
            if pattern.search(text):
                matches.append(code)
        return matches


class RegexVectorizer:
    def __init__(self, generator: RegexGenerator, label_encoder=None,
                 use_caching: bool = True, max_cache_size: int = 1000):
        self.generator = generator
        self.hash = generator.generate_hash()
        self.output_dim = 0
        self.code_to_idx = {}
        self.patterns = {}
        self.triggers = {}
        self.use_caching = use_caching

        if label_encoder is None:
            logger.error("LabelEncoder is required for RegexVectorizer.")
            raise ValueError("LabelEncoder is required.")

        self.code_to_idx = label_encoder.code2idx
        self.output_dim = len(label_encoder)

        # Build patterns and triggers
        for code_str, idx in self.code_to_idx.items():
            try:
                code_int = int(code_str)
                # Store regex
                if code_int in generator.regexes:
                    self.patterns[idx] = re.compile(
                        generator.regexes[code_int],
                        re.IGNORECASE | re.DOTALL
                    )

                if code_int in generator.triggers:
                    self.triggers[idx] = set(generator.triggers[code_int])
            except ValueError:
                continue

        self.word_to_codes = {}
        for idx, triggers in self.triggers.items():
            for trigger in triggers:
                self.word_to_codes.setdefault(trigger, []).append(idx)

        # Initialize Aho-Corasick automaton if available
        if USE_AHO_CORASICK:
            self.automaton = ahocorasick.Automaton()
            for trigger, code_list in self.word_to_codes.items():
                self.automaton.add_word(trigger, (trigger, code_list))
            self.automaton.make_automaton()
            logger.info(f"Built Aho-Corasick automaton with {len(self.word_to_codes)} triggers")
        else:
            self.automaton = None

        # Setup caching
        if use_caching:
            self._match_text_cached = lru_cache(maxsize=max_cache_size)(self._match_text_uncached)
            logger.info(f"Enabled caching with max size {max_cache_size}")
        else:
            self._match_text_cached = self._match_text_uncached

    def _match_text_uncached(self, text: str) -> List[int]:
        matches = []
        text_lower = text.lower()  # Lowercase once

        triggered_indices = self._get_triggered_indices(text_lower)

        for idx in triggered_indices:
            if self.patterns.get(idx) and self.patterns[idx].search(text):
                matches.append(idx)

        for idx, pattern in self.patterns.items():
            if idx not in self.triggers:
                if pattern.search(text):
                    matches.append(idx)

        return matches

    def _get_triggered_indices(self, text_lower: str) -> Set[int]:
        triggered_indices = set()

        if self.automaton:
            for _, (_, code_list) in self.automaton.iter(text_lower):
                triggered_indices.update(code_list)
        else:
            for trigger, code_list in self.word_to_codes.items():
                if trigger in text_lower:
                    triggered_indices.update(code_list)

        return triggered_indices

    def _match_text(self, text: str) -> List[int]:
        return self._match_text_cached(text)

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch_size = len(texts)
        features = torch.zeros((batch_size, self.output_dim), dtype=torch.float32)

        if not texts:
            return features

        avg_len = sum(len(t) for t in texts) / len(texts)

        if len(texts) < 10 or avg_len < 100:
            # Sequential processing
            for i, text in enumerate(texts):
                matched_indices = self._match_text(text)
                if matched_indices:
                    features[i, matched_indices] = 1.0
        else:
            from joblib import Parallel, delayed
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), len(texts))

            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                results = parallel(
                    delayed(self._match_text)(text)
                    for text in texts
                )

            for i, matched_indices in enumerate(results):
                if matched_indices:
                    features[i, matched_indices] = 1.0

        return features

    def clear_cache(self):
        """Clear the LRU cache."""
        if hasattr(self._match_text_cached, 'cache_clear'):
            self._match_text_cached.cache_clear()
            logger.info("Cleared regex matching cache")

def profile_method(method_name: str = None):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = method_name or func.__name__
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{name} took {elapsed:.4f}s")
            return result

        return wrapper

    return decorator

if __name__ == "__main__":
    class MockLabelEncoder:
        def __init__(self, codes):
            self.code2idx = {str(code): idx for idx, code in enumerate(codes)}
            self.idx2code = {idx: str(code) for idx, code in enumerate(codes)}

        def __len__(self):
            return len(self.code2idx)

    test_texts = [
        "Dit is een hypotheekakte voor een woning in Amsterdam.",
        "Beslaglegging op de inboedel van de woning.",
        "Kwalitatieve verplichting volgens artikel 6:252.",
        "Levering van een auto aan de koper.",
        "Een erfdienstbaarheid wordt gevestigd voor toegang.",
        "Royement van de hypotheek op het pand.",
    ]

    generator = RegexGenerator("your_csv_file.csv")  # Replace with actual path
    label_encoder = MockLabelEncoder(list(generator.regexes.keys()))
    vectorizer = RegexVectorizer(generator, label_encoder, use_caching=True)

    @profile_method("single_match")
    def test_single():
        text = test_texts[0]
        matches = vectorizer._match_text(text)
        print(f"Text: {text[:50]}...")
        print(f"Matches: {matches}")
        return matches

    @profile_method("batch_forward")
    def test_batch():
        results = vectorizer.forward(test_texts)
        print(f"Batch shape: {results.shape}")
        print(f"Non-zero counts per text: {results.sum(dim=1)}")
        return results

    print("=" * 60)
    print("Testing optimized RegexVectorizer")
    print("=" * 60)

    test_single()
    print("-" * 40)
    test_batch()

    if vectorizer.use_caching:
        cache_info = vectorizer._match_text_cached.cache_info()
        print(f"\nCache stats: Hits={cache_info.hits}, Misses={cache_info.misses}, Size={cache_info.currsize}")