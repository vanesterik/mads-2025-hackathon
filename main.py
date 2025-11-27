from typing import Optional

import pandas as pd
import typer
from loguru import logger

from kadaster_dataloader.regex_model import RegexGenerator
from kadaster_dataloader.trainer import Trainer, TrainingConfig

app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    epochs: int = 50,
    batch_size: int = 32,
    model_class: str = "HybridClassifier",
    device: Optional[str] = None,
):
    """
    Train the model.
    """
    logger.info("Starting training...")
    config = TrainingConfig(
        num_epochs=epochs, batch_size=batch_size, model_class=model_class
    )
    if device:
        config.device = device

    trainer = Trainer(config)
    trainer.run()


@app.command()
def evaluate_regex(
    data_path: str = "assets/aktes.jsonl",
    csv_path: str = "assets/rechtsfeiten.csv",
    output_path: str = "artifacts/csv/regex_evaluation.csv",
):
    """
    Evaluate the regex model using cached features.
    """
    from kadaster_dataloader.dataset import DatasetFactory
    from kadaster_dataloader.regex_model import RegexVectorizer

    logger.info("Initializing DatasetFactory...")
    factory = DatasetFactory(data_path)

    logger.info("Initializing Regex Model...")
    generator = RegexGenerator(csv_path)
    # Use the encoder from the factory for alignment
    regex_vectorizer = RegexVectorizer(generator, label_encoder=factory.encoder)

    logger.info("Getting (cached) vectorized dataset...")
    # We only need the regex features, but the factory computes them alongside embeddings if we use get_vectorized_dataset.
    # To avoid computing BERT embeddings if they aren't needed, we could have a separate method,
    # but for now let's assume we might want them or they are already cached.
    # Actually, we can pass a dummy vectorizer if we really wanted to avoid BERT, but let's stick to the standard flow.
    # Since we need to pass a vectorizer to get_vectorized_dataset, let's just instantiate the TextVectorizer.
    # It won't compute if cache exists.
    from kadaster_dataloader.model import TextVectorizer

    text_vectorizer = TextVectorizer("prajjwal1/bert-medium")  # Default

    vectorized_data = factory.get_vectorized_dataset(text_vectorizer, regex_vectorizer)
    train_data = vectorized_data["train"]

    logger.info("Evaluating on training set...")

    # Get the tensors
    # train_data[i] returns (embedding, regex_feat, label)
    # We can access the full tensors directly from the dataset object if we exposed them,
    # but let's iterate or access attributes if possible.
    # VectorizedRechtsfeitDataset stores them as attributes.

    regex_features = train_data.regex_features  # Shape: (N, NumClasses)
    true_labels = train_data.labels  # Shape: (N, NumClasses)

    if regex_features is None:
        logger.error("Regex features not found in dataset!")
        return

    # Convert to numpy for easier metric calculation or use torch
    # Let's use the same logic as before but vectorized

    # True Positives: both are 1
    tp = (regex_features * true_labels).sum(dim=0)
    # False Positives: regex is 1, true is 0
    fp = (regex_features * (1 - true_labels)).sum(dim=0)
    # False Negatives: regex is 0, true is 1
    fn = ((1 - regex_features) * true_labels).sum(dim=0)

    metrics_data = []

    # Iterate over classes in the encoder
    for code_str, idx in factory.encoder.code2idx.items():
        try:
            code = int(code_str)
        except ValueError:
            continue

        t = tp[idx].item()
        f = fp[idx].item()
        n = fn[idx].item()

        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics_data.append(
            {
                "code": code,
                "precision": f"{precision:.5f}",
                "recall": f"{recall:.5f}",
                "f1": f"{f1:.5f}",
                "tp": int(t),
                "fp": int(f),
                "fn": int(n),
                "regex": generator.regexes.get(code, "N/A"),
            }
        )

    df = pd.DataFrame(metrics_data)
    df.sort_values("f1", ascending=False, inplace=True)

    logger.info(f"Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("\nTop 10 Regexes by F1 Score:")
    logger.info(
        "\n" + df.head(10)[["code", "f1", "precision", "recall", "regex"]].to_string()
    )


@app.command()
def analyze(
    data_path: str = "assets/aktes.jsonl",
):
    """
    Analyze the dataset and generate a label distribution plot.
    """
    from kadaster_dataloader.analysis import analyze_label_distribution
    from kadaster_dataloader.dataset import DatasetFactory

    logger.info("Loading data for analysis...")
    factory = DatasetFactory(data_path)

    logger.info("Analyzing label distribution...")
    analyze_label_distribution(factory.train_dataset)
    logger.info("Analysis complete. Check artifacts/img/label_distribution.png")


if __name__ == "__main__":
    app()
