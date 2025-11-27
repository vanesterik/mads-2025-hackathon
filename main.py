from typing import Optional

import pandas as pd
import typer
from loguru import logger

from kadaster_dataloader.regex_model import RegexGenerator
from kadaster_dataloader.trainer import Trainer, TrainingConfig

app = typer.Typer(no_args_is_help=True)


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
    assert factory.train_dataset is not None, "Train dataset is None"
    analyze_label_distribution(factory.train_dataset)
    logger.info("Analysis complete. Check artifacts/img/label_distribution.png")


@app.command()
def train(
    model_class: str = typer.Option(
        "NeuralClassifier",
        help="Model class to use: NeuralClassifier, HybridClassifier, or RegexOnlyClassifier",
    ),
    model_name: str = typer.Option(
        "prajjwal1/bert-tiny", help="HuggingFace model name for text vectorization"
    ),
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    device: Optional[str] = None,
):
    """
    Train the model.
    """
    # set use_regex=True if HybridClassifier or RegexOnlyClassifier is selected
    use_regex = model_class in ["HybridClassifier", "RegexOnlyClassifier"]
    logger.info("Starting training...")
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        model_class=model_class,
        model_name=model_name,
        learning_rate=learning_rate,
        use_regex=use_regex,
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
    from pathlib import Path

    from kadaster_dataloader.dataset import DatasetFactory
    from kadaster_dataloader.model import TextVectorizer
    from kadaster_dataloader.regex_model import RegexVectorizer

    logger.info("Initializing DatasetFactory...")
    factory = DatasetFactory(data_path)

    logger.info("Initializing Regex Model...")
    generator = RegexGenerator(csv_path)
    # Use the encoder from the factory for alignment
    assert factory.encoder is not None, "LabelEncoder is None"
    regex_vectorizer = RegexVectorizer(generator, label_encoder=factory.encoder)

    # Append hash to output filename for versioning

    p = Path(output_path)
    output_path = str(p.with_name(f"{p.stem}_{regex_vectorizer.hash}{p.suffix}"))

    logger.info("Getting (cached) vectorized dataset...")
    text_vectorizer = TextVectorizer("prajjwal1/bert-tiny")  # Default

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


if __name__ == "__main__":
    app()
