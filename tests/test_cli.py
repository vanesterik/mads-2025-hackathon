import shutil
from pathlib import Path

from typer.testing import CliRunner

from akte_classifier.cli import app

runner = CliRunner()


def test_split_data():
    output_dir = Path("assets/test_output")
    # Ensure clean state
    if output_dir.exists():
        shutil.rmtree(output_dir)

    result = runner.invoke(
        app,
        [
            "split-data",
            "--data-path",
            "assets/aktes.jsonl",
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()

    # Cleanup
    shutil.rmtree(output_dir)


def test_analyze():
    result = runner.invoke(app, ["analyze", "--data-path", "assets/aktes.jsonl"])
    assert result.exit_code == 0
    assert Path("artifacts/csv/label_distribution.csv").exists()
    assert Path("artifacts/img/label_distribution.png").exists()


def test_regex():
    result = runner.invoke(app, ["regex"])
    assert result.exit_code == 0
    # Check for at least one regex evaluation file
    csv_dir = Path("artifacts/csv")
    assert any("regex_evaluation" in f.name for f in csv_dir.iterdir())


def test_llm_classify():
    result = runner.invoke(
        app,
        [
            "llm-classify",
            "--limit",
            "2",
            "--model-name",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        ],
    )
    assert result.exit_code == 0


def test_train_and_eval():
    # 1. Train a model
    result = runner.invoke(
        app,
        [
            "train",
            "--model-class",
            "NeuralClassifier",
            "--model-name",
            "prajjwal1/bert-tiny",
            "--epochs",
            "1",
        ],
    )
    if result.exit_code != 0:
        print(f"Train failed: {result.stdout}")
        print(result.exception)
    assert result.exit_code == 0

    # 2. Find the saved model
    models_dir = Path("artifacts/models")
    # Find the .pt file with the latest timestamp
    pt_files = [
        f
        for f in models_dir.iterdir()
        if f.suffix == ".pt" and "prajjwal1_bert-tiny" in f.name
    ]
    assert len(pt_files) > 0
    pt_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # Latest first
    model_path = pt_files[0]

    codes_path = model_path.with_name(model_path.stem + "_codes.json")
    # The stem might already include part of the name, but our naming convention is:
    # name.pt -> name_codes.json
    # Let's verify the actual naming convention in trainer.py
    # It appends _codes.json to the base name.
    # If filename is model.pt, stem is model.
    # So model_codes.json is correct.

    assert codes_path.exists()

    # 3. Run eval
    test_file = Path("assets/test.jsonl")
    if not test_file.exists():
        # Create dummy if not exists
        with open(test_file, "w") as f:
            f.write('{"text": "test", "labels": []}\n')

    result = runner.invoke(
        app,
        [
            "eval",
            "--eval-file",
            str(test_file),
            "--model-path",
            str(model_path),
            "--codes-path",
            str(codes_path),
            "--model-class",
            "NeuralClassifier",
            "--model-name",
            "prajjwal1/bert-tiny",
        ],
    )
    assert result.exit_code == 0, (
        f"Eval failed: {result.stdout}\nException: {result.exception}"
    )
    # Check for per-class metrics file which should contain the eval filename
    # The filename format is {timestamp}_eval_{filename}_per_class_metrics.csv
    csv_dir = Path("artifacts/csv")
    assert any(
        f"eval_{test_file.name}" in f.name and "per_class_metrics" in f.name
        for f in csv_dir.iterdir()
    )
