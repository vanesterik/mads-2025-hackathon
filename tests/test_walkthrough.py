import os
import shutil

from typer.testing import CliRunner

from akte_classifier.cli import app

runner = CliRunner()


def test_split_data():
    # Ensure clean state
    if os.path.exists("assets/test_output"):
        shutil.rmtree("assets/test_output")

    result = runner.invoke(
        app,
        [
            "split-data",
            "--data-path",
            "assets/aktes.jsonl",
            "--output-dir",
            "assets/test_output",
            "--test-size",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists("assets/test_output/train.jsonl")
    assert os.path.exists("assets/test_output/test.jsonl")

    # Cleanup
    shutil.rmtree("assets/test_output")


def test_analyze():
    result = runner.invoke(app, ["analyze", "--data-path", "assets/aktes.jsonl"])
    assert result.exit_code == 0
    assert os.path.exists("artifacts/csv/label_distribution.csv")
    assert os.path.exists("artifacts/img/label_distribution.png")


def test_regex():
    result = runner.invoke(app, ["regex"])
    assert result.exit_code == 0
    # Check for at least one regex evaluation file
    assert any("regex_evaluation" in f for f in os.listdir("artifacts/csv"))


def test_train_regex_only():
    result = runner.invoke(
        app, ["train", "--model-class", "RegexOnlyClassifier", "--epochs", "1"]
    )
    assert result.exit_code == 0


def test_train_neural():
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
    assert result.exit_code == 0


def test_train_hybrid():
    result = runner.invoke(
        app,
        [
            "train",
            "--model-class",
            "HybridClassifier",
            "--model-name",
            "prajjwal1/bert-tiny",
            "--epochs",
            "1",
        ],
    )
    assert result.exit_code == 0


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
