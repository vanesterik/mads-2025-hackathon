from typer.testing import CliRunner

from akte_classifier.cli import app

runner = CliRunner()


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
    if result.exit_code != 0:
        print(result.stdout)
        print(result.exception)
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
    if result.exit_code != 0:
        print(result.stdout)
        print(result.exception)
    assert result.exit_code == 0
