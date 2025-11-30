import torch

from akte_classifier.models.neural import HybridClassifier, NeuralClassifier


def test_neural_classifier_init():
    model = NeuralClassifier(input_dim=768, num_classes=10)
    assert isinstance(model, torch.nn.Module)


def test_hybrid_classifier_init():
    model = HybridClassifier(input_dim=768, regex_dim=50, num_classes=10)
    assert isinstance(model, torch.nn.Module)
