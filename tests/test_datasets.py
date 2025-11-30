from akte_classifier.datasets.dataset import DatasetFactory


def test_dataset_factory_init():
    factory = DatasetFactory("assets/test.jsonl")
    assert factory.file_path == "assets/test.jsonl"
    assert factory.encoder is not None


def test_get_vectorized_dataset():
    factory = DatasetFactory("assets/test.jsonl", split_ratio=0.8)
    # This might take a bit if not cached, but we are testing functionality
    # Assuming cache exists from previous runs or it will compute
    datasets = factory.get_vectorized_dataset()
    assert "train" in datasets
    assert "test" in datasets
