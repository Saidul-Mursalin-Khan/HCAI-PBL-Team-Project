"""
Dataset loading utilities for the Active Learning / Learning-to-Defer project.

AG News: 4-class news topic classification.
Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
"""

from datasets import load_dataset


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def get_ag_news():
    """
    Loads the AG News dataset from Hugging Face.

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    ds = load_dataset("fancyzhx/ag_news")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    return train_texts, train_labels, test_texts, test_labels
