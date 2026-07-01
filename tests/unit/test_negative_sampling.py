import numpy as np

from training.negative_sampling import sample_in_batch_negatives


def test_negatives_exclude_the_positive_item():
    positives = np.array([5, 10, 20])
    negatives = sample_in_batch_negatives(positives, num_items=100, ratio=4)

    assert negatives.shape == (3, 4)
    for row, positive in zip(negatives, positives, strict=True):
        assert positive not in row


def test_negative_ratio_respected():
    positives = np.array([1])
    negatives = sample_in_batch_negatives(positives, num_items=50, ratio=8)
    assert negatives.shape == (1, 8)
