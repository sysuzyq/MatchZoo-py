import numpy as np

from matchzoo.engine.base_metric import sort_and_couple
from matchzoo import metrics


def test_sort_and_couple():
    l = [0, 1, 2]
    s = [0.1, 0.4, 0.2]
    c = sort_and_couple(l, s)
    assert (c == np.array([(1, 0.4), (2, 0.2), (0, 0.1)])).all()


def test_mean_reciprocal_rank_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.MeanReciprocalRank(k=1)(label, score) == 1


def test_precision_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.Precision(k=1)(label, score) == 1.
    assert metrics.Precision(k=2)(label, score) == 1.
    assert round(metrics.Precision(k=3)(label, score), 2) == 0.67


def test_recall_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.Recall(k=1)(label, score) == 0.5
    assert metrics.Recall(k=3)(label, score) == 1.


def test_average_precision():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert round(metrics.AveragePrecision()(label, score), 2) == 0.89


def test_mean_average_precision_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.MeanAveragePrecision(k=1)(label, score) == 1.
    assert metrics.MeanAveragePrecision(k=3)(label, score) == 1.


def test_dcg_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    dcg = metrics.DiscountedCumulativeGain
    assert round(dcg(k=1)(label, score), 2) == 1.44
    assert round(dcg(k=2)(label, score), 2) == 4.17
    assert round(dcg(k=3)(label, score), 2) == 4.17


def test_ndcg_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    ndcg = metrics.NormalizedDiscountedCumulativeGain
    assert round(ndcg(k=1)(label, score), 2) == 0.33
    assert round(ndcg(k=2)(label, score), 2) == 0.80
    assert round(ndcg(k=3)(label, score), 2) == 0.80


def test_accuracy():
    label = np.array([1])
    score = np.array([[0, 1]])
    assert metrics.Accuracy()(label, score) == 1


def test_f1():
    label = np.array([1, 1, 0, 0])
    score = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]])
    assert F1()(label, score) == 0.5


def test_cross_entropy():
    label = [0, 1]
    score = [[0.25, 0.25], [0.01, 0.90]]
    assert round(metrics.CrossEntropy()(label, score), 2) == 0.75
