"""F1 metric for Classification."""
import numpy as np

from matchzoo.engine.base_metric import ClassificationMetric


class F1(ClassificationMetric):
    """F1 metric."""

    ALIAS = ['f1']

    def __init__(self):
        """:class:`F1` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate f1.

        Example:
            >>> import numpy as np
            >>> y_true = np.array([1, 1, 0, 0])
            >>> y_pred = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]])
            >>> F1()(y_true, y_pred)
            0.5

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: F1.
        """
        y_pred = np.argmax(y_pred, axis=1)

        tp = np.sum(y_pred * y_true)
        recall = tp * 1.0 / np.sum(y_true)
        precision = tp * 1.0 / np.sum(y_pred)

        f1 = 2 * recall * precision / (recall + precision)
        return f1
