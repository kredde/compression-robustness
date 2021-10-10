from typing import NamedTuple
import numpy as np


def calculate_ece(predictions: np.ndarray,
                  labels: np.ndarray,
                  n_bins: int = 10) -> float:
    """Calculates the expected calibration error.

    Args:
        predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
        labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
        n_bins: Number of equidistant bins for the scores. Defaults to 10.

    Returns:
        The expected calibration error
    """
    # Get the accuracy, mean confidence, and size/cardinality of each bin
    accuracies, mean_confidences, bin_sizes = calculate_calibration(
        predictions, labels, n_bins)

    # Calculate the absolute calibration error for each bin
    bin_calibration_error = np.abs(accuracies - mean_confidences)

    # Calculate the expected calibration error
    return np.average(np.abs(bin_calibration_error), weights=bin_sizes / len(predictions))


def calculate_class_wise_ece(predictions: np.ndarray,
                             labels: np.ndarray,
                             n_bins: int = 10) -> np.ndarray:
    """Calculates the expected calibration error for each class individually.

    Args:
        predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
        labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
        n_bins: Number of equidistant bins for the scores. Defaults to 10.

    Returns:
        np.ndarray[n_classes]: The expected calibration error for each class
    """
    n_classes = predictions.shape[1]
    class_wise_ece = np.zeros((n_classes,), np.float32)

    for i in range(n_classes):
        # Get the indices where the ground truth corresponds to the current class
        cur_class_indices = np.where(labels == i)

        # Calculate the ECE just for the samples where the ground truth label corresponds to the current class
        class_wise_ece[i] = calculate_ece(
            predictions[cur_class_indices], labels[cur_class_indices], n_bins)

    return class_wise_ece


def calculate_mce(predictions: np.ndarray,
                  labels: np.ndarray,
                  n_bins: int = 10) -> float:
    """Calculates the maximum calibration error, i.e, the highest calibration error of any bin.

    Args:
        predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
        labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
        n_bins: Number of equidistant bins for the scores. Defaults to 10.

    Returns:
        The maximum calibration error
    """
    calibration_errors = calculate_calibration_errors(
        predictions, labels, n_bins)

    # Calculate the expected calibration error
    return calibration_errors[np.argmax(np.abs(calibration_errors))].item()


def calculate_calibration_errors(predictions: np.ndarray,
                                 labels: np.ndarray,
                                 n_bins: int = 10) -> np.ndarray:
    """Calculates the calibration errors of each bin for the provided predictions w.r.t. the provided ground truth.

    Args:
        predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
        labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
        n_bins: Number of equidistant bins for the scores. Defaults to 10.

    Returns:
        calibration_errors (np.ndarray[n_bins]): The calibration error, i.e., difference between observed accuracy and
            mean confidence within a bin, for each bin
    """
    accuracies, mean_confidences, _ = calculate_calibration(
        predictions, labels, n_bins)

    return accuracies - mean_confidences


class CalibrationResult(NamedTuple):
    accuracies: np.ndarray
    mean_confidences: np.ndarray
    bin_sizes: np.ndarray


def calculate_calibration(predictions: np.ndarray,
                          labels: np.ndarray,
                          n_bins: int = 10) -> CalibrationResult:
    """Calculates the calibration of the provided predictions w.r.t. the provided ground truth.

    Args:
        predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
        labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
        n_bins: Number of equidistant bins for the scores. Defaults to 10.

    Returns:
        A NamedTuple containing the following:

        accuracies (np.ndarray[n_bins]): The accuracy of the predictions for each bin

        mean_confidences (np.ndarray[n_bins]): The average confidence of the predictions for each bin

        bin_sizes (np.ndarray[n_bins]): The number of predictions for each bin.
    """
    accuracies = np.zeros(n_bins, np.float32)
    mean_confidences = np.zeros(n_bins, np.float32)
    bin_sizes = np.zeros(n_bins, np.int32)

    for bin_idx in range(n_bins):
        # Filter the predictions to only consider those that fall in the current confidence bin
        confidence_range = (bin_idx / n_bins, (bin_idx + 1) / n_bins)

        prediction_confidences = np.max(predictions, 1)
        if bin_idx < n_bins - 1:
            bin_mask = np.where((confidence_range[0] <= prediction_confidences)
                                & (confidence_range[1] > prediction_confidences), True, False)
        else:
            bin_mask = np.where(
                confidence_range[0] <= prediction_confidences, True, False)

        bin_predictions = predictions[bin_mask]
        bin_labels = labels[bin_mask]

        # Get the accuracy and average confidence of predictions within this bin
        if len(bin_predictions) > 0:
            accuracies[bin_idx] = np.sum(
                np.argmax(bin_predictions, 1) == bin_labels) / len(bin_predictions)
            mean_confidences[bin_idx] = np.mean(np.max(bin_predictions, 1))

        bin_sizes[bin_idx] = len(bin_predictions)

    return CalibrationResult(accuracies, mean_confidences, bin_sizes)


class ConfidenceCalibration():
    """Multiple metrics on confidence calibration, see https://arxiv.org/abs/1706.04599."""

    def __init__(self,
                 predictions: np.ndarray,
                 labels: np.ndarray,
                 n_bins: int = 10):
        """Inititialize confidence calibration metrics for classification tasks.

        Args:
            predictions (np.ndarray[n_samples, n_classes]): The predictions of the classifier for a dataset
            labels (np.ndarray[n_samples]): The ground truth labels (class indices) for the dataset
            n_bins: Number of equidistant bins for the scores. Defaults to 10.
        """
        self.predictions = predictions
        self.labels = labels
        self.n_bins = n_bins

    def get_ece(self) -> float:
        """Get the expected calibration error."""
        return calculate_ece(self.predictions, self.labels, self.n_bins)

    def get_class_wise_ece(self) -> np.ndarray:
        """Calculates the expected calibration error for each class individually.

        Returns:
            np.ndarray[n_bins]: The calibration error for each class individually
        """
        return calculate_class_wise_ece(self.predictions, self.labels, self.n_bins)

    def get_mce(self) -> float:
        """Get the maximum calibration error, i.e, the highest calibration error of any bin."""
        return calculate_mce(self.predictions, self.labels, self.n_bins)

    def get_calibration(self) -> CalibrationResult:
        """Get the calibration of the provided predictions w.r.t. the provided ground truth.

        Returns:
            A NamedTuple containing the following:

            accuracies (np.ndarray[n_bins]): The accuracy of the predictions for each bin

            mean_confidences (np.ndarray[n_bins]): The average confidence of the predictions for each bin

            bin_sizes (np.ndarray[n_bins]): The number of predictions for each bin.
        """
        return calculate_calibration(self.predictions, self.labels, self.n_bins)

    def get_calibration_errors(self) -> np.ndarray:
        """Calculates the calibration errors of each bin for the provided predictions w.r.t. the provided ground truth.

        Returns:
            np.ndarray[n_bins]: The calibration error, i.e., difference between observed accuracy and mean confidence
            within a bin, for each bin
        """
        return calculate_calibration_errors(self.predictions, self.labels, self.n_bins)
