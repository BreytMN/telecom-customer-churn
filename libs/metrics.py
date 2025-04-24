from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self, TypedDict

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore


class SizeDistribution(TypedDict):
    """Dictionary mapping set sizes to their frequency."""


class SizeMetrics(TypedDict):
    """Metrics related to the size of prediction sets."""

    mean_set_size: float
    median_set_size: float
    empty_sets_ratio: float
    singleton_sets_ratio: float
    max_set_size: int
    size_distribution: dict[int, float]


class ClassConditionalCoverage(TypedDict):
    """Coverage calculated per class."""


class CoverageMetrics(TypedDict):
    """Metrics related to coverage of conformal predictions."""

    marginal_coverage: float
    class_conditional_coverage: dict[str, float]
    average_class_conditional_coverage: float
    coverage_deviation: float
    worst_class_coverage: float


class EfficiencyMetrics(TypedDict):
    """Metrics related to efficiency of conformal predictions."""

    average_efficiency: float
    n_efficiency: float
    o_efficiency: float
    s_efficiency: float
    u_efficiency: float
    f_efficiency: float


class ClassCalibrationErrors(TypedDict):
    """Calibration errors per class."""


class CalibrationMetrics(TypedDict):
    """Metrics related to calibration of conformal predictions."""

    calibration_error: float
    class_calibration_errors: dict[str, float]
    max_calibration_error: float
    calibration_score: float


class SummaryMetrics(TypedDict):
    """Summary of all conformal prediction metrics."""

    size_metrics: SizeMetrics
    coverage_metrics: CoverageMetrics
    efficiency_metrics: EfficiencyMetrics
    calibration_metrics: CalibrationMetrics
    informativeness_score: float


@dataclass
class ConformalMetrics:
    """Class for calculating and visualizing conformal prediction metrics.

    This class provides methods to evaluate the quality of conformal predictions
    for multiclass classification tasks.

    Attributes:
        predictions_df: DataFrame with binary columns for each class and a true_values column
        class_columns: Names of the columns containing the predicted classes
        true_column: Name of the column containing the true class values
        expected_coverage: Expected coverage level (confidence level) of the conformal predictor
        prediction_sets: Cached prediction sets (computed on first access)

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'Attitude': [False, False, False, False, False],
        ...     'Competitor': [True, False, False, False, False],
        ...     'Dissatisfaction': [False, False, False, False, False],
        ...     'Other': [False, False, False, False, False],
        ...     'Price': [False, False, False, False, False],
        ...     'Stayed': [True, True, True, True, True],
        ...     'true_values': ['Dissatisfaction', 'Stayed', 'Stayed', 'Stayed', 'Stayed']
        ... })
        >>> metrics = ConformalMetrics(data)
        >>> metrics.get_size_metrics()['mean_set_size']
        2.0
    """

    predictions_df: pd.DataFrame
    class_columns: Sequence[str] | None = None
    true_column: str = "true_values"
    expected_coverage: float = 0.9
    _prediction_sets: list[set[str]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize the class columns if not provided."""
        if self.class_columns is None:
            exclude_cols = {self.true_column}
            self.class_columns = [
                col for col in self.predictions_df.columns if col not in exclude_cols
            ]

        if self.true_column not in self.predictions_df.columns:
            raise ValueError(f"True column '{self.true_column}' not found in DataFrame")

        missing_cols = [
            col for col in self.class_columns if col not in self.predictions_df.columns
        ]
        if missing_cols:
            raise ValueError(f"Class columns {missing_cols} not found in DataFrame")

        if not 0 < self.expected_coverage < 1:
            raise ValueError(
                f"Expected coverage must be between 0 and 1, got {self.expected_coverage}"
            )

        self._prediction_sets = self._compute_prediction_sets()

    @property
    def prediction_sets(self) -> list[set[str]]:
        """Get the prediction sets for each sample (cached for efficiency).

        Returns:
            A list of sets, where each set contains the predicted classes for a sample.

        Examples:
            >>> data = pd.DataFrame({
            ...     'Attitude': [False, True],
            ...     'Competitor': [True, False],
            ...     'Stayed': [True, True],
            ...     'true_values': ['Competitor', 'Attitude']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['Attitude', 'Competitor', 'Stayed'])
            >>> metrics.prediction_sets
            [{'Competitor', 'Stayed'}, {'Attitude', 'Stayed'}]
        """
        if self._prediction_sets is None:
            self._prediction_sets = self._compute_prediction_sets()
        return self._prediction_sets

    def _compute_prediction_sets(self) -> list[set[str]]:
        """Internal method to compute prediction sets."""
        prediction_sets = []
        assert self.class_columns

        for _, row in self.predictions_df.iterrows():
            pred_set = {col for col in self.class_columns if row[col]}
            prediction_sets.append(pred_set)

        return prediction_sets

    def get_size_metrics(self) -> SizeMetrics:
        """Calculate metrics related to the size of prediction sets.

        Returns:
            Dictionary with the following metrics:
            - mean_set_size: Average number of classes in the prediction sets
            - median_set_size: Median number of classes in the prediction sets
            - empty_sets_ratio: Ratio of empty prediction sets
            - singleton_sets_ratio: Ratio of prediction sets with exactly one class
            - max_set_size: Maximum size of a prediction set
            - size_distribution: Distribution of set sizes (counts for each size)

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, True],
            ...     'B': [True, True, False],
            ...     'C': [False, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> size_metrics = metrics.get_size_metrics()
            >>> size_metrics['mean_set_size']
            1.6666666666666667
            >>> size_metrics['empty_sets_ratio']
            0.0
        """
        set_sizes = [len(pred_set) for pred_set in self.prediction_sets]

        size_distribution: dict[int, float] = {}
        for size in range(max(set_sizes) + 1):
            size_distribution[size] = sum(1 for s in set_sizes if s == size) / len(
                set_sizes
            )

        return {
            "mean_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
            "median_set_size": float(np.median(set_sizes)) if set_sizes else 0.0,
            "empty_sets_ratio": sum(1 for size in set_sizes if size == 0)
            / len(set_sizes)
            if set_sizes
            else 0.0,
            "singleton_sets_ratio": sum(1 for size in set_sizes if size == 1)
            / len(set_sizes)
            if set_sizes
            else 0.0,
            "max_set_size": max(set_sizes) if set_sizes else 0,
            "size_distribution": size_distribution,
        }

    def get_coverage_metrics(self) -> CoverageMetrics:
        """Calculate coverage metrics for the conformal predictions.

        Returns:
            Dictionary with the following metrics:
            - marginal_coverage: Fraction of samples where the true class is in the prediction set
            - class_conditional_coverage: Coverage calculated per class
            - average_class_conditional_coverage: Average of class-conditional coverages
            - coverage_deviation: Deviation from expected coverage
            - worst_class_coverage: Coverage for the worst-covered class

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, False],
            ...     'B': [False, True, False],
            ...     'C': [True, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> coverage = metrics.get_coverage_metrics()
            >>> coverage['marginal_coverage']
            1.0
        """
        true_values = self.predictions_df[self.true_column].tolist()

        covered = sum(
            1
            for pred_set, true_val in zip(self.prediction_sets, true_values)
            if true_val in pred_set
        )
        marginal_coverage = covered / len(true_values) if true_values else 0.0

        class_coverage: dict[str, float] = {}
        for class_name in set(true_values):
            class_indices = [
                i for i, val in enumerate(true_values) if val == class_name
            ]
            if not class_indices:
                class_coverage[class_name] = np.nan
                continue

            class_covered = sum(
                1 for i in class_indices if true_values[i] in self.prediction_sets[i]
            )
            class_coverage[class_name] = class_covered / len(class_indices)

        valid_coverages = [cov for cov in class_coverage.values() if not np.isnan(cov)]
        avg_class_coverage = float(np.mean(valid_coverages)) if valid_coverages else 0.0
        worst_class_coverage = (
            float(np.min(valid_coverages)) if valid_coverages else 0.0
        )

        coverage_deviation = abs(marginal_coverage - self.expected_coverage)

        return {
            "marginal_coverage": marginal_coverage,
            "class_conditional_coverage": class_coverage,
            "average_class_conditional_coverage": avg_class_coverage,
            "coverage_deviation": coverage_deviation,
            "worst_class_coverage": worst_class_coverage,
        }

    def get_efficiency_metrics(self) -> EfficiencyMetrics:
        """Calculate efficiency metrics for the conformal predictions.

        Efficiency measures how informative the prediction sets are, with smaller
        sets being more informative while maintaining the desired confidence level.

        Returns:
            Dictionary with efficiency metrics including:
            - average_efficiency: Average normalized set size (smaller is better)
            - n_efficiency: N-efficiency (fraction of singleton prediction sets)
            - o_efficiency: O-efficiency (1 - max_set_size/n_classes)
            - s_efficiency: S-efficiency (average ratio of 1/set_size for correct predictions)
            - u_efficiency: U-efficiency (average of reciprocals of all set sizes)
            - f_efficiency: F-efficiency (harmonic mean of precision and recall)

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, True],
            ...     'B': [False, True, False],
            ...     'C': [False, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> efficiency = metrics.get_efficiency_metrics()
            >>> efficiency['s_efficiency'] != efficiency['n_efficiency']  # S-efficiency differs from N-efficiency
            True
            >>> round(efficiency['s_efficiency'], 2) > 0
            True
        """
        assert self.class_columns

        true_values = self.predictions_df[self.true_column].tolist()
        set_sizes = [len(pred_set) for pred_set in self.prediction_sets]
        n_classes = len(self.class_columns)

        non_empty_sizes = [max(1, size) for size in set_sizes]

        n_efficiency = (
            sum(1 for size in set_sizes if size == 1) / len(set_sizes)
            if set_sizes
            else 0.0
        )

        o_efficiency = (
            1 - (max(set_sizes) / n_classes) if set_sizes and n_classes > 0 else 0.0
        )

        s_efficiency_values = [
            1 / len(pred_set)
            for pred_set, true_val in zip(self.prediction_sets, true_values)
            if pred_set and true_val in pred_set
        ]
        s_efficiency = (
            float(np.mean(s_efficiency_values)) if s_efficiency_values else 0.0
        )

        u_efficiency = (
            np.mean([1 / size for size in non_empty_sizes]) if non_empty_sizes else 0.0
        )

        correct_preds = sum(
            1
            for pred_set, true_val in zip(self.prediction_sets, true_values)
            if true_val in pred_set
        )
        precision = np.mean(
            [
                1 / size if true_val in pred_set else 0
                for pred_set, true_val, size in zip(
                    self.prediction_sets, true_values, non_empty_sizes
                )
            ]
        )
        recall = correct_preds / len(true_values) if true_values else 0.0

        f_efficiency = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "average_efficiency": float(
                np.mean([size / n_classes for size in set_sizes])
            )
            if set_sizes and n_classes > 0
            else 0.0,
            "n_efficiency": n_efficiency,
            "o_efficiency": o_efficiency,
            "s_efficiency": s_efficiency,
            "u_efficiency": float(u_efficiency),
            "f_efficiency": float(f_efficiency),
        }

    def get_calibration_metrics(self) -> CalibrationMetrics:
        """Calculate calibration metrics for conformal predictions.

        Returns:
            Dictionary with calibration metrics including:
            - calibration_error: Absolute difference between observed and expected coverage
            - class_calibration_errors: Calibration errors per class
            - max_calibration_error: Maximum absolute calibration error across classes
            - calibration_score: Overall calibration score (lower is better)

        Examples:
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> # Create data where 80% of true values are in prediction sets
            >>> n_samples = 10
            >>> data = pd.DataFrame({
            ...     'A': np.random.choice([True, False], n_samples),
            ...     'B': np.random.choice([True, False], n_samples),
            ...     'true_values': np.random.choice(['A', 'B'], n_samples)
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B'], expected_coverage=0.8)
            >>> calib = metrics.get_calibration_metrics()
            >>> abs(calib['calibration_error']) <= 0.5  # Allow some variance due to random data
            True
        """
        coverage = self.get_coverage_metrics()

        calibration_error = coverage["marginal_coverage"] - self.expected_coverage

        class_calibration_errors = {
            class_name: class_cov - self.expected_coverage
            for class_name, class_cov in coverage["class_conditional_coverage"].items()
            if not np.isnan(class_cov)
        }

        max_calibration_error = (
            max(abs(error) for error in class_calibration_errors.values())
            if class_calibration_errors
            else 0.0
        )

        calibration_score = (
            (
                abs(calibration_error)
                + sum(abs(err) for err in class_calibration_errors.values())
                / len(class_calibration_errors)
            )
            if class_calibration_errors
            else abs(calibration_error)
        )

        return {
            "calibration_error": calibration_error,
            "class_calibration_errors": class_calibration_errors,
            "max_calibration_error": float(max_calibration_error),
            "calibration_score": float(calibration_score),
        }

    def get_confusion_metrics(self) -> tuple[np.ndarray, list[str]]:
        """Calculate a modified confusion matrix for prediction sets.

        For conformal prediction sets, this uses the most confident prediction
        (first class alphabetically in the set) when creating the confusion matrix.

        Returns:
            A tuple containing:
            - Confusion matrix as numpy array
            - List of class names in the order they appear in the matrix

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, False],
            ...     'B': [False, True, False],
            ...     'C': [False, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> conf_matrix, class_names = metrics.get_confusion_metrics()
            >>> conf_matrix.shape
            (3, 3)
        """
        true_values = self.predictions_df[self.true_column].tolist()

        y_pred = []
        for pred_set in self.prediction_sets:
            if not pred_set:
                y_pred.append("__no_prediction__")
            else:
                y_pred.append(sorted(pred_set)[0])

        all_classes = set(true_values) | set(y_pred) - {"__no_prediction__"}
        class_names = sorted(all_classes)

        if "__no_prediction__" in y_pred:
            class_names.append("__no_prediction__")

        conf_matrix = confusion_matrix(true_values, y_pred, labels=class_names)

        return conf_matrix, class_names

    def get_informativeness_score(self) -> float:
        """Calculate an overall informativeness score for the conformal predictor.

        This score balances coverage (reliability) with efficiency (specificity).
        Higher is better, with 1.0 being perfect (singleton sets with perfect coverage).

        Returns:
            Float between 0 and 1 representing overall predictor quality

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, False],
            ...     'B': [False, True, False],
            ...     'C': [True, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> score = metrics.get_informativeness_score()
            >>> 0 <= score <= 1
            True
        """
        coverage = self.get_coverage_metrics()
        efficiency = self.get_efficiency_metrics()

        coverage_quality = (
            1
            - min(abs(coverage["marginal_coverage"] - self.expected_coverage), 0.2)
            / 0.2
        )
        efficiency_quality = efficiency["n_efficiency"]

        score = 0.7 * coverage_quality + 0.3 * efficiency_quality

        return float(score)

    def summarize_metrics(self) -> SummaryMetrics:
        """Get a summary of all conformal prediction metrics.

        Returns:
            Dictionary containing all metrics grouped by category.

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, False],
            ...     'B': [False, True, False],
            ...     'C': [True, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> summary = metrics.summarize_metrics()
            >>> list(summary.keys())
            ['size_metrics', 'coverage_metrics', 'efficiency_metrics', 'calibration_metrics', 'informativeness_score']
        """
        return {
            "size_metrics": self.get_size_metrics(),
            "coverage_metrics": self.get_coverage_metrics(),
            "efficiency_metrics": self.get_efficiency_metrics(),
            "calibration_metrics": self.get_calibration_metrics(),
            "informativeness_score": self.get_informativeness_score(),
        }

    def summarize_metrics_as_df(self) -> pd.DataFrame:
        """Convert summarized metrics to a single-row DataFrame with flattened column names.

        Returns:
            DataFrame with one row containing all metrics with flattened names

        Examples:
            >>> data = pd.DataFrame({
            ...     'A': [True, False, False],
            ...     'B': [False, True, False],
            ...     'C': [True, False, True],
            ...     'true_values': ['A', 'B', 'C']
            ... })
            >>> metrics = ConformalMetrics(data, class_columns=['A', 'B', 'C'])
            >>> df = metrics.summarize_metrics_as_df()
            >>> 'size_metrics_mean_set_size' in df.columns
            True
        """
        result = self.summarize_metrics()
        flat_dict: dict[str, int | float] = {}
        for category, metrics in result.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, int | float):
                        flat_dict[f"{category}_{metric_name}"] = value
            else:
                if isinstance(metrics, int | float):
                    flat_dict[category] = metrics

        return pd.DataFrame([flat_dict])

    def compare_with(self, other: Self) -> pd.DataFrame:
        """Compare metrics with another ConformalMetrics instance.

        Args:
            other: Another ConformalMetrics instance to compare with

        Returns:
            DataFrame with comparison of key metrics between two models
            and their differences

        Examples:
            >>> data1 = pd.DataFrame({
            ...     'A': [True, False, True],
            ...     'B': [False, True, False],
            ...     'true_values': ['A', 'B', 'A']
            ... })
            >>> data2 = pd.DataFrame({
            ...     'A': [True, True, True],
            ...     'B': [False, True, False],
            ...     'true_values': ['A', 'B', 'A']
            ... })
            >>> metrics1 = ConformalMetrics(data1, class_columns=['A', 'B'])
            >>> metrics2 = ConformalMetrics(data2, class_columns=['A', 'B'])
            >>> comparison = metrics1.compare_with(metrics2)
            >>> comparison.shape[1] > 0
            True
        """
        # Get summaries as DataFrames
        summary1 = self.summarize_metrics_as_df()
        summary2 = other.summarize_metrics_as_df()

        # Combine and rename columns
        summary1.columns = [f"model1_{col}" for col in summary1.columns]
        summary2.columns = [f"model2_{col}" for col in summary2.columns]

        result = pd.concat([summary1, summary2], axis=1)

        # Add difference column for numeric values
        for col1 in summary1.columns:
            metric_name = col1.replace("model1_", "")
            col2 = f"model2_{metric_name}"

            if col2 in result.columns and isinstance(
                result[col1].iloc[0], (int, float)
            ):
                result[f"diff_{metric_name}"] = result[col1] - result[col2]

        return result
