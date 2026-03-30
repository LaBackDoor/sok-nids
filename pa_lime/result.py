"""Rich result objects for PA-LIME explanations."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExplanationResult:
    """Result of a single PA-LIME explanation.

    Attributes:
        feature_names: Ordered list of feature names matching coefficient indices.
        coefficients: Ridge surrogate coefficients (feature importances). Shape: (d,).
        intercept: Ridge surrogate intercept term.
        r_squared: R-squared fidelity of the surrogate on the weighted neighborhood.
        predicted_class: The class index or label being explained.
        local_prediction: The surrogate's prediction at the query point.
        num_samples: Number of neighborhood samples used.
    """

    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    predicted_class: int | str | None
    local_prediction: float
    num_samples: int

    def top_features(self, k: int = 10, absolute: bool = True) -> list[tuple[str, float]]:
        """Return the top-k most important features by coefficient magnitude.

        Args:
            k: Number of features to return.
            absolute: If True, rank by absolute value. If False, rank by raw value.

        Returns:
            List of (feature_name, coefficient) tuples sorted descending.
        """
        if absolute:
            order = np.argsort(np.abs(self.coefficients))[::-1]
        else:
            order = np.argsort(self.coefficients)[::-1]
        return [(self.feature_names[i], float(self.coefficients[i])) for i in order[:k]]

    def as_dict(self) -> dict[str, float]:
        """Return feature_name -> coefficient mapping."""
        return dict(zip(self.feature_names, self.coefficients.tolist()))
