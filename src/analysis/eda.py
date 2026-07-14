"""Exploratory data analysis utilities for the BI platform."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ExploratoryDataAnalyzer:
    """Analyze datasets and generate EDA reports."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("ExploratoryDataAnalyzer requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.dataframe = dataframe.copy()

    def summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for numeric and categorical columns."""
        numeric_summary = self.dataframe.describe(include=[np.number]).T
        categorical_summary = self.dataframe.describe(include=[object, "category"]).T
        numeric_summary["type"] = "numeric"
        categorical_summary["type"] = "categorical"
        summary = pd.concat([numeric_summary, categorical_summary], axis=0, sort=False)
        return summary

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Return a correlation matrix for numeric columns."""
        numeric_columns = self.dataframe.select_dtypes(include=[np.number])
        if numeric_columns.shape[1] < 2:
            return pd.DataFrame()

        return numeric_columns.corr(method=method)

    def missing_value_report(self) -> pd.DataFrame:
        """Create a report with missing value counts and percentages."""
        report = pd.DataFrame(
            {
                "missing_count": self.dataframe.isna().sum(),
                "missing_percent": self.dataframe.isna().mean() * 100,
                "unique_count": self.dataframe.nunique(dropna=True),
            }
        )
        return report.sort_values(by="missing_percent", ascending=False)

    def feature_distribution(self, column: str, bins: int = 10) -> Tuple[pd.Series, str]:
        """Generate distribution data for the requested feature."""
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

        series = self.dataframe[column].dropna()
        if series.empty:
            raise ValueError(f"Column '{column}' contains no valid values.")

        if pd.api.types.is_numeric_dtype(series.dtype):
            counts, bin_edges = np.histogram(series, bins=bins)
            labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(counts))]
            distribution = pd.Series(counts, index=labels)
            return distribution, "numeric"

        distribution = series.value_counts().sort_values(ascending=False)
        return distribution, "categorical"

    def available_columns(self) -> List[str]:
        """Return the list of columns available for analysis."""
        return list(self.dataframe.columns)
