"""Automatic data cleaning utilities for the BI platform."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DataCleaner:
    """Clean and prepare data for analytics."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("DataCleaner requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.original = dataframe.copy()
        self.cleaned = dataframe.copy()

    def remove_duplicates(self) -> Tuple[pd.DataFrame, int]:
        """Drop duplicate rows from the DataFrame."""
        before = len(self.cleaned)
        self.cleaned = self.cleaned.drop_duplicates(keep="first")
        removed = before - len(self.cleaned)
        return self.cleaned, removed

    def convert_data_types(self) -> Dict[str, str]:
        """Convert object-like columns to numeric or datetime types when possible."""
        conversion_report: Dict[str, str] = {}

        for column in self.cleaned.columns:
            if self.cleaned[column].dtype == object:
                series = self.cleaned[column].astype(str).replace({"nan": ""})
                numeric_converted = pd.to_numeric(series, errors="coerce")
                if numeric_converted.notna().sum() > len(series) * 0.5:
                    self.cleaned[column] = numeric_converted
                    conversion_report[column] = "numeric"
                    continue

                datetime_converted = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
                if datetime_converted.notna().sum() > len(series) * 0.5:
                    self.cleaned[column] = datetime_converted
                    conversion_report[column] = "datetime"
                    continue

                conversion_report[column] = "unchanged"
            else:
                conversion_report[column] = str(self.cleaned[column].dtype)

        return conversion_report

    def handle_missing_values(self) -> Dict[str, str]:
        """Fill or flag missing values using a sensible default strategy."""
        missing_report: Dict[str, str] = {}

        for column in self.cleaned.columns:
            if self.cleaned[column].isna().sum() == 0:
                missing_report[column] = "no missing values"
                continue

            dtype = self.cleaned[column].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                median_value = self.cleaned[column].median(skipna=True)
                self.cleaned[column] = self.cleaned[column].fillna(median_value)
                missing_report[column] = f"filled numeric with median ({median_value})"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.cleaned[column] = self.cleaned[column].fillna(method="ffill").fillna(method="bfill")
                missing_report[column] = "filled datetime with forward/backward fill"
            else:
                mode_value = self.cleaned[column].mode(dropna=True)
                fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
                self.cleaned[column] = self.cleaned[column].fillna(fill_value)
                missing_report[column] = f"filled categorical with mode ({fill_value})"

        return missing_report

    def detect_outliers(self, threshold: float = 3.0) -> Dict[str, int]:
        """Detect outliers in numeric columns using z-score."""
        numeric_columns = self.cleaned.select_dtypes(include=[np.number]).columns
        outlier_counts: Dict[str, int] = {}

        if len(numeric_columns) == 0:
            return outlier_counts

        numeric_data = self.cleaned[numeric_columns]
        mean = numeric_data.mean()
        std = numeric_data.std(ddof=0).replace(0, np.nan)
        z_scores = ((numeric_data - mean) / std).abs()

        for column in numeric_columns:
            count = int((z_scores[column] > threshold).sum())
            outlier_counts[column] = count

        return outlier_counts

    def create_cleaned_dataframe(self) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, int]]:
        """Run the full cleaning pipeline and return the cleaned DataFrame plus reports."""
        duplicate_report: Dict[str, str] = {}
        _, duplicates_removed = self.remove_duplicates()
        duplicate_report["duplicates_removed"] = str(duplicates_removed)

        conversion_report = self.convert_data_types()
        missing_report = self.handle_missing_values()
        outlier_report = self.detect_outliers()

        return self.cleaned, duplicate_report, conversion_report, missing_report, outlier_report
