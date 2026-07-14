"""Anomaly detection utilities for the BI platform."""
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class AnomalyDetector:
    """Detect unusual behavior and outliers in transactional data."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("AnomalyDetector requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.df = dataframe.copy()
        self.columns = {col.lower(): col for col in self.df.columns}
        self.date_column = self._find_column(["order date", "date", "transaction date", "sale date", "timestamp"])
        self.amount_column = self._find_column(["sales", "revenue", "amount", "total"])
        self.profit_column = self._find_column(["profit", "gross profit", "net profit"])
        self.product_column = self._find_column(["product", "product name", "item", "item name"])
        self.customer_column = self._find_column(["customer id", "customer_id", "customerid", "client id", "client"])
        self.city_column = self._find_column(["city", "region", "state", "location"])

        if self.date_column is not None:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors="coerce")

    def _find_column(self, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in self.columns:
                return self.columns[candidate]
        return None

    def _numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def detect_unusual_transactions(self, threshold: float = 3.0) -> pd.DataFrame:
        numeric_cols = self._numeric_columns()
        if not numeric_cols:
            return pd.DataFrame()

        data = self.df[numeric_cols].dropna()
        mean = data.mean()
        std = data.std(ddof=0).replace(0, np.nan)
        z_scores = ((data - mean) / std).abs()
        anomaly_mask = (z_scores > threshold).any(axis=1)

        if not anomaly_mask.any():
            return pd.DataFrame()

        flagged = self.df.loc[anomaly_mask].copy()
        flagged["anomaly_reason"] = "numeric outlier"
        return flagged

    def detect_outliers(self) -> Dict[str, int]:
        numeric_cols = self._numeric_columns()
        counts: Dict[str, int] = {}
        if not numeric_cols:
            return counts

        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty:
                counts[col] = 0
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            counts[col] = int(((series < lower) | (series > upper)).sum())

        return counts

    def detect_sudden_sales_drops(self, threshold_pct: float = 30.0) -> pd.DataFrame:
        if self.date_column is None or self.amount_column is None:
            return pd.DataFrame()

        df = self.df.dropna(subset=[self.date_column, self.amount_column]).copy()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(self.date_column)
        df["month"] = df[self.date_column].dt.to_period("M")
        monthly_sales = df.groupby("month")[self.amount_column].sum().reset_index()
        monthly_sales["pct_change"] = monthly_sales[self.amount_column].pct_change() * 100
        drops = monthly_sales[monthly_sales["pct_change"] <= -threshold_pct]

        if drops.empty:
            return pd.DataFrame()

        drops["month"] = drops["month"].astype(str)
        return drops[["month", self.amount_column, "pct_change"]].copy()

    def detect_fraud_like_behavior(self, amount_threshold: Optional[float] = None, margin_threshold: float = 0.1) -> pd.DataFrame:
        if self.amount_column is None or self.profit_column is None:
            return pd.DataFrame()

        df = self.df.dropna(subset=[self.amount_column, self.profit_column]).copy()
        if df.empty:
            return pd.DataFrame()

        if amount_threshold is None:
            amount_threshold = df[self.amount_column].mean() + 3 * df[self.amount_column].std(ddof=0)

        df["profit_margin"] = df[self.profit_column] / df[self.amount_column].replace(0, np.nan)
        fraud_mask = (df[self.amount_column] >= amount_threshold) & (df["profit_margin"] <= margin_threshold)

        flagged = df.loc[fraud_mask].copy()
        if flagged.empty:
            return pd.DataFrame()

        flagged["anomaly_reason"] = "high amount low margin"
        return flagged

    def summary_counts(self) -> Dict[str, int]:
        return {
            "unusual_transactions": len(self.detect_unusual_transactions()),
            "sudden_sales_drops": len(self.detect_sudden_sales_drops()),
            "fraud_like_activities": len(self.detect_fraud_like_behavior()),
            "numeric_outliers": sum(self.detect_outliers().values()),
        }

    def generate_report(self) -> Dict[str, object]:
        return {
            "unusual_transactions": self.detect_unusual_transactions(),
            "sudden_sales_drops": self.detect_sudden_sales_drops(),
            "fraud_like_activities": self.detect_fraud_like_behavior(),
            "outlier_summary": self.detect_outliers(),
        }
