"""Forecasting utilities using Prophet for the BI platform."""
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
from prophet import Prophet


class ForecastModel:
    """Generate future sales, revenue, and profit forecasts."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("ForecastModel requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.df = dataframe.copy()
        self.columns = {col.lower(): col for col in self.df.columns}
        self.date_column = self._find_column(["order date", "date", "transaction date", "sale date", "timestamp"])
        self.sales_column = self._find_column(["sales", "revenue", "amount", "total"])
        self.profit_column = self._find_column(["profit", "gross profit", "net profit"])

        if self.date_column is not None:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors="coerce")

    def _find_column(self, candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in self.columns:
                return self.columns[candidate]
        return None

    def _prepare_series(self, metric_column: str) -> pd.DataFrame:
        df = self.df.dropna(subset=[self.date_column, metric_column]).copy()
        if df.empty:
            raise ValueError(f"No valid rows found for metric '{metric_column}'.")

        ts = (
            df[[self.date_column, metric_column]]
            .rename(columns={self.date_column: "ds", metric_column: "y"})
            .groupby("ds", as_index=False)
            .sum()
        )
        if ts.empty:
            raise ValueError(f"Unable to create a time series for '{metric_column}'.")

        return ts

    def available_metrics(self) -> Dict[str, Optional[str]]:
        return {
            "sales": self.sales_column,
            "profit": self.profit_column,
        }

    def _forecast(self, metric_key: str, horizon_days: int = 30) -> Tuple[Prophet, pd.DataFrame]:
        metric_column = self.available_metrics().get(metric_key)
        if metric_column is None:
            raise ValueError(f"Metric '{metric_key}' not found in dataset.")

        if self.date_column is None:
            raise ValueError("Date column not found in dataset.")

        ts = self._prepare_series(metric_column)
        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=horizon_days, freq="D")
        forecast = model.predict(future)
        return model, forecast

    def forecast_metric(self, metric_key: str, horizon_days: int = 30) -> pd.DataFrame:
        _, forecast = self._forecast(metric_key, horizon_days=horizon_days)
        required_columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        return forecast[required_columns].copy()

    def forecast_chart(self, metric_key: str, horizon_days: int = 30) -> px.Figure:
        forecast = self.forecast_metric(metric_key, horizon_days=horizon_days)
        metric_name = metric_key.capitalize()
        return px.line(forecast, x="ds", y="yhat", title=f"{metric_name} Forecast", labels={"ds": "Date", "yhat": metric_name})

    def forecast_summary(self, metric_key: str, horizon_days: int = 30) -> Dict[str, Optional[float]]:
        forecast = self.forecast_metric(metric_key, horizon_days=horizon_days)
        last_values = forecast.tail(horizon_days)
        return {
            "metric": metric_key,
            "forecast_horizon_days": horizon_days,
            "last_forecast_value": float(last_values["yhat"].iloc[-1]) if not last_values.empty else None,
            "average_forecast_value": float(last_values["yhat"].mean()) if not last_values.empty else None,
        }
