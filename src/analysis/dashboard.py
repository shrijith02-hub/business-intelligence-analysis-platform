"""Dashboard visualization utilities for the BI platform."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DashboardVisualizer:
    """Create interactive dashboard charts from transactional data."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("DashboardVisualizer requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.dataframe = dataframe.copy()
        self.column_map = self._infer_columns()
        self.date_column = self.column_map.get("date")

        if self.date_column is not None:
            self.dataframe[self.date_column] = pd.to_datetime(
                self.dataframe[self.date_column],
                errors="coerce",
            )

    # ------------------------------------------------------------------
    # Column Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        return (
            str(name)
            .strip()
            .lower()
            .replace("_", " ")
        )

    @staticmethod
    def _find_column(
        columns: Dict[str, str],
        candidates: List[str],
    ) -> Optional[str]:
        for candidate in candidates:
            candidate = candidate.lower()
            if candidate in columns:
                return columns[candidate]

        for candidate in candidates:
            candidate = candidate.lower()
            for key, value in columns.items():
                if candidate in key:
                    return value

        return None

    def _infer_columns(self) -> Dict[str, Optional[str]]:
        columns = {
            self._normalize(col): col
            for col in self.dataframe.columns
        }

        return {
            "sales": self._find_column(
                columns,
                [
                    "sales",
                    "sale",
                    "revenue",
                    "amount",
                    "total",
                    "sales amount",
                ],
            ),
            "profit": self._find_column(
                columns,
                [
                    "profit",
                    "gross profit",
                    "net profit",
                ],
            ),
            "order_id": self._find_column(
                columns,
                [
                    "order id",
                    "orderid",
                    "order_id",
                    "invoice",
                    "transaction id",
                ],
            ),
            "customer_id": self._find_column(
                columns,
                [
                    "customer id",
                    "customerid",
                    "customer_id",
                    "client id",
                    "client",
                ],
            ),
            "product": self._find_column(
                columns,
                [
                    "product",
                    "product name",
                    "item",
                    "item name",
                ],
            ),
            "category": self._find_column(
                columns,
                [
                    "category",
                    "product category",
                    "segment",
                ],
            ),
            "city": self._find_column(
                columns,
                [
                    "city",
                    "state",
                    "region",
                    "location",
                ],
            ),
            "quantity": self._find_column(
                columns,
                [
                    "quantity",
                    "qty",
                    "units",
                ],
            ),
            "date": self._find_column(
                columns,
                [
                    "date",
                    "order date",
                    "transaction date",
                    "sale date",
                    "timestamp",
                ],
            ),
        }

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_by_date_range(
        self,
        dataframe: pd.DataFrame,
        date_range: Optional[Tuple[datetime, datetime]],
    ) -> pd.DataFrame:
        if self.date_column is None:
            return dataframe

        if date_range is None:
            return dataframe

        start_date, end_date = date_range

        return dataframe[
            (dataframe[self.date_column] >= pd.to_datetime(start_date))
            & (dataframe[self.date_column] <= pd.to_datetime(end_date))
        ]

    def _filter_by_category(
        self,
        dataframe: pd.DataFrame,
        categories: Optional[List[str]],
    ) -> pd.DataFrame:
        category_column = self.column_map.get("category")

        if category_column is None:
            return dataframe

        if not categories:
            return dataframe

        return dataframe[
            dataframe[category_column].isin(categories)
        ]

    def filter_dataframe(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        dataframe = self.dataframe.copy()
        dataframe = self._filter_by_date_range(dataframe, date_range)
        dataframe = self._filter_by_category(dataframe, categories)
        return dataframe

    # ------------------------------------------------------------------
    # Available Filters
    # ------------------------------------------------------------------

    def available_categories(self) -> List[str]:
        category_column = self.column_map.get("category")

        if (
            category_column is None
            or category_column not in self.dataframe.columns
        ):
            return []

        return sorted(
            self.dataframe[category_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    def available_date_range(
        self,
    ) -> Optional[Tuple[datetime, datetime]]:
        if (
            self.date_column is None
            or self.date_column not in self.dataframe.columns
        ):
            return None

        dates = self.dataframe[self.date_column].dropna()

        if dates.empty:
            return None

        return dates.min(), dates.max()

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def monthly_sales_chart(
        self,
        dataframe: pd.DataFrame,
    ) -> Optional[go.Figure]:
        sales_column = self.column_map.get("sales")

        if self.date_column is None or sales_column is None:
            return None

        df = dataframe.copy()

        df[self.date_column] = pd.to_datetime(
            df[self.date_column],
            errors="coerce",
        )

        df = df.dropna(
            subset=[self.date_column, sales_column]
        )

        if df.empty:
            return None

        df["Month"] = (
            df[self.date_column]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            df.groupby("Month")[sales_column]
            .sum()
            .reset_index()
        )

        return px.line(
            monthly,
            x="Month",
            y=sales_column,
            markers=True,
            title="Monthly Sales Trend",
        )

    def profit_trend_chart(
        self,
        dataframe: pd.DataFrame,
    ) -> Optional[go.Figure]:
        profit_column = self.column_map.get("profit")

        if self.date_column is None or profit_column is None:
            return None

        df = dataframe.copy()

        df[self.date_column] = pd.to_datetime(
            df[self.date_column],
            errors="coerce",
        )

        df = df.dropna(
            subset=[self.date_column, profit_column]
        )

        if df.empty:
            return None

        df["Month"] = (
            df[self.date_column]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            df.groupby("Month")[profit_column]
            .sum()
            .reset_index()
        )

        return px.line(
            monthly,
            x="Month",
            y=profit_column,
            markers=True,
            title="Monthly Profit Trend",
        )

    def category_sales_chart(
        self,
        dataframe: pd.DataFrame,
    ) -> Optional[go.Figure]:
        category_column = self.column_map.get("category")
        sales_column = self.column_map.get("sales")

        if category_column is None or sales_column is None:
            return None

        df = dataframe.dropna(
            subset=[category_column, sales_column]
        )

        if df.empty:
            return None

        summary = (
            df.groupby(category_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        return px.bar(
            summary,
            x=category_column,
            y=sales_column,
            title="Sales by Category",
        )

    def top_customers_chart(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 10,
    ) -> Optional[go.Figure]:
        customer_column = self.column_map.get("customer_id")
        sales_column = self.column_map.get("sales")

        if customer_column is None or sales_column is None:
            return None

        df = dataframe.dropna(
            subset=[customer_column, sales_column]
        )

        if df.empty:
            return None

        summary = (
            df.groupby(customer_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        return px.bar(
            summary,
            x=customer_column,
            y=sales_column,
            title=f"Top {top_n} Customers by Sales",
        )

    def city_sales_chart(
    self,
    dataframe: pd.DataFrame,) -> Optional[go.Figure]:
        city_column = self.column_map.get("city")
        sales_column = self.column_map.get("sales")
        if city_column is None or sales_column is None:
            return None
        df = dataframe.dropna(
            subset=[city_column, sales_column]
    )
        if df.empty:
            return None
        summary = (
            df.groupby(city_column)[sales_column]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
        return px.bar(
        summary,
        x=city_column,
        y=sales_column,
        title="Top 10 Cities by Sales",
    )
    
    def product_sales_chart(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 10,
    ) -> Optional[go.Figure]:
        product_column = self.column_map.get("product")
        sales_column = self.column_map.get("sales")

        if product_column is None or sales_column is None:
            return None

        df = dataframe.dropna(
            subset=[product_column, sales_column]
        )

        if df.empty:
            return None

        summary = (
            df.groupby(product_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        return px.bar(
            summary,
            x=sales_column,
            y=product_column,
            orientation="h",
            title="Top 10 Products by Sales",
        )