"""KPI generation utilities for the BI platform."""
from typing import Dict, List, Optional

import pandas as pd


class KPIGenerator:
    """Generate business KPIs from transactional datasets."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("KPIGenerator requires a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

        self.df = dataframe.copy()
        self.column_map = self._infer_columns()

    def _infer_columns(self) -> Dict[str, Optional[str]]:
        """Infer common column names from the dataset."""
        columns = {col.lower(): col for col in self.df.columns}
        mapping = {
            "sales": self._find_column(columns, ["sales", "revenue", "amount", "total"]),
            "profit": self._find_column(columns, ["profit", "gross profit", "net profit"]),
            "order_id": self._find_column(columns, ["order id", "order_id", "orderid", "transaction id", "invoice"]),
            "customer_id": self._find_column(columns, ["customer id", "customer_id", "customerid", "client id", "client"]),
            "product": self._find_column(columns, ["product", "product name", "item", "item name"]),
            "category": self._find_column(columns, ["category", "product category", "segment"]),
            "city": self._find_column(columns, ["city", "region", "state", "location"]),
            "quantity": self._find_column(columns, ["quantity", "qty", "units"]),
        }
        return mapping

    @staticmethod
    def _find_column(columns: Dict[str, str], candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns:
                return columns[candidate]
        return None

    def _sum_column(self, key: str) -> Optional[float]:
        column = self.column_map.get(key)
        if column is None or column not in self.df.columns:
            return None

        return float(self.df[column].dropna().astype(float).sum())

    def total_sales(self) -> Optional[float]:
        return self._sum_column("sales")

    def total_profit(self) -> Optional[float]:
        return self._sum_column("profit")

    def total_revenue(self) -> Optional[float]:
        return self._sum_column("sales") or self._sum_column("profit")

    def total_orders(self) -> int:
        order_column = self.column_map.get("order_id")
        if order_column and order_column in self.df.columns:
            return int(self.df[order_column].nunique())
        return int(len(self.df))

    def total_customers(self) -> Optional[int]:
        customer_column = self.column_map.get("customer_id")
        if customer_column and customer_column in self.df.columns:
            return int(self.df[customer_column].nunique())
        return None

    def average_order_value(self) -> Optional[float]:
        sales = self.total_sales()
        orders = self.total_orders()
        if sales is None or orders == 0:
            return None
        return float(sales / orders)

    def _top_items(self, key: str, metric: str = "sales", top_n: int = 10) -> pd.DataFrame:
        column = self.column_map.get(key)
        metric_column = self.column_map.get(metric)
        if column is None or column not in self.df.columns or metric_column is None or metric_column not in self.df.columns:
            return pd.DataFrame()

        grouped = self.df.groupby(column)[metric_column].sum().sort_values(ascending=False).head(top_n)
        return grouped.reset_index().rename(columns={column: key, metric_column: f"total_{metric}"})

    def top_products(self, top_n: int = 10) -> pd.DataFrame:
        return self._top_items("product", "sales", top_n=top_n)

    def top_categories(self, top_n: int = 10) -> pd.DataFrame:
        return self._top_items("category", "sales", top_n=top_n)

    def top_cities(self, top_n: int = 10) -> pd.DataFrame:
        return self._top_items("city", "sales", top_n=top_n)

    def kpi_summary(self) -> Dict[str, Optional[float]]:
        return {
            "total_sales": self.total_sales(),
            "total_profit": self.total_profit(),
            "total_revenue": self.total_revenue(),
            "total_orders": self.total_orders(),
            "total_customers": self.total_customers(),
            "average_order_value": self.average_order_value(),
        }

    def available_kpis(self) -> Dict[str, Optional[str]]:
        return self.column_map.copy()
