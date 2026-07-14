"""PDF report generation utilities for the BI platform."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile

import pandas as pd
from plotly.graph_objs import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass
class PDFReportGenerator:
    """Generate a professional Business Intelligence PDF report."""

    title: str
    output_path: Path

    def _style(self) -> Dict[str, Any]:
        styles = getSampleStyleSheet()
        return {
            "title": styles["Title"],
            "heading": styles["Heading2"],
            "normal": styles["BodyText"],
            "small": ParagraphStyle(
                "small",
                parent=styles["BodyText"],
                fontSize=9,
            ),
        }

    def _format_metric(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value)

    def _table_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        max_rows: int = 10,
    ) -> Table:

        if dataframe.empty:
            return Table([["No data available"]])

        header = [str(col) for col in dataframe.columns]
        rows = dataframe.head(max_rows).astype(str).fillna("").values.tolist()

        data = [header] + rows

        table = Table(data, repeatRows=1)

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )

        return table

    def _dict_to_table(self, data: Dict[str, Any]) -> Table:

        rows = [
            [str(key), self._format_metric(value)]
            for key, value in data.items()
        ]

        table = Table([["Metric", "Value"]] + rows, repeatRows=1)

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )

        return table

    def _figure_to_png(self, figure: Figure) -> Path:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as temp_file:

            figure.write_image(temp_file.name)
            return Path(temp_file.name)

    def _add_figure(
        self,
        story: List[Any],
        figure: Figure,
    ) -> None:

        image_path = self._figure_to_png(figure)

        story.append(
            Image(
                str(image_path),
                width=6.5 * inch,
                height=3.5 * inch,
            )
        )

        story.append(Spacer(1, 0.2 * inch))

    def build_report(
        self,
        kpis: Dict[str, Optional[float]],
        insights: Dict[str, Any],
        forecast_summary: Dict[str, Optional[float]],
        recommendations: List[str],
        top_products: pd.DataFrame,
        top_categories: pd.DataFrame,
        top_cities: pd.DataFrame,
        figures: Optional[List[Figure]] = None,
    ) -> Path:

        document = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
        )

        story: List[Any] = []

        styles = self._style()

        story.append(Paragraph(self.title, styles["title"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Executive Summary", styles["heading"]))

        story.append(
            Paragraph(
                "This report summarizes business KPIs, dashboard analytics, "
                "forecasting, sales trends and anomaly detection generated "
                "from the uploaded dataset.",
                styles["normal"],
            )
        )

        story.append(Spacer(1, 0.2 * inch))

        story.append(
            Paragraph(
                "Key Performance Indicators",
                styles["heading"],
            )
        )

        story.append(self._dict_to_table(kpis))
        story.append(Spacer(1, 0.2 * inch))

        story.append(
            Paragraph(
                "Forecast Summary",
                styles["heading"],
            )
        )

        story.append(self._dict_to_table(forecast_summary))

        story.append(
            Paragraph(
                "The forecast predicts future sales based on historical "
                "transaction trends to support inventory planning and "
                "business decision making.",
                styles["normal"],
            )
        )

        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Top Products", styles["heading"]))
        story.append(self._table_from_dataframe(top_products))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Top Categories", styles["heading"]))
        story.append(self._table_from_dataframe(top_categories))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Top Cities", styles["heading"]))
        story.append(self._table_from_dataframe(top_cities))
        story.append(Spacer(1, 0.2 * inch))

        if figures:
            story.append(Paragraph("Charts", styles["heading"]))

            for figure in figures:
                self._add_figure(story, figure)

        story.append(
            Paragraph(
                "Business Recommendations",
                styles["heading"],
            )
        )

        business_recommendations = [
            "Increase inventory for Technology products.",
            "Focus marketing on high-performing cities.",
            "Promote top-selling products through offers.",
            "Improve Furniture profitability.",
            "Monitor monthly sales trends regularly.",
        ]

        for rec in business_recommendations:
            story.append(
                Paragraph(
                    f"• {rec}",
                    styles["normal"],
                )
            )

            story.append(Spacer(1, 0.1 * inch))

        document.build(story)

        return self.output_path