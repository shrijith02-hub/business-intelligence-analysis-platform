"""Streamlit application entrypoint for the AI BI platform."""
from pathlib import Path
import tempfile
from typing import Optional

import pandas as pd
import streamlit as st

from config.settings import BASE_DIR
from src.analysis.anomaly import AnomalyDetector
from src.analysis.dashboard import DashboardVisualizer
from src.analysis.eda import ExploratoryDataAnalyzer
from src.analysis.forecast import ForecastModel
from src.analysis.kpi import KPIGenerator
from src.data.csv_loader import CSVDataLoader
from src.data.data_cleaner import DataCleaner
from src.reporting.pdf_report import PDFReportGenerator

APP_TITLE = "Business Intelligence Analysis platform"


def render_home_page() -> None:
    st.write(
        "Upload your business dataset to generate dashboards, KPIs, forecasts, anomaly detection reports, and a downloadable PDF report."
    )
    st.info("Get started by uploading a dataset in CSV, XLSX, or XLS format.")


def render_upload_page() -> None:
    st.header("Upload Dataset")
    st.write("Upload a CSV, XLSX, or XLS file to begin your analysis.")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
    )

    if not uploaded_file:
        st.warning("Please upload a dataset to continue.")
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                dataframe = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                dataframe = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            dataframe = pd.read_excel(uploaded_file)

        st.success("Dataset loaded successfully!")
        st.subheader("Data Preview")
        st.dataframe(dataframe.head())

        st.subheader("Dataset Information")
        st.write(f"Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")
        st.write(dataframe.dtypes.astype(str).to_frame("Type"))

        st.session_state["uploaded_dataframe"] = dataframe
    except Exception as error:
        st.error(f"Error loading file: {error}")


def render_data_cleaning_page() -> None:
    st.header("Automatic Data Cleaning")
    if "uploaded_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state["uploaded_dataframe"]
    st.write("The cleaning module will remove duplicates, convert types, fill missing values, and detect outliers.")

    cleaner = DataCleaner(dataframe)

    if st.button("Run Cleaning Pipeline"):
        try:
            cleaned_df, duplicate_report, conversion_report, missing_report, outlier_report = cleaner.create_cleaned_dataframe()
        except Exception as error:
            st.error(f"Data cleaning failed: {error}")
            return

        st.success("Data cleaned successfully.")
        st.subheader("Summary Reports")
        st.write(duplicate_report)
        st.write(conversion_report)
        st.write(missing_report)
        st.write(outlier_report)

        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        st.session_state["cleaned_dataframe"] = cleaned_df


def render_analysis_page() -> None:
    st.header("Exploratory Data Analysis")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    analyzer = ExploratoryDataAnalyzer(dataframe)

    st.write("This module generates summary statistics, correlation matrices, missing value reports, and feature distributions.")

    if st.button("Run EDA"):
        try:
            summary_stats = analyzer.summary_statistics()
            correlation = analyzer.correlation_matrix()
            missing_report = analyzer.missing_value_report()
        except Exception as error:
            st.error(f"EDA generation failed: {error}")
            return

        st.subheader("Summary Statistics")
        st.dataframe(summary_stats)

        st.subheader("Missing Value Report")
        st.dataframe(missing_report)

        if not correlation.empty:
            st.subheader("Correlation Matrix")
            st.dataframe(correlation)
        else:
            st.info("Not enough numeric columns to generate a correlation matrix.")

        st.subheader("Feature Distribution")
        selected_column = st.selectbox("Select a column for distribution", analyzer.available_columns())
        if st.button("Show Distribution"):
            try:
                distribution, distribution_type = analyzer.feature_distribution(selected_column)
            except ValueError as error:
                st.error(f"Unable to generate distribution: {error}")
            else:
                st.write(f"Distribution type: {distribution_type}")
                st.dataframe(distribution)


def render_dashboard_page() -> None:
    st.header("Interactive Dashboard")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    visualizer = DashboardVisualizer(dataframe)

    date_range = None
    if visualizer.date_column is not None:
        available_range = visualizer.available_date_range()
        if available_range is not None:
            start, end = available_range
            selected_dates = st.date_input("Date range", [start.date(), end.date()])
            if len(selected_dates) == 2:
                date_range = (selected_dates[0], selected_dates[1])

    categories = None
    available_categories = visualizer.available_categories()
    if available_categories:
        categories = st.multiselect("Filter categories", available_categories, default=available_categories)

    filtered_df = visualizer.filter_dataframe(date_range=date_range, categories=categories)

    st.subheader("Dashboard Charts")
    sales_chart = visualizer.monthly_sales_chart(filtered_df)
    profit_chart = visualizer.profit_trend_chart(filtered_df)
    category_chart = visualizer.category_sales_chart(filtered_df)
    customer_chart = visualizer.top_customers_chart(filtered_df)
    city_chart = visualizer.city_sales_chart(filtered_df)
    product_chart = visualizer.product_sales_chart(filtered_df)

    if sales_chart is not None:
        st.plotly_chart(sales_chart, use_container_width=True)
    if profit_chart is not None:
        st.plotly_chart(profit_chart, use_container_width=True)
    if category_chart is not None:
        st.plotly_chart(category_chart, use_container_width=True)
    if customer_chart is not None:
        st.plotly_chart(customer_chart, use_container_width=True)
    if city_chart is not None:
        st.plotly_chart(city_chart, use_container_width=True)
    if product_chart is not None:
        st.plotly_chart(product_chart, use_container_width=True)

    if all(chart is None for chart in [sales_chart, profit_chart, category_chart, customer_chart, city_chart, product_chart]):
        st.info("Not enough data available to render dashboard charts.")


def render_kpi_page() -> None:
    st.header("Automatic KPI Dashboard")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    kpi_generator = KPIGenerator(dataframe)

    st.write("This module calculates business KPIs from the dataset.")

    if st.button("Generate KPIs"):
        try:
            kpis = kpi_generator.kpi_summary()
            top_products = kpi_generator.top_products()
            top_categories = kpi_generator.top_categories()
            top_cities = kpi_generator.top_cities()
        except Exception as error:
            st.error(f"KPI generation failed: {error}")
            return

        st.subheader("Key Performance Indicators")
        st.metric("Total Sales", f"{kpis['total_sales']:.2f}" if kpis["total_sales"] is not None else "N/A")
        st.metric("Total Profit", f"{kpis['total_profit']:.2f}" if kpis["total_profit"] is not None else "N/A")
        st.metric("Total Orders", f"{kpis['total_orders']}")
        st.metric("Total Customers", f"{kpis['total_customers']}" if kpis["total_customers"] is not None else "N/A")
        st.metric("Average Order Value", f"{kpis['average_order_value']:.2f}" if kpis["average_order_value"] is not None else "N/A")

        st.subheader("Top Products")
        st.dataframe(top_products)

        st.subheader("Top Categories")
        st.dataframe(top_categories)

        st.subheader("Top Cities")
        st.dataframe(top_cities)


def render_forecasting_page() -> None:
    st.header("Forecasting")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    forecast_model = ForecastModel(dataframe)

    st.write("Generate future forecasts for sales and profit using Prophet.")

    available_metrics = forecast_model.available_metrics()
    metric_options = [metric for metric, column in available_metrics.items() if column is not None]
    selected_metric = st.selectbox("Select metric to forecast", metric_options)
    horizon = st.number_input("Forecast horizon in days", min_value=7, max_value=365, value=30, step=7)

    if st.button("Run Forecast"):
        try:
            forecast_chart = forecast_model.forecast_chart(selected_metric, horizon_days=horizon)
            summary = forecast_model.forecast_summary(selected_metric, horizon_days=horizon)
        except Exception as error:
            st.error(f"Forecasting failed: {error}")
            return

        st.subheader("Forecast Chart")
        st.plotly_chart(forecast_chart, use_container_width=True)

        st.subheader("Forecast Summary")
        st.write(summary)





def render_anomaly_detection_page() -> None:
    st.header("Anomaly Detection")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    detector = AnomalyDetector(dataframe)

    st.write("Detect unusual transactions, sudden sales drops, fraud-like patterns, and numeric outliers.")

    if st.button("Run Anomaly Detection"):
        try:
            report = detector.generate_report()
            summary = detector.summary_counts()
        except Exception as error:
            st.error(f"Anomaly detection failed: {error}")
            return

        st.subheader("Anomaly Summary")
        st.json(summary)

        st.subheader("Unusual Transactions")
        st.dataframe(report["unusual_transactions"])

        st.subheader("Sudden Sales Drops")
        st.dataframe(report["sudden_sales_drops"])

        st.subheader("Fraud-like Activities")
        st.dataframe(report["fraud_like_activities"])

        st.subheader("Numeric Outlier Counts")
        st.json(report["outlier_summary"])


def render_pdf_report_page() -> None:
    st.header("PDF Report Generation")
    if "uploaded_dataframe" not in st.session_state and "cleaned_dataframe" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload Data page.")
        return

    dataframe = st.session_state.get("cleaned_dataframe", st.session_state.get("uploaded_dataframe"))
    st.write("Generate a downloadable PDF report containing KPIs, insights, forecasts, and visualizations.")

    report_title = st.text_input("Report title", "BUSINESS INTELLIGENCE REPORT")
    forecast_model = ForecastModel(dataframe)
    available_metrics = {metric: column for metric, column in forecast_model.available_metrics().items() if column is not None}
    selected_metric = None
    if available_metrics:
        selected_metric = st.selectbox("Select metric for forecast summary", list(available_metrics.keys()))
    horizon = st.number_input("Forecast horizon in days", min_value=7, max_value=365, value=30, step=7)

    if st.button("Generate PDF Report"):
        try:
            kpi_generator = KPIGenerator(dataframe)
            visualizer = DashboardVisualizer(dataframe)

            kpis = kpi_generator.kpi_summary()
            top_products = kpi_generator.top_products()
            top_categories = kpi_generator.top_categories()
            top_cities = kpi_generator.top_cities()

            figures = []
            for chart_fn in [
                visualizer.monthly_sales_chart,
                visualizer.profit_trend_chart,
                visualizer.category_sales_chart,
                visualizer.top_customers_chart,
                visualizer.city_sales_chart,
                visualizer.product_sales_chart,
            ]:
                chart = chart_fn(dataframe)
                if chart is not None:
                    figures.append(chart)

            forecast_summary = {}
            if selected_metric:
                forecast_summary = forecast_model.forecast_summary(selected_metric, horizon_days=horizon)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                report_generator = PDFReportGenerator(report_title, Path(temp_file.name))
            output_path = report_generator.build_report(
                   kpis=kpis,
                   insights={},
                   forecast_summary=forecast_summary,
                   recommendations=[],
                   top_products=top_products,
                   top_categories=top_categories,
                   top_cities=top_cities,
    figures=figures,
)

            pdf_bytes = Path(output_path).read_bytes()
            st.success("PDF report generated successfully.")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="ai_bi_report.pdf",
                mime="application/pdf",
            )
        except Exception as error:
            st.error(f"Report generation failed: {error}")
            return


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.sidebar.header("Navigation")
    menu_options = ["Home", "Upload Data", "Data Cleaning", "Analysis", "Dashboard", "KPIs", "Forecasting", "Anomaly Detection", "PDF Report"]
    selection = st.sidebar.radio("Go to", menu_options)

    if selection == "Home":
        render_home_page()
    elif selection == "Upload Data":
        render_upload_page()
    elif selection == "Data Cleaning":
        render_data_cleaning_page()
    elif selection == "Analysis":
        render_analysis_page()
    elif selection == "Dashboard":
        render_dashboard_page()
    elif selection == "KPIs":
        render_kpi_page()
    elif selection == "Forecasting":
        render_forecasting_page()
    elif selection == "Anomaly Detection":
        render_anomaly_detection_page()
    elif selection == "PDF Report":
        render_pdf_report_page()


if __name__ == "__main__":
    main()
