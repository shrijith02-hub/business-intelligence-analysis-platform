# AI-Driven Autonomous Business Intelligence Platform

## Overview
This mini project is a modular, production-quality BI platform built with Streamlit for the frontend and Python for the backend. It supports data upload, cleaning, analysis, KPI generation, AI insights, forecasting, anomaly detection, and PDF reporting.

## Project Structure
- `app.py` - main Streamlit entrypoint
- `src/` - Python modules for data handling, analysis, AI, and reporting
- `config/` - configuration and environment setup
- `requirements.txt` - Python dependencies

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows CMD:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run
```bash
streamlit run app.py
```

## Notes
- Use `.env` in `config/` for API keys.
- This repository starts with the core app shell and configuration.
