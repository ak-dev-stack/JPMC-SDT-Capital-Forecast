# JPMorgan Chase SDT Progress Analyzer & Scenario Forecaster

## Overview
This is an interactive Python web application that analyzes JPMorgan Chase's progress toward their **$2.5 Trillion Sustainable Development Target (SDT)**. 

Built using **Streamlit**, the app extracts hard data from the *2024 Sustainability Report* to visualize cumulative progress, forecast future trends using machine learning (Prophet), and simulate commercial scenarios such as the "AI Energy Boom."

## Key Features

1.  **Progress Visualization:** 
    -   Tracks the $900B cumulative financing achieved (2021-2024) against the 2030 target.
    -   Breakdown by objective: Green, Development Finance, and Community Development.

2.  **ML Forecasting (Prophet):**
    -   Uses Facebook Prophet to model a logistic growth curve based on historical data points to predict if the $2.5T target will be met under current trends.

3.  **Scenario Simulator:**
    -   Allows users to adjust growth parameters (e.g., impact of AI/Data Center energy demand on financing volumes) to see how external market forces impact the 2030 goal.
    -   *Context:* Based on specific risks cited in the 2024 Report (Page 3 & 23) regarding the trade-off between AI energy demand and Carbon Intensity targets.

4.  **NLP Semantic Analysis:**
    -   Utilizes a Hugging Face `zero-shot-classification` pipeline to categorize text excerpts from the report into strategic pillars (Risk vs. Opportunity vs. Governance).

## Data Source
All data is hardcoded from the **JPMorgan Chase 2024 Sustainability Report** (provided as PDF).
-   **Page 10:** Quantitative progress data (Annual volumes, Cumulative totals).
-   **Page 23:** Qualitative data regarding AI impact on energy targets.

## Technology Stack
-   **Frontend:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **Visualization:** Plotly Express / Graph Objects
-   **Forecasting:** Facebook Prophet
-   **NLP:** Hugging Face Transformers (`facebook/bart-large-mnli`)

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/jpmc-sdt-analyzer.git
    cd jpmc-sdt-analyzer
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Project Insights
This tool demonstrates the intersection of **Sustainable Finance**, **Risk Management**, and **Data Science**. It moves beyond static PDF reporting to dynamic modeling, allowing stakeholders to visualize the non-linear path to 2030 targets.
