# Blinkit Business Intelligence Project

A full-stack data application analyzing Marketing ROI, Delivery Operations, and Customer Feedback.

## Overview
This project integrates Data Engineering, Machine Learning, and GenAI to provide actionable insights for Blinkit:
1.  **Marketing ROI**: Analyzes Return on Ad Spend (ROAS) across channels.
2.  **Delivery Prediction**: ML model to predict "Late" delivery risk.
3.  **Customer Intelligence**: RAG-based chatbot to query customer feedback.

## Requirements
-   Python 3.8+
-   PostgreSQL
-   Mistral API Key (Optional, for GenAI summaries)

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `streamlit`, `pandas`, `sqlalchemy`, `scikit-learn`, `langchain`, `faiss-cpu`, `sentence-transformers` are installed)*

2.  **Database**:
    -   Ensure PostgreSQL is running.
    -   Update `.env` with your DB credentials.

3.  **Data Processing**:
    The dataset is processed from the SQL database:
    ```bash
    python src/data_processing.py
    ```

4.  **Train Model**:
    Train the delivery delay prediction model:
    ```bash
    python src/train_model.py
    ```

## Running the App
Launch the Streamlit dashboard:
```bash
streamlit run src/app.py
```

## Features
-   **Marketing Trends Tab**: Interactive dual-axis chart (Spend vs Revenue).
-   **Risk Calculator Tab**: Input order details to check delivery delay probability.
-   **GenAI Insights Tab**: Ask questions about customer feedback (e.g., "What are common complaints?").
