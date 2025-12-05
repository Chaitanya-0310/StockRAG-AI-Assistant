# StockRAG Backend

This is the Python backend for the StockRAG AI Analyst.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Copy `.env.example` to `.env` and fill in your `GOOGLE_API_KEY` and `DATABASE_URL`.
    Ensure your PostgreSQL database has the `pgvector` extension enabled:
    ```sql
    CREATE EXTENSION vector;
    ```

## Running the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.
Swagger UI documentation is available at `http://localhost:8000/docs`.

## API Endpoints

-   `POST /ingest/{symbol}`: Fetch stock data for a symbol and ingest it into the database (Structured + Vector).
-   `POST /chat`: Chat with the AI about the stock data.

