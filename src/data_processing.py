import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

def process_data(orders, marketing):
    """Aggregates and merges data for ROAS analysis and Operations enrichment."""
    print("Processing data...")
    
    # 1. Prepare Orders Data
    # Convert order_date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(orders['order_date']):
         orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # Extract date part for daily aggregation
    orders['date'] = orders['order_date'].dt.date
    orders['date'] = pd.to_datetime(orders['date']) # Ensure consistent type for merge

    # Aggregate Revenue by Date
    daily_revenue = orders.groupby('date')['order_total'].sum().reset_index()
    daily_revenue.rename(columns={'order_total': 'total_revenue'}, inplace=True)

    # 2. Prepare Marketing Data
    if not pd.api.types.is_datetime64_any_dtype(marketing['date']):
        marketing['date'] = pd.to_datetime(marketing['date'])
    
    # Aggregate Spend by Date
    daily_spend = marketing.groupby('date')['spend'].sum().reset_index()
    daily_spend.rename(columns={'spend': 'total_spend'}, inplace=True)

    # 3. Merge Data (Time-Series Merge)
    # Using outer join to capture days with spend but no revenue, or revenue but no spend
    merged_df = pd.merge(daily_revenue, daily_spend, on='date', how='outer')

    # 4. Fill NaN values with 0
    merged_df.fillna(0, inplace=True)

    # 5. Calculate ROAS
    # Avoid division by zero
    merged_df['roas'] = merged_df.apply(
        lambda row: row['total_revenue'] / row['total_spend'] if row['total_spend'] > 0 else 0, axis=1
    )

    # 6. Operations Data Enrichment (for predictive model)
    print("Enriching Orders data for Operations...")
    if 'actual_delivery_time' in orders.columns and 'promised_delivery_time' in orders.columns:
        # Ensure datetime
        orders['actual_delivery_time'] = pd.to_datetime(orders['actual_delivery_time'])
        orders['promised_delivery_time'] = pd.to_datetime(orders['promised_delivery_time'])
        
        # Calculate Delay in minutes
        orders['delivery_delay_minutes'] = (orders['actual_delivery_time'] - orders['promised_delivery_time']).dt.total_seconds() / 60
        
        # Create Binary Target: Is_Late (1 if actual > promised, else 0)
        orders['is_late'] = (orders['delivery_delay_minutes'] > 0).astype(int)
        
    return merged_df, orders

def get_data_from_db():
    """Connects to DB, fetches data, and returns processed ROAS and Orders dataframes."""
    # Load environment variables
    load_dotenv()
    
    DB_USER = os.getenv('DB_USERNAME')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME') or os.getenv('DB_DATABASE')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise ValueError("Missing database credentials in .env file.")
        
    try:
        connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_str)
        
        with engine.connect() as connection:
            # Fetch Orders
            try:
                orders_df = pd.read_sql('SELECT * FROM "Order"', connection) 
            except Exception:
                orders_df = pd.read_sql('SELECT * FROM orders', connection)

            # Fetch Marketing
            try:
                marketing_df = pd.read_sql('SELECT * FROM "Market_Performance"', connection)
            except Exception:
                marketing_df = pd.read_sql('SELECT * FROM market_performance', connection)

            # Fetch Customers (for detailed features like Region/Area)
            try:
                customers_df = pd.read_sql('SELECT * FROM "Customers"', connection)
            except Exception:
                customers_df = pd.read_sql('SELECT * FROM customers', connection)
        
        if not orders_df.empty and not marketing_df.empty:
            # Pre-merge Customers to get Area if available
            if not customers_df.empty and 'customer_id' in orders_df.columns and 'customer_id' in customers_df.columns:
                 print("Merging Customers data for feature enrichment...")
                 orders_df = pd.merge(orders_df, customers_df[['customer_id', 'area']], on='customer_id', how='left')
            
            return process_data(orders_df, marketing_df)
        else:
            return None, None
            
    except Exception as e:
        print(f"Database Error: {e}")
        return None, None

def get_feedback_data():
    """Fetches customer feedback data from the database."""
    # Load environment variables
    load_dotenv()
    
    DB_USER = os.getenv('DB_USERNAME')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME') or os.getenv('DB_DATABASE')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
       print("Error: Missing database credentials.")
       return None

    try:
        connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_str)
        
        with engine.connect() as connection:
            try:
                feedback_df = pd.read_sql('SELECT * FROM "Customer_Feedback"', connection)
            except Exception:
                feedback_df = pd.read_sql('SELECT * FROM customer_feedback', connection)
                
        return feedback_df
    except Exception as e:
        print(f"Database Error: {e}")
        return None

if __name__ == "__main__":
    roas_df, enriched_orders = get_data_from_db()
    
    if roas_df is not None:
        print("\nMerged ROAS Data (First 5 rows):")
        print(roas_df[['date', 'total_revenue', 'total_spend', 'roas']].head())
        
        print("\nEnriched Orders Data (First 5 rows):")
        cols = ['order_id', 'promised_delivery_time', 'actual_delivery_time', 'delivery_delay_minutes', 'is_late', 'area']
        existing_cols = [c for c in cols if c in enriched_orders.columns]
        print(enriched_orders[existing_cols].head())
        
    feedback = get_feedback_data()
    if feedback is not None:
        print("\nFeedback Data (First 5 rows):")
        print(feedback[['feedback_text', 'sentiment']].head())

