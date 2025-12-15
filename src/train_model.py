import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from data_processing import get_data_from_db

def train_delivery_model():
    print("Loading data...")
    _, orders_df = get_data_from_db()
    
    if orders_df is None:
        print("Failed to load data.")
        return

    print(f"Data Loaded: {orders_df.shape}")
    
    # Feature Engineering
    print("Engineering features...")
    df = orders_df.copy()
    
    # 1. Date Features from Promised Time
    if 'promised_delivery_time' in df.columns:
        df['promised_hour'] = df['promised_delivery_time'].dt.hour
        df['promised_day'] = df['promised_delivery_time'].dt.dayofweek
    else:
        print("Error: 'promised_delivery_time' missing.")
        return

    # 2. Encode Categorical Features (Area)
    if 'area' in df.columns:
        le_area = LabelEncoder()
        df['area_encoded'] = le_area.fit_transform(df['area'].astype(str))
        # Save encoder for inference
        joblib.dump(le_area, 'src/label_encoder_area.pkl')
    else:
        print("Warning: 'area' missing, skipping.")
        df['area_encoded'] = 0

    # Select Features and Target
    features = ['promised_hour', 'promised_day', 'area_encoded', 'order_total']
    target = 'is_late'
    
    # Drop rows with missing features/target
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(X)} samples...")
    print(f"Target Distribution:\n{y.value_counts()}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    model_path = 'src/model.pkl'
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_delivery_model()
