
import random
import pandas as pd
# You might already have imports for pandas, datetime, etc.
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from datetime import datetime, timedelta

from faker import Faker

faker = Faker()


def generate_transaction_row(
    timestamp: datetime,
    amount_min: float,
    amount_max: float,
    purposes: list,
    countries: list
):
    row = {
      "timestamp": timestamp,
      "amount": float("{0:.2f}".format(random.uniform(amount_min, amount_max))),
      "purpose": (
        random.choice(purposes)
        if purposes
        else random.choice(('Entertainment', 'Holiday', 'Transportation', 'Bills', 'Medical', 'Misc'))
      ),
      "country": random.choice(countries) if countries else faker.country_code('alpha-3')
    }
    return row
    

def generate_timeseries_data(num_rows: int, start_timestamp: datetime, **kwargs):
    data = []
    now = datetime.now()
    timestamp = start_timestamp or datetime.now()
    for _ in range(num_rows):
        timestamp += timedelta(seconds=random.randint(1, 3600))
        params = dict(timestamp=timestamp, **kwargs)
        data.append(generate_transaction_row(**params))
    return pd.DataFrame(data)


# Assuming your existing functions are defined here

def preprocess_data_for_isolation_forest(df):
    # Minimal preprocessing for demonstration
    # Convert 'purpose' and 'country' to categorical codes
    # In a real scenario, consider using OneHotEncoder or similar techniques
    df['purpose_code'] = df['purpose'].astype('category').cat.codes
    df['country_code'] = df['country'].astype('category').cat.codes
    
    # Selecting numerical and encoded features for isolation forest
    features = df[['amount', 'purpose_code', 'country_code']]
    return features

def detect_outliers_with_isolation_forest(df):
    features = preprocess_data_for_isolation_forest(df)
    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(n_estimators=100, contamination='auto')
    clf.fit(features)
    
    # Predict anomalies (-1 for outliers, 1 for inliers)
    preds = clf.predict(features)
    df['is_outlier'] = np.where(preds == -1, 1, 0)  # Mark outliers with 1
    return df
