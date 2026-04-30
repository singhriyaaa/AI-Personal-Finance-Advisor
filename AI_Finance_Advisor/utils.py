import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Feature engineering
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    return df


def preprocess(df):
    # Convert category to numeric
    df = pd.get_dummies(df, columns=['category'])

    X = df.drop(['amount', 'date'], axis=1)
    y = df['amount']

    return X, y
