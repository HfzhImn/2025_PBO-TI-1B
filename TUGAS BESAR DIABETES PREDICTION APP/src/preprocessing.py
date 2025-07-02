import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    assert df.isnull().sum().sum() == 0, "Data mengandung missing value!"
    df['BMI'] = df['BMI'].clip(12, 60)
    return df

def split_data(df):
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
