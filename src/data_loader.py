import pandas as pd
import os

def load_data(file_name):
    file_path = os.path.join('data', 'raw', file_name)
    return pd.read_csv(file_path)

if __name__ == '__main__':
    df = load_data('train.csv')
    print(df.head())
