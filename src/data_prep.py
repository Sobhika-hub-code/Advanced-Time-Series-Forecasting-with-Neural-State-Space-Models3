"""Prepare California Housing dataset and save CSV for experiments."""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import argparse, os

def main(out_csv='data/california_housing.csv', test_size=0.2, random_state=42):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.reset_index(drop=True, inplace=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Saved', out_csv, 'shape=', df.shape)
    # also save a train/test split for quicker runs
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print('Saved train.csv and test.csv')

if __name__ == '__main__':
    main()
