import pandas as pd
import matplotlib.pyplot as plt

from shallow_learning.ml_models import Data


def summarize_dataset()->None:
    df = Data.df

    print(df.head())
    print('\n')
    print(df.describe())
    print('\n')
    df.info()
    
    df.hist(bins=50, figsize=(20,15))

    
if __name__ == "__main__":
    summarize_dataset()