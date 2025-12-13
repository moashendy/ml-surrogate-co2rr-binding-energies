import pandas as pd
import numpy as np

def basic_clean(df):
    # placeholder: drop duplicates, na handling, type casting
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['adsorption_energy'], how='any')
    return df
