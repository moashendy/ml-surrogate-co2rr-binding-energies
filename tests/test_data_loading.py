from src.data.load_data import load_csv
import os
import pandas as pd

def test_load_csv_tmp(tmp_path):
    p = tmp_path / "tmp.csv"
    df = pd.DataFrame({'a':[1,2,3]})
    df.to_csv(p, index=False)
    loaded = load_csv(str(p))
    assert 'a' in loaded.columns
