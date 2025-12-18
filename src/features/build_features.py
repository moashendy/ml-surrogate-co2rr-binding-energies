import numpy as np
import pandas as pd

def featurize(df, target_col="adsorption_energy"):
    """
    Convert raw dataframe into ML-ready numerical features (X) and target (y).
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # --- Target ---
    y = df[target_col]

    # --- Drop target from features ---
    X = df.drop(columns=[target_col]).copy()

    # --- Keep only numerical columns ---
    X = X.select_dtypes(include=[np.number])

    # --- Example engineered feature ---
    if "d_band_center" in X.columns:
        X["d_band_center_abs"] = X["d_band_center"].abs()

    return X, y
