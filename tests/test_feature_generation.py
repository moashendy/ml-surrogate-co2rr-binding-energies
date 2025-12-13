from src.features.build_features import featurize
import pandas as pd

def test_featurize_minimal():
    df = pd.DataFrame({'d_band_center':[ -1.2, 0.3 ]})
    out = featurize(df)
    assert 'd_band_center_abs' in out.columns
