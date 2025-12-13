import yaml
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import load_csv, save_csv
from src.data.preprocess import basic_clean
from src.features.build_features import featurize
from src.models.neural_network import MLPRegressor
from src.models.baseline_model import train_rf, save_model
import os
import numpy as np

def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load raw data (user should place a csv at the path below)
    raw_path = os.path.join(cfg['raw_data_dir'], 'dataset.csv')
    df = load_csv(raw_path)
    df = basic_clean(df)
    df_feat = featurize(df)

    # simple numeric features extraction
    # This assumes the DataFrame already contains numeric columns suitable for ML.
    target = cfg.get('target_column', 'adsorption_energy')
    X = df_feat.select_dtypes(include=[np.number]).drop(columns=[target], errors=False)
    y = df_feat[target].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg.get('val_split',0.2)+cfg.get('test_split',0.1), random_state=cfg.get('random_seed',42))
    # split X_temp into val+test
    val_frac = cfg.get('val_split',0.2) / (cfg.get('val_split',0.2)+cfg.get('test_split',0.1))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_frac, random_state=cfg.get('random_seed',42))

    # Baseline: train random forest
    rf = train_rf(X_train, y_train, X_val, y_val, params={'n_estimators':100, 'n_jobs':-1})
    os.makedirs('models', exist_ok=True)
    save_model(rf, 'models/rf_baseline.joblib')

    # Simple PyTorch training (very minimal example)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPRegressor(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    X_tr = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val = float('inf')
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = criterion(preds, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v)
            val_loss = criterion(val_preds, y_v).item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'models/mlp_best.pth')

    print('Training finished. Best val loss:', best_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train.yaml')
    args = parser.parse_args()
    train(args.config)
