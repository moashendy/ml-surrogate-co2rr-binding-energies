import argparse
import pandas as pd
import torch
from src.models.neural_network import MLPRegressor
import os

def predict(input_csv, output_csv, model_path='models/mlp_best.pth'):
    df = pd.read_csv(input_csv)
    # Minimal assumption: numeric columns are features
    X = df.select_dtypes(include=['number'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPRegressor(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X.values, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out = df.copy()
    out['predicted_adsorption_energy'] = preds
    out.to_csv(output_csv, index=False)
    print('Saved predictions to', output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/04_predictions/preds.csv')
    args = parser.parse_args()
    predict(args.input, args.output)
