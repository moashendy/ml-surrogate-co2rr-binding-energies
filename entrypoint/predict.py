# CLI for running inference
import argparse
from src.inference.inference import predict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/04_predictions/preds.csv')
    args = parser.parse_args()
    predict(args.input, args.output)
