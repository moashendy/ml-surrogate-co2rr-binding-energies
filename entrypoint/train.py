# Simple CLI entrypoint to start training
import argparse
from src.train.train_model import train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train.yaml')
    args = parser.parse_args()
    train(args.config)
