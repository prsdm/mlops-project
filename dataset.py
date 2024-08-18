# dataset.py
import s3fs
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path='config.yml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data_from_source(source_path):
    """Load dataset from an external source such as a data lake or buckets."""
    if source_path.startswith('s3://'):
        # If the source is an S3 bucket
        fs = s3fs.S3FileSystem()
        with fs.open(source_path) as file:
            data = pd.read_csv(file)
    else:
        # For local files
        data = pd.read_csv(source_path)
    return data

def split_data(data, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def save_data(data, path):
    """Save dataset to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)

def main():
    config = load_config()

    # Load data
    raw_data_path = config['data']['raw']
    data = load_data_from_source(raw_data_path)
    print(f"Data loaded from {raw_data_path}")

    # Split data
    train_data, test_data = split_data(data, test_size=config['split']['test_size'])
    print("Data split into training and testing sets")

    # Save split data
    save_data(train_data, config['data']['train'])
    save_data(test_data, config['data']['test'])
    print("Training and testing data saved")

if __name__ == "__main__":
    main()
