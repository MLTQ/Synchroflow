import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Import tqdm for progress bar
import joblib  # For saving/loading scaler
import os  # To check file existence

# ---------------------------
# Define the Model Components
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.layer_norm = nn.LayerNorm(d_model)  # Layer normalization added for stability

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# ---------------------------
# Define the Dataset Class
# ---------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, data, group_size=10):
        self.data = data
        self.group_size = group_size

    def __len__(self):
        total_groups = len(self.data) // self.group_size
        return total_groups

    def __getitem__(self, idx):
        start_idx = idx * self.group_size
        end_idx = start_idx + self.group_size
        group_data = self.data[start_idx:end_idx]
        return torch.tensor(group_data, dtype=torch.float32)

# ---------------------------
# Function to Load the Model
# ---------------------------

def load_model(model_path, input_dim, d_model, nhead, num_encoder_layers, num_classes):
    if not os.path.isfile(model_path):
        print(f"Error: The model file '{model_path}' does not exist.")
        exit(1)
    model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Ensure compatibility
    model.eval()
    return model

# ---------------------------
# Function to Print Data Statistics
# ---------------------------

def print_data_statistics(data, title):
    """
    Prints the min, max, mean, and std for each feature in the dataset.

    Parameters:
    - data: numpy.ndarray, the data to analyze.
    - title: str, title for the statistics section.
    """
    print(f"\n{title}")
    print(f"{'Feature':<10} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    for i in range(data.shape[1]):
        feature = data[:, i]
        print(f"Feature {i+1:<5} {feature.min():>10.4f} {feature.max():>10.4f} {feature.mean():>10.4f} {feature.std():>10.4f}")

# ---------------------------
# Function to Load and Preprocess Data
# ---------------------------

def load_and_preprocess_data(csv_file, max_entries=10000, group_size_large=1000, group_size_small=100, scaler_path='scaler.save'):
    if not os.path.isfile(csv_file):
        print(f"Error: The file '{csv_file}' does not exist.")
        exit(1)
    
    print("Loading the entire dataset with progress bar...")
    chunksize = 10000  # Number of rows per chunk
    chunks = []
    
    # Estimate total number of rows for better progress estimation (optional)
    try:
        total_rows = sum(1 for _ in open(csv_file)) - 1  # Subtract 1 if there's a header
    except Exception as e:
        print(f"Warning: Could not determine the number of rows. Progress bar may be inaccurate. {e}")
        total_rows = None
    
    try:
        # Read the CSV in chunks with tqdm progress bar
        reader = pd.read_csv(csv_file, header=None, chunksize=chunksize)
        with tqdm(reader, desc="Loading data", total=(total_rows // chunksize) + 1 if total_rows else None) as pbar:
            for chunk in pbar:
                chunks.append(chunk)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)
    
    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)
    data = df.values
    print(f"Dataset loaded with {len(data)} entries.")
    
    # Determine dataset size and adjust accordingly
    if len(data) > max_entries:
        # If dataset has more than 10,000 entries, grab the first 10,000
        data = data[:max_entries]
        group_size = group_size_large
        # print(f"Dataset has more than {max_entries} entries. Grabbing the first {max_entries} entries.")
    else:
        # If dataset has 10,000 entries or fewer, truncate to be evenly divisible by 100
        group_size = group_size_small
        trunc_size = (len(data) // group_size) * group_size  # Largest multiple of 100 less than or equal to len(data)
        data = data[:trunc_size]
        # print(f"Dataset has {len(data)} entries. Truncated to {trunc_size} entries to be divisible by {group_size}.")
    
    # Normalize the data using the same scaler from training
    # Load the scaler if it exists, else fit a new one and save it
    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded scaler from 'scaler.save'.")
    else:
        print(f"Scaler file '{scaler_path}' not found. Fitting a new scaler.")
        scaler = MinMaxScaler()
        scaler.fit(data)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler fitted and saved to '{scaler_path}'.")
    
    # Print scaler parameters
    print("\nScaler Parameters:")
    print(f"Feature-wise Min Values: {scaler.data_min_}")
    print(f"Feature-wise Max Values: {scaler.data_max_}")
    print(f"Feature-wise Scale Factors: {scaler.scale_}")
    print(f"Feature-wise Min (scaled): {scaler.min_}")
    
    # Print data statistics before scaling
    print_data_statistics(data, "Original Data Statistics:")
    
    data_normalized = scaler.transform(data)  # Use transform, not fit_transform
    
    # Print data statistics after scaling
    print_data_statistics(data_normalized, "Scaled Data Statistics:")
    
    return data_normalized, group_size

# ---------------------------
# Function to Perform Inference
# ---------------------------

def perform_inference(model, data_normalized, group_size, batch_size=1):
    total_groups = len(data_normalized) // group_size
    print(f"\nTotal groups for inference: {total_groups}")
    
    dataset = TimeSeriesDataset(data_normalized, group_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    print("\nStarting inference...")
    with torch.no_grad():  # No need to track gradients during inference
        for inputs in tqdm(dataloader, desc="Inference Progress", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    
    print("Inference completed.\n")
    return predictions

# ---------------------------
# Function to Print Specific Predictions
# ---------------------------

def print_specific_predictions(predictions):
    case = predictions[-1]
    if case == 1:
        print("EEG data signals are most similar to Max's EEG data")
    elif case == 2:
        print("EEG data signals are most similar to Rui's EEG data")
    elif case == 3:
        print("EEG data signals are most similar to Page's EEG data")
    elif case == 4:
        print("EEG data signals are most similar to Adam's EEG data")
    elif case == 5:
        print("EEG data signals are most similar to Brian's EEG data")
    else:
        print("EEG data signals are most similar to Max's EEG data")

# ---------------------------
# Function to Save Predictions
# ---------------------------

def save_predictions_to_csv(predictions, filename='inference_predictions.csv'):
    if not predictions:
        print("No predictions to save.")
        return
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Class'])
    predictions_df.to_csv(filename, index=False)
    print(f"\nPredictions have been saved to '{filename}'.")

# ---------------------------
# Main Execution Flow
# ---------------------------

def main():
    # Configuration Parameters
    model_save_path = "transformer_model.pth"
    csv_file = "labeled_data.csv"
    scaler_path = 'scaler.save'
    max_entries = 10000
    group_size_large = 1000
    group_size_small = 100
    batch_size = 1  # Adjust based on your system's capabilities
    save_predictions = True  # Set to False if you don't want to save
    
    # Load the model
    model = load_model(model_save_path, input_dim=8, d_model=32, nhead=4, num_encoder_layers=2, num_classes=5)
    
    # Load and preprocess the data
    data_normalized, group_size = load_and_preprocess_data(
        csv_file=csv_file,
        max_entries=max_entries,
        group_size_large=group_size_large,
        group_size_small=group_size_small,
        scaler_path=scaler_path
    )
    
    # Perform inference
    predictions = perform_inference(model, data_normalized, group_size, batch_size=batch_size)
    
    # Print specific predictions
    print_specific_predictions(predictions)
    
    # Save predictions to CSV if desired
    if save_predictions:
        save_predictions_to_csv(predictions, filename='inference_predictions.csv')

if __name__ == "__main__":
    main()
