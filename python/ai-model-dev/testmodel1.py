import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Define the PositionalEncoding and TransformerModel classes (same as in training)
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

# Dataset class for inference (same as during training)
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

# Load the model weights
model_save_path = "transformer_model.pth"
d_model = 32
input_dim = 8  # Adjust based on your dataset
nhead = 4
num_encoder_layers = 2
num_classes = 5  # Adjust based on your number of classes

model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_classes)
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

# Load and preprocess the new dataset
csv_file = "testdata1.csv"
df = pd.read_csv(csv_file, header=None)  # Adjust if needed for your data
data = df.values

# Normalize the new data using the same scaler from training
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)  # Normalize the data

# Prepare the DataLoader for inference
group_size = df.size // 8 # Same as used during training
print(group_size)
inference_loader = DataLoader(TimeSeriesDataset(data_normalized, group_size), batch_size=1)

# Perform inference
with torch.no_grad():  # No need to track gradients during inference
    for inputs in inference_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class: {predicted.item()}")
