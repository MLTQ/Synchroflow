import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- Step 1: Define GroupedDataset and TransformerModel classes ---

# Sample dataset
class GroupedDataset(Dataset):
    def __init__(self, data, labels, group_size):
        self.data = data
        self.labels = labels
        self.group_size = group_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        group_idx = idx // self.group_size
        position_in_group = idx % self.group_size
        
        # Positional Encoding
        positional_encoding = self.get_positional_encoding(position_in_group, self.group_size)
        
        # Combine input features and positional encoding
        data_row = self.data[idx]
        input_tensor = np.concatenate((data_row, positional_encoding), axis=-1)
        
        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(self.labels[group_idx], dtype=torch.long)

    def get_positional_encoding(self, pos, group_size):
        # Simple positional encoding function
        encoding = np.zeros(group_size)
        encoding[pos] = 1
        return encoding

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_groups):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 512)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6
        )
        self.fc = nn.Linear(512, num_groups)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# --- Step 2: Load and Preprocess Data from CSV ---

# Load data from CSV
data = pd.read_csv('testdata3.csv').values  # Replace with actual path if needed

# Number of samples and features
num_samples = data.shape[0]
num_features = data.shape[1]  # Should be 17 as per your description
num_groups = 5
group_size = num_samples // num_groups

# Assign labels to each group (assuming 3400 samples per group)
labels = np.repeat(np.arange(num_groups), group_size)

# Dataset and DataLoader
dataset = GroupedDataset(data, labels, group_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Step 3: Train the Model ---

# Model, Loss, Optimizer
model = TransformerModel(input_dim=num_features, num_groups=num_groups)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)
    
    # Print stats
    accuracy = total_correct / total_samples
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}')

# Save the trained PyTorch model
torch.save(model.state_dict(), 'transformer_model.pth')
print("Trained model saved as 'transformer_model.pth'")
