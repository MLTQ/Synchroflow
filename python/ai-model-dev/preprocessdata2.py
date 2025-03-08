import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse

def positional_encoding(data, num_labels):
    """
    Assign positional labels to the dataset such that the first x/num_labels entries
    are labeled 1, the next x/num_labels entries are labeled 2, and so on. Any remaining rows are ignored.

    Args:
        data (numpy.ndarray): The dataset as a NumPy array (rows = samples, columns = features).
        num_labels (int): The number of labels (groups) to split the dataset into.

    Returns:
        numpy.ndarray: The dataset with positional labels.
        numpy.ndarray: The labels for each group.
    """
    # Calculate group size and valid number of samples
    num_samples = data.shape[0]
    group_size = num_samples // num_labels
    valid_samples = group_size * num_labels

    print(f"Group size calculated: {group_size}")
    print(f"Number of valid samples: {valid_samples}")

    # Truncate data to make it evenly divisible
    data = data[:valid_samples]

    # Assign labels to the data
    labels = np.repeat(np.arange(1, num_labels + 1), group_size)

    return data, labels

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply positional encoding to a dataset.")
    parser.add_argument('csv_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('--num_labels', type=int, default=5, help="Number of labels to divide the dataset into (default: 5).")
    args = parser.parse_args()

    # Load data from the provided CSV file (specifying tab as the delimiter)
    csv_file = args.csv_file
    df = pd.read_csv(csv_file, header=None, delimiter='\t')  # Specify tab as the delimiter

    # Normalize the dataset
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)

    # Apply positional labeling
    labeled_data, labels = positional_encoding(data, args.num_labels)

    # Save the labeled data and labels to files
    labeled_data_file = "labeled_data.csv"
    labels_file = "labels.csv"

    pd.DataFrame(labeled_data).to_csv(labeled_data_file, index=False, header=False)
    pd.DataFrame(labels).to_csv(labels_file, index=False, header=False)

    print(f"Positional labeling applied. Labeled data saved to {labeled_data_file} and labels saved to {labels_file}.")
