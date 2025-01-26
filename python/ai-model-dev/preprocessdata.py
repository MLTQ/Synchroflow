import pandas as pd
import argparse
import sys

def process_csv(input_file, output_file, delimiter='\t'):
    try:
        print(f"Reading input CSV file: {input_file} with delimiter '{delimiter}'")
        
        # Read the CSV file without headers and specify the delimiter
        df = pd.read_csv(input_file, header=None, delimiter=delimiter, encoding='utf-8')
        
        # Print the shape of the DataFrame
        print(f"Input CSV Shape: {df.shape}")  # Should be (number_of_rows, number_of_columns)
        
        # Verify the number of columns
        expected_columns = 23  # Update this based on your actual number of columns
        actual_columns = df.shape[1]
        if actual_columns != expected_columns:
            print(f"Warning: Expected {expected_columns} columns, but found {actual_columns} columns in the input CSV.")
        else:
            print(f"Confirmed: Input CSV has {actual_columns} columns as expected.")
        
        # Print the first few rows to verify data
        print("Preview of the first 5 rows of the input CSV:")
        print(df.head())
        
        # Select columns 2 to 9 (0-based indexing: 1 to 8)
        selected_columns = df.columns[1:9]
        print(f"Selected Columns (0-based indexing): {selected_columns.tolist()}")
        
        # Create a new DataFrame with selected columns
        new_df = df[selected_columns]
        
        # Print the shape of the new DataFrame
        print(f"Output CSV Shape (should be number_of_rows x 8): {new_df.shape}")
        
        # Print the first few rows of the new DataFrame
        print("Preview of the first 5 rows of the output DataFrame:")
        print(new_df.head())
        
        # Write the new DataFrame to the output CSV without headers and with tab delimiter
        new_df.to_csv(output_file, index=False, header=False, sep=delimiter)
        print(f"Processed CSV saved to '{output_file}' successfully.")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The input CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Process a tab-delimited CSV file by removing the first column, keeping the next eight columns, and deleting all subsequent columns."
    )
    parser.add_argument('input_csv', help="Path to the input CSV file.")
    parser.add_argument('output_csv', help="Path to save the processed CSV file.")
    parser.add_argument('--delimiter', default='\t', help="Delimiter used in the CSV file (default is tab '\\t').")
    
    args = parser.parse_args()
    
    process_csv(args.input_csv, args.output_csv, args.delimiter)

if __name__ == "__main__":
    main()
