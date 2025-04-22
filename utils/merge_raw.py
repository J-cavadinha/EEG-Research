import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path):
    # Read the first CSV file
    df1 = pd.read_csv(file1_path)
    
    # Read the second CSV file
    df2 = pd.read_csv(file2_path)
    
    # Concatenate the dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Files merged successfully. Output saved to: {output_path}")

if __name__ == "__main__":
    # Replace these paths with your actual file paths
    file1_path = "/Users/joaomachado/Desktop/IC_V3_BKP/dataset/raw_1/eeg_data_20250416-235923.csv"
    file2_path = "/Users/joaomachado/Desktop/IC_V3_BKP/eeg_data_20250422-181634.csv"
    output_path = "merged_output_md.csv"
    
    merge_csv_files(file1_path, file2_path, output_path)