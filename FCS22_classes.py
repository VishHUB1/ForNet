import pandas as pd
import os
import shutil

# Define paths
excel_file = r"C:\Users\visha\Desktop\forest_research\FSC22\Metadata-20220916T202011Z-001\Metadata\Metadata V1.0 FSC22_.xlsx"  # Path to the Excel file
source_dir = r"C:\Users\visha\Desktop\forest_research\FSC22\Audio Wise V1.0-20220916T202003Z-001\Audio Wise V1.0"  # Replace with your source directory containing WAV files
output_base_dir = "C:/Users/visha/Desktop/forest_research/FCS22_classes"  # Replace with your output directory

# Read the Excel file
df = pd.read_excel(excel_file)

# Ensure output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    dataset_file_name = row['Dataset File Name']
    class_name = row['Class Name']

    # Define the source file path
    source_file_path = os.path.join(source_dir, dataset_file_name)

    # Define the destination directory based on class name
    class_dir = os.path.join(output_base_dir, class_name)

    # Create the class directory if it doesn't exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Define the destination file path
    dest_file_path = os.path.join(class_dir, dataset_file_name)

    # Check if the source file exists and copy it to the destination
    if os.path.exists(source_file_path):
        shutil.copy2(source_file_path, dest_file_path)
        print(f"Copied {dataset_file_name} to {class_dir}")
    else:
        print(f"Source file {dataset_file_name} not found in {source_dir}")

print("File segregation completed.")