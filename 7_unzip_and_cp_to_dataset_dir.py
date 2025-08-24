#!/usr/bin/env python3
import gzip
import json
import os
import shutil
from pathlib import Path

def unzip_and_copy():
    # Define source and destination directories
    source_dir = Path("dpo_datasets")
    dest_dir = Path("LLaMA-Factory/data")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .json.gz files in source directory
    json_gz_files = list(source_dir.glob("*.json.gz"))
    
    if not json_gz_files:
        print(f"No .json.gz files found in {source_dir}")
        return
    
    print(f"Found {len(json_gz_files)} .json.gz files to process")
    
    # Process each file
    for file_path in json_gz_files:
        output_file = dest_dir / file_path.stem  # removes the .gz extension
        
        print(f"Processing: {file_path} -> {output_file}")
        
        # Uncompress the file
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Successfully uncompressed {file_path.name}")
    
    print(f"All files have been uncompressed and copied to {dest_dir}")

if __name__ == "__main__":
    unzip_and_copy()