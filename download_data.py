"""
Temporary script to download Telco Customer Churn dataset from Kaggle.
"""

import os
import sys
from pathlib import Path
import subprocess


def setup_kaggle_credentials():
    """Setup Kaggle credentials from .env file."""
    # Read Kaggle key from .env
    env_path = Path('.env')
    if not env_path.exists():
        print("Error: .env file not found")
        sys.exit(1)
    
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('KAGGLE_KEY='):
                kaggle_key = line.strip().split('=', 1)[1]
                break
        else:
            print("Error: KAGGLE_KEY not found in .env")
            sys.exit(1)
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json with the key
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    # The key format is KGAT_xxx, we need to extract username and key
    # For Kaggle API token, we'll use a simpler approach
    # Since we have the key, let's try using kaggle CLI directly
    
    # Set environment variable
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    print(f"Kaggle credentials configured")
    return kaggle_key


def download_dataset():
    """Download the Telco Customer Churn dataset."""
    print("Downloading Telco Customer Churn dataset...")
    
    # Create data directory
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try using kaggle CLI
        # Note: This requires proper kaggle.json setup
        # For now, we'll provide instructions
        
        dataset_url = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        
        print(f"\nPlease download the dataset manually from:")
        print(f"{dataset_url}")
        print(f"\nAnd place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in: {data_dir.absolute()}")
        print(f"Then rename it to: telco_customer_churn.csv")
        
        # Alternative: Try to download using kaggle API if configured
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', 'blastchar/telco-customer-churn', '-p', str(data_dir), '--unzip'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("Dataset downloaded successfully!")
                
                # Rename the file if needed
                original_file = data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
                target_file = data_dir / 'telco_customer_churn.csv'
                
                if original_file.exists():
                    original_file.rename(target_file)
                    print(f"Dataset saved to: {target_file}")
                    return True
            else:
                print(f"Kaggle CLI error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("\nKaggle CLI not found. Please install it with: pip install kaggle")
            print("Or download the dataset manually from the URL above.")
            return False
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Telco Customer Churn Dataset Downloader")
    print("=" * 60)
    
    # Check if dataset already exists
    target_file = Path('data/raw/telco_customer_churn.csv')
    if target_file.exists():
        print(f"\nDataset already exists at: {target_file}")
        print("Skipping download.")
        sys.exit(0)
    
    setup_kaggle_credentials()
    download_dataset()

