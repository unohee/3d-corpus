import sys
import os
import asyncio
import requests
import zipfile
import shutil
import argparse
from tqdm import tqdm
from featureExtractor import readfile, async_featureExtract

# FSD50K dataset download constants
ZENODO_RECORD_ID = "4060432"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
FSD50K_FILES = {
    "dev_audio": [
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
        "FSD50K.dev_audio.zip"
    ],
    "eval_audio": [
        "FSD50K.eval_audio.z01",
        "FSD50K.eval_audio.zip"
    ],
    "metadata": [
        "FSD50K.ground_truth.zip",
        "FSD50K.metadata.zip",
        "FSD50K.doc.zip"
    ]
}

def download_file(url, filename, desc=None):
    """
    Download a file from the specified URL and display progress.
    
    Args:
        url: URL to download from
        filename: Path to save the downloaded file
        desc: Description for the progress bar
    """
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            
            tqdm_params = {
                'desc': desc or url,
                'total': total,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

def download_and_extract_fsd50k(base_dir="."):
    """
    Download and extract the FSD50K dataset.
    
    Args:
        base_dir: Base directory to extract files to
    """
    print("Starting FSD50K dataset download and extraction...")
    
    # Create temporary directory
    temp_dir = os.path.join(base_dir, "temp_download")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download metadata files
    for file in FSD50K_FILES["metadata"]:
        file_url = f"{ZENODO_BASE_URL}/{file}?download=1"
        file_path = os.path.join(temp_dir, file)
        
        if not os.path.exists(file_path):
            print(f"Downloading: {file}")
            download_file(file_url, file_path, desc=f"Downloading: {file}")
        
        # Extract
        print(f"Extracting: {file}")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    
    # Download and extract dev_audio files
    if not os.path.exists(os.path.join(base_dir, "FSD50K.dev_audio")):
        for file in FSD50K_FILES["dev_audio"]:
            file_url = f"{ZENODO_BASE_URL}/{file}?download=1"
            file_path = os.path.join(temp_dir, file)
            
            if not os.path.exists(file_path):
                print(f"Downloading: {file}")
                download_file(file_url, file_path, desc=f"Downloading: {file}")
        
        # Merge split zip files
        print("Merging dev_audio files...")
        unsplit_zip = os.path.join(temp_dir, "unsplit_dev.zip")
        os.system(f"zip -s 0 {os.path.join(temp_dir, 'FSD50K.dev_audio.zip')} --out {unsplit_zip}")
        
        # Extract
        print("Extracting dev_audio...")
        with zipfile.ZipFile(unsplit_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    
    # Download and extract eval_audio files
    if not os.path.exists(os.path.join(base_dir, "FSD50K.eval_audio")):
        for file in FSD50K_FILES["eval_audio"]:
            file_url = f"{ZENODO_BASE_URL}/{file}?download=1"
            file_path = os.path.join(temp_dir, file)
            
            if not os.path.exists(file_path):
                print(f"Downloading: {file}")
                download_file(file_url, file_path, desc=f"Downloading: {file}")
        
        # Merge split zip files
        print("Merging eval_audio files...")
        unsplit_zip = os.path.join(temp_dir, "unsplit_eval.zip")
        os.system(f"zip -s 0 {os.path.join(temp_dir, 'FSD50K.eval_audio.zip')} --out {unsplit_zip}")
        
        # Extract
        print("Extracting eval_audio...")
        with zipfile.ZipFile(unsplit_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    
    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print("FSD50K dataset download and extraction completed.")

def check_fsd50k_exists(folder):
    """
    Check if the specified FSD50K folder exists.
    
    Args:
        folder: Path to check
        
    Returns:
        bool: True if the folder exists, False otherwise
    """
    if folder.startswith("./"):
        folder = folder[2:]
    
    # Check if it's one of the standard FSD50K folders
    if folder in ["FSD50K.dev_audio", "FSD50K.eval_audio"]:
        return os.path.exists(folder)
    
    # For user-specified folders
    return True

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='FSD50K Dataset Feature Extraction Tool')
    parser.add_argument('folder', nargs='?', default='./FSD50K.dev_audio',
                        help='Path to the audio folder to process (default: ./FSD50K.dev_audio)')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download the FSD50K dataset without extracting features')
    parser.add_argument('--no-onset', action='store_true',
                        help='Disable onset detection and extract features for entire audio files')
    parser.add_argument('--save-splits', action='store_true',
                        help='Save onset-split audio files to disk')
    parser.add_argument('--output-dir', default='splitted_files',
                        help='Directory to save split audio files (default: splitted_files)')
    
    return parser.parse_args()

def main():
    """Main function to run the feature extraction pipeline."""
    try:
        args = parse_arguments()
        folder = args.folder
        
        # Check if FSD50K dataset exists, download if not
        if not check_fsd50k_exists(folder):
            print(f"FSD50K dataset not found at path '{folder}'.")
            download_choice = input("Do you want to download the FSD50K dataset from Zenodo? (y/n): ")
            if download_choice.lower() == 'y':
                download_and_extract_fsd50k()
            else:
                print("Download canceled. Exiting program.")
                return
        
        # If download-only option is specified
        if args.download_only:
            print("Download completed. Skipping feature extraction due to --download-only option.")
            return
        
        # Determine pickle filenames based on folder name (e.g., dataset/my_folder.pkl, dataset/my_folder_features.pkl)
        folder_name = os.path.basename(os.path.normpath(folder))
        
        print(f"Processing folder: {folder}")
        # 1) Load audio files from folder + onset splitting + save pickle
        readfile(
            directory_path=folder,
            filename=folder_name,
            detect_onset=not args.no_onset,
            output_dir=args.output_dir if args.save_splits else None,
            save_splitted_files=args.save_splits
        )
        
        # 2) Extract features for split buffers (asynchronously)
        asyncio.run(async_featureExtract(folder_name))
    except Exception as e:
        print("Exception occurred during program execution:", e)
        raise e

if __name__ == "__main__":
    main()