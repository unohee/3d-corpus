
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import librosa
import numpy as np
import pickle
from multiprocessing import Pool

def process_buffer(buffer):
    audio_data, sample_rate = buffer
    # Perform feature extraction (for example, computing the Mel spectrogram)
    features = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    features_db = librosa.power_to_db(features, ref=np.max)
    return features_db

def featureExtract(path):
    directory_path = path
    buffers_pkl_path = path + '.pkl'
    features_pkl_path = path + 'features_.pkl'

    buffers = []
    features = []

    if os.path.exists(buffers_pkl_path):
        with open(buffers_pkl_path, 'rb') as f:
            buffers = pickle.load(f)
        print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")

        if os.path.exists(features_pkl_path):
            print("Process finished.")
        else:
            # Use ThreadPoolExecutor to parallelize feature extraction
            with ThreadPoolExecutor() as executor:
                future_to_buffer = {executor.submit(process_buffer, buffer): buffer for buffer in buffers}
                
                for future in as_completed(future_to_buffer):
                    try:
                        feature = future.result()
                        features.append(feature)
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
            
            # Save the extracted features
            with open(features_pkl_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Features extracted and saved to {features_pkl_path}.")
    else:
        print(f"No buffers found at {buffers_pkl_path}.")

        
def load_from_pickle(buffers_pkl_path):
    """
    Load buffers from a pickle file.

    Args:
        buffers_pkl_path (str): Path to the pickle file.

    Returns:
        list: Loaded buffers.
    """
    try:
        with open(buffers_pkl_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"No existing buffer file found at {buffers_pkl_path}.")
        return []

def process_audio_files(directory_path):
    """
    Walk through the directory and its subdirectories to find audio files.

    Args:
        directory_path (str): Path to the directory containing audio files.

    Returns:
        list: List of audio file paths.
    """
    audio_files = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                audio_files.append(os.path.join(root, filename))
    return audio_files

def save_to_pickle(buffers, buffers_pkl_path):
    """
    Save buffers to a pickle file.

    Args:
        buffers (list): List of buffers to be saved.
        buffers_pkl_path (str): Path to the pickle file.
    """
    with open(buffers_pkl_path, 'wb') as f:
        pickle.dump(buffers, f)
    print(f"Buffers have been saved to {buffers_pkl_path}.")

def readfile(directory_path, filename):
    """
    Read audio files from a specified directory and store them in a list called buffers.

    Args:
        directory_path (str): Path to the directory containing audio files.
        filename (str): Base name of the pickle file (without extension).

    Returns:
        list: List of loaded buffers.
    """
    buffers = []
    buffers_pkl_path = f"{filename}.pkl"

    # Check if the buffers pickle file exists
    if os.path.exists(buffers_pkl_path):
        buffers = load_from_pickle(buffers_pkl_path)
    else:
        audio_files = process_audio_files(directory_path)

        # Use multiprocessing Pool to parallelize loading of audio files
        with Pool(processes=8) as pool:
            buffers = pool.map(load_audio_file, audio_files)

        save_to_pickle(buffers, buffers_pkl_path)

    return buffers
