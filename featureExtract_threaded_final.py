
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import librosa
import numpy as np
import pickle
from sklearn.manifold import TSNE
from multiprocessing import Pool
import soundfile as sf

def process_buffer(audio_data, sample_rate, filename):
    # Extract MFCC features and calculate mean
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=256)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extract Spectral Centroid and calculate mean
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=512, hop_length=256)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # Extract Chroma feature and calculate mean
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_fft=512, hop_length=256)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Store the extracted features
    feature_data = {
        'filename': filename,
        'mfccs_mean': mfccs_mean.tolist(),
        'spectral_centroid_mean': spectral_centroid_mean,
        'chroma_mean': chroma_mean.tolist()
    }
    
    return feature_data

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
                future_to_buffer = {executor.submit(process_buffer, audio_data, sample_rate, filename): (audio_data, sample_rate, filename)
                                    for audio_data, sample_rate, filename in buffers}
                
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

def load_audio_file(file_path):
    """Helper function to load an audio file."""
    audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    return audio_data, sample_rate

def readfile(directory_path, filename):
    """ 디렉토리에서 오디오 파일을 읽어들여 buffers 리스트에 저장 """
    global buffers

    # Define the paths for the pickle file
    buffers_pkl_path = filename + '.pkl'

    # Initialize the buffers list
    buffers = []

    # Check if the buffers pickle file exists
    if os.path.exists(buffers_pkl_path):
        # Load buffers from the pickle file
        with open(buffers_pkl_path, 'rb') as f:
            buffers = pickle.load(f)
        print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")
    else:
        print(f"No existing buffer file found. Processing audio files from {directory_path}.")

        # Walk through the directory, including all subdirectories
        audio_files = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    audio_files.append(os.path.join(root, filename))
        
        # Use multiprocessing Pool to parallelize loading of audio files
        with Pool(processes=8) as pool:
            buffers = pool.map(load_audio_file, audio_files)

        # Save buffers to a file using pickle
        with open(buffers_pkl_path, 'wb') as f:
            pickle.dump(buffers, f)
        print(f"Buffers have been saved to {buffers_pkl_path}.")

    return buffers