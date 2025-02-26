
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import librosa
import numpy as np
import pickle
from sklearn.manifold import TSNE

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
