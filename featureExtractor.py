from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
import librosa
import numpy as np
import pickle
from sklearn.manifold import TSNE
from multiprocessing import Pool
import soundfile as sf
import asyncio
from tqdm.auto import tqdm
import gc
import time

def load_audio_file(file_path):
    """
    Load audio file with 16kHz sample rate and normalize it.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        tuple: Normalized audio data and sample rate
    """
    audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    audio_data = librosa.util.normalize(audio_data)
    return audio_data, sample_rate

def detect_onsets_and_split(audio_data, sample_rate, original_filename,
                            output_dir=None, save_splitted_files=False, 
                            min_length_ms=500, onset_threshold_db=-40):
    """
    Detect onsets in normalized audio data and split it into segments.
    
    Args:
        audio_data: Normalized audio data
        sample_rate: Audio sample rate
        original_filename: Original audio file name
        output_dir: Output directory for split files
        save_splitted_files: Whether to save split files
        min_length_ms: Minimum segment length in milliseconds
        onset_threshold_db: Onset detection threshold in dBFS
        
    Returns:
        list: List of split audio segments
    """
    delta_val = np.float32(librosa.db_to_amplitude(onset_threshold_db, ref=1.0))
    min_length_samples = int(sample_rate * (min_length_ms / 1000.0))
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, delta=delta_val)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)  

    splitted_buffers = []
    total_length = len(audio_data)

    for i in tqdm(range(len(onset_samples)), desc='Splitting onsets', leave=False):
        start = onset_samples[i]
        end = onset_samples[i+1] if i+1 < len(onset_samples) else total_length

        chunk_audio = audio_data[start:end]
        if len(chunk_audio) < min_length_samples:
            continue

        chunk_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_onset_{i}.wav"

        if save_splitted_files and output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            chunk_path = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_path, chunk_audio, sample_rate)

        splitted_buffers.append((chunk_audio, sample_rate, chunk_filename, start, end))

    return splitted_buffers

def readfile(directory_path, filename,
             detect_onset=True,
             output_dir=None,
             save_splitted_files=False):
    """
    Read audio files from a directory and create a list of buffers.
    
    Args:
        directory_path: Path to the directory containing audio files
        filename: Name for the dataset file
        detect_onset: Whether to detect onsets
        output_dir: Output directory for split files
        save_splitted_files: Whether to save split files
        
    Returns:
        list: List of processed audio buffers
    """
    start_time = time.time()

    buffers_pkl_path = 'dataset/' + filename + '.pkl'
    buffers = []

    if os.path.exists(buffers_pkl_path):
        with open(buffers_pkl_path, 'rb') as f:
            buffers = pickle.load(f)
        print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")
    else:
        print(f"No existing buffer file found. Processing audio files from {directory_path}.")
        audio_files = []
        for root, _, files in os.walk(directory_path):
            for fname in files:
                if fname.endswith(".wav") or fname.endswith(".mp3"):
                    audio_files.append(os.path.join(root, fname))
        print(f"Found {len(audio_files)} audio files. Starting to load them...")

        with Pool(processes=4) as pool:
            loaded_results = list(tqdm(pool.imap(load_audio_file, audio_files),
                                        total=len(audio_files),
                                        desc="Loading audio files"))
        print(f"Loaded {len(loaded_results)} audio buffers in memory.")

        if detect_onset:
            print("Onset detection is enabled. Splitting audio by detected onsets...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for idx, (audio_data, sample_rate) in enumerate(tqdm(loaded_results, desc="Submitting onset detection tasks")):
                    original_file = audio_files[idx]
                    futures.append(
                        executor.submit(
                            detect_onsets_and_split,
                            audio_data,
                            sample_rate,
                            original_filename=original_file,
                            output_dir=output_dir,
                            save_splitted_files=save_splitted_files
                        )
                    )
                for future in tqdm(as_completed(futures), total=len(futures), desc="Onset Detection"):
                    buffers.extend(future.result())
        else:
            for idx, (audio_data, sample_rate) in enumerate(tqdm(loaded_results, desc="Appending buffers")):
                buffers.append((audio_data, sample_rate, audio_files[idx]))

        with open(buffers_pkl_path, 'wb') as f:
            pickle.dump(buffers, f)
        print(f"Buffers have been saved to {buffers_pkl_path}.")

    end_time = time.time()
    print(f"readfile processing time: {end_time - start_time:.2f} seconds")
    return buffers

def process_buffer(audio_data, sample_rate, filename):
    """
    Extract MFCC, Spectral Centroid, and Chroma features for each audio buffer.
    
    Args:
        audio_data: Audio data
        sample_rate: Audio sample rate
        filename: Audio file name
        
    Returns:
        dict: Dictionary of extracted features
    """
    min_fft = 256
    if np.max(np.abs(audio_data)) < 1e-4 or len(audio_data) < min_fft:
        return {
            'filename': filename,
            'mfccs_mean': [],
            'spectral_centroid_mean': 0.0,
            'chroma_mean': []
        }

    fmax_value = min(sample_rate / 2, 8000)
    n_mels_value = 40

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate,
                                n_mfcc=13, n_fft=256, hop_length=256,
                                fmax=fmax_value, n_mels=n_mels_value)
    mfccs_mean = np.mean(mfccs, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate,
                                                        n_fft=256, hop_length=256)
    spectral_centroid_mean = np.mean(spectral_centroid)

    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate,
                                        n_fft=256, hop_length=256, tuning=0.0)
    chroma_mean = np.mean(chroma, axis=1)

    feature_data = {
        'filename': filename,
        'mfccs_mean': mfccs_mean.tolist(),
        'spectral_centroid_mean': float(spectral_centroid_mean),
        'chroma_mean': chroma_mean.tolist()
    }
    return feature_data

async def async_featureExtract(path):
    """
    Process the entire buffer asynchronously in chunks and save the final results to a single file.
    
    Args:
        path: Name of the dataset
    """
    start_time = time.time()

    buffers_pkl_path = 'dataset/' + path + '.pkl'
    features_pkl_path = 'dataset/' + path + '_features.pkl'

    if not os.path.exists(buffers_pkl_path):
        print(f"No buffers found at {buffers_pkl_path}.")
        return

    with open(buffers_pkl_path, 'rb') as f:
        buffers = pickle.load(f)
    print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")

    if os.path.exists(features_pkl_path):
        print("Process finished.")
        return

    loop = asyncio.get_event_loop()
    chunk_size = 500
    total_chunks = (len(buffers) + chunk_size - 1) // chunk_size
    all_features = []

    for chunk_idx in tqdm(range(total_chunks), desc='Processing chunks'):
        chunk = buffers[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        with ProcessPoolExecutor(max_workers=8) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    process_buffer,
                    audio_data,
                    sr,
                    fname
                )
                for (audio_data, sr, fname, *_) in chunk
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            chunk_features = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Chunk {chunk_idx} Buffer {idx}: Exception - {result}")
                else:
                    chunk_features.append(result)
            all_features.extend(chunk_features)
        
        gc.collect()

    # Save all features to a file
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(all_features, f)

    end_time = time.time()
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    print(f"Extracted features for {len(all_features)} buffers")
    print(f"Saved features to {features_pkl_path}")

def featureExtract(path):
    """
    Process the entire buffer and save the results to a file (synchronous version).
    
    Args:
        path: Name of the dataset
    """
    start_time = time.time()

    os.makedirs('dataset', exist_ok=True)
    buffers_pkl_path = 'dataset/' + path + '.pkl'
    features_pkl_path = 'dataset/' + path + '_features.pkl'

    if not os.path.exists(buffers_pkl_path):
        print(f"No buffers found at {buffers_pkl_path}.")
        return

    with open(buffers_pkl_path, 'rb') as f:
        buffers = pickle.load(f)
    print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")

    if os.path.exists(features_pkl_path):
        print("Process finished.")
        return

    all_features = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for audio_data, sr, fname, *_ in tqdm(buffers, desc="Submitting processing tasks"):
            futures.append(executor.submit(process_buffer, audio_data, sr, fname))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Buffers"):
            try:
                result = future.result()
                all_features.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")
    
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(all_features, f)

    end_time = time.time()
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    print(f"Extracted features for {len(all_features)} buffers")
    print(f"Saved features to {features_pkl_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        readfile(path, os.path.basename(path))

