import os
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Pool, current_process, Manager, cpu_count
import soundfile as sf
import asyncio
from tqdm.auto import tqdm
import gc
from functools import partial
import math
import librosa
import torch.nn.functional as TF
import psutil
from contextlib import nullcontext
import time
import sys

# System settings
NUM_CPU_CORES = cpu_count()
BATCH_SIZE = 128
SUB_BATCH_SIZE = 32
NUM_WORKERS = min(NUM_CPU_CORES - 1, 8)

# Minimum window size definitions (in samples)
MIN_WINDOW_MS = 32
MIN_WINDOW_SAMPLES = int(16000 * (MIN_WINDOW_MS / 1000.0))
FADE_MS = 5
FADE_SAMPLES = int(16000 * (FADE_MS / 1000.0))

def get_system_memory():
    """Return system memory in GB."""
    return psutil.virtual_memory().total / (1024 ** 3)

SYSTEM_MEMORY = get_system_memory()
MEMORY_PER_WORKER = SYSTEM_MEMORY / NUM_WORKERS
MAX_CONCURRENT_BATCHES = NUM_WORKERS

def get_device():
    """
    Get the available computation device (CUDA GPU or CPU).
    
    Returns:
        torch.device: The device to use for computations
    """
    if not hasattr(get_device, "device"):
        if torch.cuda.is_available():
            get_device.device = torch.device("cuda")
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            get_device.device = torch.device("cpu")
            print("Using CPU for FFT operations.")
        
        if current_process().name == 'MainProcess':
            print(f"Device: {get_device.device}")
            if get_device.device.type == "cuda":
                print(f"CUDA memory optimization applied (GPU: {torch.cuda.get_device_name(0)})")
                print(f"Available CUDA devices: {torch.cuda.device_count()}")
    return get_device.device

def clear_memory(device):
    """
    Clear memory for the specific device type (CUDA, MPS, or CPU).
    
    Args:
        device: The device to clear memory for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

transforms_cache = {}

def get_transforms(device):
    """
    Get cached audio transforms to avoid recreating them.
    
    Args:
        device: The device to use for transforms
        
    Returns:
        dict: Dictionary of cached transforms
    """
    if device not in transforms_cache:
        transforms_cache[device] = {
            'mfcc': torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=13,
                melkwargs={
                    'n_fft': 256,
                    'hop_length': 256,
                    'n_mels': 13,
                    'f_min': 0,
                    'f_max': 8000
                }
            ).to(device),
            'spectral': torchaudio.transforms.SpectralCentroid(
                sample_rate=16000,
                n_fft=128,
                hop_length=64
            ).to(device),
            'spec': torchaudio.transforms.Spectrogram(
                n_fft=256,
                hop_length=256,
                power=None
            ).to(device)
        }
    return transforms_cache[device]

def apply_replication_padding_and_fade(audio_data, target_length):
    """
    Apply replication padding and fading to audio data.
    
    This function repeats short segments and applies smooth fade-in/out transitions.
    It also safely handles empty audio data.
    
    Args:
        audio_data: Input audio array
        target_length: Desired output length
        
    Returns:
        np.ndarray: Padded audio data
    """
    original_length = len(audio_data)
    
    if original_length == 0:
        return np.zeros(target_length)
    
    if original_length >= target_length:
        return audio_data[:target_length]
    
    padded_audio = np.zeros(target_length)
    padded_audio[:original_length] = audio_data
    
    remaining = target_length - original_length
    repetitions_needed = math.ceil(remaining / original_length)
    
    for i in range(repetitions_needed):
        start_idx = original_length + (i * original_length)
        end_idx = min(start_idx + original_length, target_length)
        copy_length = end_idx - start_idx
        padded_audio[start_idx:end_idx] = audio_data[:copy_length]
    
    fade_samples = min(FADE_SAMPLES, original_length // 4)
    
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        padded_audio[:fade_samples] *= fade_in
        
        fade_out = np.linspace(1, 0, fade_samples)
        padded_audio[-fade_samples:] *= fade_out
        
        for i in range(repetitions_needed):
            crossfade_start = original_length + (i * original_length) - fade_samples
            if crossfade_start >= 0 and crossfade_start + (2 * fade_samples) <= target_length:
                fade_out_curve = np.linspace(1, 0, fade_samples)
                fade_in_curve = np.linspace(0, 1, fade_samples)
                
                crossfade_region = padded_audio[crossfade_start:crossfade_start + (2 * fade_samples)]
                first_half = crossfade_region[:fade_samples] * fade_out_curve
                second_half = crossfade_region[fade_samples:] * fade_in_curve
                
                padded_audio[crossfade_start:crossfade_start + fade_samples] = first_half
                padded_audio[crossfade_start + fade_samples:crossfade_start + (2 * fade_samples)] = second_half
    
    return padded_audio

async def load_single_file(path, fixed_segment_ms=1000, samples_per_segment=4, energy_threshold_dbfs=-60):
    """
    Asynchronously load a single audio file, split it into fixed-length segments,
    and extract samples based on energy levels.
    
    Args:
        path: Path to the audio file
        fixed_segment_ms: Length of each segment in milliseconds
        samples_per_segment: Number of samples to extract per segment
        energy_threshold_dbfs: Energy threshold in dBFS
        
    Returns:
        list: List of extracted audio samples
    """
    try:
        loop = asyncio.get_event_loop()
        waveform, sample_rate = await loop.run_in_executor(None, torchaudio.load, path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform_max = torch.max(torch.abs(waveform))
        if waveform_max > 1e-6:
            waveform = waveform / waveform_max
        
        audio_data = waveform.squeeze().numpy()
        total_samples = len(audio_data)
        
        if total_samples == 0:
            print(f"File '{path}' is empty. Replacing with dummy data.")
            dummy_sample = np.zeros(MIN_WINDOW_SAMPLES)
            sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_dummy.wav"
            return [(dummy_sample, 16000, sample_filename)]
        
        if total_samples < MIN_WINDOW_SAMPLES:
            audio_data = apply_replication_padding_and_fade(audio_data, MIN_WINDOW_SAMPLES)
            total_samples = len(audio_data)
            print(f"File '{path}' is too short and was padded to {MIN_WINDOW_SAMPLES} samples ({MIN_WINDOW_MS}ms).")
        
        segment_samples = int(16000 * (fixed_segment_ms / 1000.0))
        
        segment_energies = []
        segment_positions = []
        
        energy_threshold = 10 ** (energy_threshold_dbfs / 10)
        
        for seg_idx, start in enumerate(range(0, total_samples, segment_samples)):
            end = min(start + segment_samples, total_samples)
            segment_length = end - start
            
            if segment_length == 0:
                continue
            
            if segment_length < MIN_WINDOW_SAMPLES:
                segment_data = apply_replication_padding_and_fade(audio_data[start:end], MIN_WINDOW_SAMPLES)
            else:
                segment_data = audio_data[start:end]
            
            rms_energy = np.mean(segment_data ** 2)
            energy_dbfs = 10 * np.log10(rms_energy) if rms_energy > 0 else -100
            
            segment_energies.append(rms_energy)
            segment_positions.append((seg_idx, start, end))
        
        if not segment_energies:
            segment_data = audio_data
            if len(segment_data) < MIN_WINDOW_SAMPLES:
                segment_data = apply_replication_padding_and_fade(segment_data, MIN_WINDOW_SAMPLES)
            
            sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_full.wav"
            return [(segment_data, 16000, sample_filename)]
        
        max_energy_idx = np.argmax(segment_energies)
        best_seg_idx, start, end = segment_positions[max_energy_idx]
        
        segment_data = audio_data[start:end]
        segment_length = end - start
        
        if segment_length < MIN_WINDOW_SAMPLES:
            segment_data = apply_replication_padding_and_fade(segment_data, MIN_WINDOW_SAMPLES)
            segment_length = MIN_WINDOW_SAMPLES
        
        window_size = MIN_WINDOW_SAMPLES
        window_energies = []
        window_positions = []
        
        for window_start in range(0, segment_length - window_size + 1, window_size // 2):
            window_end = window_start + window_size
            window = segment_data[window_start:window_end]
            energy = np.mean(window ** 2)
            window_energies.append(energy)
            window_positions.append(start + window_start)
        
        if not window_energies:
            best_pos = start
        else:
            max_window_idx = np.argmax(window_energies)
            best_pos = window_positions[max_window_idx]
        
        end_pos = min(best_pos + MIN_WINDOW_SAMPLES, total_samples)
        if end_pos - best_pos < MIN_WINDOW_SAMPLES:
            sample = apply_replication_padding_and_fade(audio_data[best_pos:end_pos], MIN_WINDOW_SAMPLES)
        else:
            sample = audio_data[best_pos:end_pos]
        
        sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_best.wav"
        return [(sample, 16000, sample_filename)]
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        dummy_sample = np.zeros(MIN_WINDOW_SAMPLES)
        sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_dummy.wav"
        print(f"Error processing {path}, using dummy data instead.")
        return [(dummy_sample, 16000, sample_filename)]

async def load_audio_batch(file_paths):
    """
    Asynchronously load audio files in batches.
    
    Args:
        file_paths: List of paths to audio files
        
    Returns:
        list: List of extracted audio samples
    """
    tasks = [load_single_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    
    all_segments = []
    for segments in results:
        if segments is not None:
            all_segments.extend(segments)
    
    return all_segments

async def process_audio_batch(audio_batch, device):
    if not audio_batch:
        return []
    
    max_len = max(len(audio_data) for audio_data, _ in audio_batch)
    
    audio_arrays = [audio_data for audio_data, _ in audio_batch]
    
    audio_tensors = []
    for audio_data in audio_arrays:
        tensor = torch.from_numpy(audio_data).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        pad_size = max_len - tensor.size(-1)
        if pad_size > 0:
            tensor = TF.pad(tensor, (0, pad_size))
        
        rms = torch.sqrt(torch.mean(tensor ** 2))
        if rms > 1e-6:
            tensor = tensor / rms
        
        audio_tensors.append(tensor)
    
    batch_tensor = torch.stack(audio_tensors).to(device, non_blocking=True)
    
    transforms = get_transforms(device)
    
    async def extract_features():
        with torch.no_grad():
            if device.type == 'mps':
                batch_tensor_cont = batch_tensor.contiguous()
            else:
                batch_tensor_cont = batch_tensor
                
            num_sub_batches = math.ceil(len(batch_tensor_cont) / SUB_BATCH_SIZE)
            
            all_mfccs = []
            all_spectral = []
            all_specs = []
            
            for i in range(num_sub_batches):
                start_idx = i * SUB_BATCH_SIZE
                end_idx = min((i + 1) * SUB_BATCH_SIZE, len(batch_tensor_cont))
                sub_batch = batch_tensor_cont[start_idx:end_idx]
                
                mfccs = transforms['mfcc'](sub_batch)
                mfccs_mean = torch.mean(mfccs, dim=2)
                all_mfccs.append(mfccs_mean.to(device))
                
                sub_batch_processed = torch.where(
                    torch.abs(sub_batch) < 1e-6,
                    torch.zeros_like(sub_batch),
                    sub_batch
                )
                
                sub_batch_processed = sub_batch_processed - torch.mean(sub_batch_processed, dim=-1, keepdim=True)
                
                spectral = transforms['spectral'](sub_batch_processed)
                
                spectral = torch.where(
                    torch.isnan(spectral),
                    torch.zeros_like(spectral),
                    spectral
                )
                
                all_spectral.append(spectral.to(device))
                
                spec = transforms['spec'](sub_batch)
                all_specs.append(spec.to(device))
                
                del mfccs, mfccs_mean, spectral, spec
                if device.type == 'mps':
                    torch.mps.empty_cache()
            
            mfccs = torch.cat(all_mfccs, dim=0)
            spectral = torch.cat(all_spectral, dim=0)
            spec = torch.cat(all_specs, dim=0)
            
            del all_mfccs, all_spectral, all_specs
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            magnitude = torch.abs(spec)
            magnitude = magnitude.squeeze(1)
            
            n_fft = 256
            sample_rate = 16000
            
            bin_frequencies = torch.arange(magnitude.shape[1], device=device) * (sample_rate / n_fft)
            
            min_freq = 32.7
            max_freq = 4186.0
            valid_freq_mask = (bin_frequencies >= min_freq) & (bin_frequencies <= max_freq)
            
            valid_frequencies = bin_frequencies[valid_freq_mask]
            magnitude_valid = magnitude[:, valid_freq_mask, :]
            
            midi_notes = 12 * torch.log2(valid_frequencies / 440.0) + 69
            
            chroma_bins = torch.remainder(midi_notes.round(), 12).long()
            
            chroma = torch.zeros(len(batch_tensor_cont), 12, magnitude.shape[-1], device=device)
            
            for note in range(12):
                note_mask = (chroma_bins == note)
                if note_mask.any():
                    weights = 1.0 / (1.0 + torch.exp((midi_notes[note_mask] - 69) / 12))
                    weights = weights.view(1, -1, 1)
                    matching_magnitudes = magnitude_valid[:, note_mask, :]
                    weighted_magnitudes = matching_magnitudes * weights
                    chroma[:, note, :] = torch.sum(weighted_magnitudes, dim=1)
            
            chroma_sum = torch.sum(chroma, dim=1, keepdim=True)
            eps = 1e-6
            chroma = torch.where(
                chroma_sum > eps,
                chroma / (chroma_sum + eps),
                torch.zeros_like(chroma)
            )
            
            chroma_means = torch.mean(chroma, dim=2)
            
            del magnitude, magnitude_valid, bin_frequencies, valid_frequencies, midi_notes, chroma_bins, chroma, chroma_sum
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            return mfccs, spectral, chroma_means
    
    mfccs, spectral, chroma_means = await extract_features()
    
    cpu_mfcc = mfccs.cpu()
    cpu_spectral = spectral.cpu()
    cpu_chroma = chroma_means.cpu()
    
    del mfccs, spectral, chroma_means
    if device.type == 'mps':
        torch.mps.empty_cache()
        
    results = []
    for i in range(len(audio_batch)):
        mfcc_data = cpu_mfcc[i].numpy()
        
        if mfcc_data.ndim > 1:
            if mfcc_data.ndim == 3:
                mfcc_data = np.mean(mfcc_data.squeeze(0), axis=1)
            elif mfcc_data.ndim == 2:
                mfcc_data = np.mean(mfcc_data, axis=1)
        
        spectral_data = cpu_spectral[i].numpy()
        if spectral_data.ndim > 1:
            spectral_data = np.mean(spectral_data)
        elif spectral_data.ndim == 0:
            spectral_data = float(spectral_data)
        
        chroma_data = cpu_chroma[i].numpy()
        if chroma_data.ndim > 1:
            chroma_data = np.mean(chroma_data, axis=1)
        
        results.append({
            'mfccs_mean': mfcc_data,
            'spectral_centroid_mean': spectral_data,
            'chroma_mean': chroma_data
        })
    
    del batch_tensor, audio_tensors, cpu_mfcc, cpu_spectral, cpu_chroma
    if device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    return results

async def process_features_async(buffers):
    """
    Process all buffers asynchronously.
    Optimized for MPS utilization.
    
    Args:
        buffers: List of buffers to process
        
    Returns:
        list: List of processed features
    """
    device = get_device()
    all_features = []
    
    excessive_onset_files = []
    
    total_batches = math.ceil(len(buffers) / BATCH_SIZE)
    
    if device.type == 'mps':
        concurrent_batches = min(MAX_CONCURRENT_BATCHES, 8)
        print(f"MPS device detected: optimizing concurrent batch count to {concurrent_batches}")
    else:
        concurrent_batches = MAX_CONCURRENT_BATCHES
    
    print(f"Concurrent batch count: {concurrent_batches} (per worker memory: {MEMORY_PER_WORKER:.1f}GB)")
    
    async def process_batch_group(start_batch_idx, pbar):
        group_results = []
        group_tasks = []
        
        for batch_idx in range(start_batch_idx, min(start_batch_idx + concurrent_batches, total_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(buffers))
            batch = buffers[start_idx:end_idx]
            
            audio_batch = [(audio_data, sr) for audio_data, sr, _, _ in batch]
            filenames = [fname for _, _, fname, _ in batch]
            metadata_list = [metadata for _, _, _, metadata in batch]
            
            task = asyncio.create_task(process_audio_batch(audio_batch, device))
            group_tasks.append((batch_idx, filenames, metadata_list, task))
        
        try:
            completed_tasks = await asyncio.gather(*(task for _, _, _, task in group_tasks))
            
            for (batch_idx, filenames, metadata_list, _), batch_results in zip(group_tasks, completed_tasks):
                batch_features = []
                for filename, metadata, result in zip(filenames, metadata_list, batch_results):
                    feature_data = {'filename': filename}
                    feature_data.update(result)
                    feature_data.update(metadata)
                    
                    if metadata['excessive_onsets']:
                        original_filename = os.path.basename(filename).split('_onset_')[0]
                        if original_filename not in excessive_onset_files:
                            excessive_onset_files.append(original_filename)
                    
                    batch_features.append(feature_data)
                    file_name = os.path.basename(filename)
                    pbar.set_postfix({'Current file': file_name})
                    pbar.update(1)
                group_results.append((batch_idx, batch_features))
                
        except Exception as e:
            print(f"Error in batch group processing: {e}")
            for batch_idx, _, _, _ in group_tasks:
                group_results.append((batch_idx, []))
        
        clear_memory(device)
        
        return group_results
    
    with tqdm(total=len(buffers), desc='Feature extraction progress', unit='files') as pbar:
        for start_idx in range(0, total_batches, concurrent_batches):
            group_results = await process_batch_group(start_idx, pbar)
            
            group_results.sort(key=lambda x: x[0])
            
            for _, batch_features in group_results:
                all_features.extend(batch_features)
            
            gc.collect()
            clear_memory(device)
    
    if excessive_onset_files:
        print("\nExcessive onset detection occurred in files:")
        for i, filename in enumerate(excessive_onset_files, 1):
            print(f"{i}. {filename}")
        print(f"Total {len(excessive_onset_files)} files had excessive onset detection.")
    
    return all_features

async def async_featureExtract(path):
    """
    Process all buffers asynchronously.
    Optimized for MPS utilization.
    
    Args:
        path: Path to the dataset folder or feature file
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
    
    all_features = await process_features_async(buffers)
    
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Features extracted and saved to {features_pkl_path}.")
    
    end_time = time.time()
    print(f"async_featureExtract processing time: {end_time - start_time:.2f} seconds")

def check_features(features_pkl_path):
    """
    Analyze and verify feature extraction data statistically.
    - Basic statistics (mean, standard deviation, median, percentiles, etc.)
    - Outlier detection
    - Feature correlation analysis
    - Distribution analysis
    - Analysis of zeros and NaNs
    
    Args:
        features_pkl_path: Path to the feature file
    """
    try:
        with open(features_pkl_path, 'rb') as f:
            features = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {features_pkl_path}")
        return

    print(f"\nFeature extraction data statistical analysis results:")
    print(f"Total audio unit count: {len(features)}")

    if not features or not isinstance(features, list) or not isinstance(features[0], dict):
        print("Error: Data structure is not as expected.")
        return

    stats = {}
    for key in features[0].keys():
        if key not in ['filename', 'excessive_onsets', 'original_onset_count', 'onsets_per_second', 'audio_duration']:
            stats[key] = {
                'values': [],
                'nan_count': 0,
                'zero_count': 0,
                'total_elements': 0,
                'shape_counts': {},
                'percentiles': {},
                'outliers': [],
            }

    for idx, feature in enumerate(features):
        for key, value in feature.items():
            if key in stats:
                if isinstance(value, np.ndarray):
                    shape_str = str(value.shape)
                    stats[key]['shape_counts'][shape_str] = stats[key]['shape_counts'].get(shape_str, 0) + 1
                    
                    flat_value = value.flatten()
                    stats[key]['values'].extend(flat_value)
                    stats[key]['nan_count'] += np.isnan(flat_value).sum()
                    stats[key]['zero_count'] += np.sum(np.abs(flat_value) < 1e-6)
                    stats[key]['total_elements'] += flat_value.size

    print("\n=== Detailed statistical analysis by feature ===")
    for key in stats.keys():
        values = np.array(stats[key]['values'])
        non_nan_values = values[~np.isnan(values)]
        
        print(f"\n{key}:")
        print(f"  1. Basic information:")
        print(f"    - Total element count: {stats[key]['total_elements']:,}")
        print(f"    - NaN ratio: {(stats[key]['nan_count'] / stats[key]['total_elements'] * 100):.2f}%")
        print(f"    - Zero ratio: {(stats[key]['zero_count'] / stats[key]['total_elements'] * 100):.2f}%")
        
        if len(non_nan_values) > 0:
            percentiles = np.percentile(non_nan_values, [0, 25, 50, 75, 100])
            mean = np.mean(non_nan_values)
            std = np.std(non_nan_values)
            
            print(f"  2. Distribution statistics:")
            print(f"    - Mean: {mean:.3f}")
            print(f"    - Standard deviation: {std:.3f}")
            print(f"    - Median: {percentiles[2]:.3f}")
            print(f"    - Minimum: {percentiles[0]:.3f}")
            print(f"    - Maximum: {percentiles[4]:.3f}")
            print(f"    - IQR: {(percentiles[3] - percentiles[1]):.3f}")
            
            q1, q3 = percentiles[1], percentiles[3]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = non_nan_values[(non_nan_values < lower_bound) | (non_nan_values > upper_bound)]
            outlier_ratio = len(outliers) / len(non_nan_values) * 100
            
            print(f"  3. Outlier analysis:")
            print(f"    - Outlier ratio: {outlier_ratio:.2f}%")
            print(f"    - Outlier range: [{lower_bound:.3f}, {upper_bound:.3f}] outside")
            
            skewness = np.mean(((non_nan_values - mean) / std) ** 3) if std > 0 else 0
            kurtosis = np.mean(((non_nan_values - mean) / std) ** 4) - 3 if std > 0 else 0
            
            print(f"  4. Distribution characteristics:")
            print(f"    - Skewness: {skewness:.3f}")
            print(f"    - Kurtosis: {kurtosis:.3f}")
        
        print(f"  5. Data type:")
        for shape, count in stats[key]['shape_counts'].items():
            print(f"    - Shape {shape}: {count:,} ({count/len(features)*100:.1f}%)")

    feature_means = {}
    for key in stats.keys():
        if len(stats[key]['values']) > 0:
            feature_means[key] = []
            for feature in features:
                if isinstance(feature[key], np.ndarray):
                    if key == 'mfccs_mean':
                        mean_value = np.mean(feature[key])
                    elif key == 'spectral_centroid_mean':
                        mean_value = np.mean(feature[key])
                    elif key == 'chroma_mean':
                        mean_value = np.mean(feature[key])
                    feature_means[key].append(mean_value)
    
    if len(feature_means) >= 2:
        correlation_matrix = np.zeros((len(feature_means), len(feature_means)))
        feature_keys = list(feature_means.keys())
        
        print("\nFeature mean value statistics:")
        for key in feature_keys:
            values = np.array(feature_means[key])
            print(f"\n{key}:")
            print(f"  - Mean: {np.mean(values):.3f}")
            print(f"  - Standard deviation: {np.std(values):.3f}")
            print(f"  - Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        
        print("\nCorrelation matrix:")
        for i, key1 in enumerate(feature_keys):
            print(f"\n{key1}:")
            for j, key2 in enumerate(feature_keys):
                if i != j:
                    correlation = np.corrcoef(
                        feature_means[key1],
                        feature_means[key2]
                    )[0, 1]
                    correlation_matrix[i, j] = correlation
                    strength = "Strong" if abs(correlation) > 0.7 else "Medium" if abs(correlation) > 0.3 else "Weak"
                    direction = "Positive" if correlation > 0 else "Negative"
                    print(f"  - Correlation coefficient with {key2}: {correlation:.3f} ({strength} {direction} correlation)")

    print("\n=== Detailed structure of the first sample ===")
    first_feature = features[0]
    for key, value in first_feature.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    - shape: {value.shape}")
            print(f"    - dtype: {value.dtype}")
            print(f"    - mean: {np.nanmean(value):.3f}")
            print(f"    - standard deviation: {np.nanstd(value):.3f}")
        else:
            print(f"  {key}: {value}")

def convert_features(input_path, output_path=None):
    """
    Convert feature file data to a consistent 1D array format.
    
    Conversion rules:
    - mfccs_mean: Convert multi-dimensional to 1D array (dimension reduction)
    - spectral_centroid_mean: Convert array to scalar value
    - chroma_mean: Convert multi-dimensional to 1D array
    
    Args:
        input_path: Input feature file path
        output_path: Output feature file path. If not specified, '_normalized' is added to the input file name
    """
    print(f"Loading file '{input_path}'...")
    with open(input_path, 'rb') as f:
        features = pickle.load(f)
    
    print(f"Converting {len(features)} items...")
    
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_normalized{ext}"
    
    first_item = features[0]
    print("\nBefore conversion, first item:")
    for key, value in first_item.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")
    
    converted_features = []
    for feature in tqdm(features):
        converted_feature = {}
        
        for key, value in feature.items():
            if key == 'mfccs_mean':
                if isinstance(value, np.ndarray):
                    if value.ndim == 3:
                        value = np.mean(value.squeeze(0), axis=1)
                    elif value.ndim == 2:
                        value = np.mean(value, axis=1)
                elif isinstance(value, list):
                    value = np.array(value, dtype=np.float32)
                
                if isinstance(value, np.ndarray) and value.ndim != 1:
                    value = value.flatten()
                    
            elif key == 'spectral_centroid_mean':
                if isinstance(value, np.ndarray):
                    value = float(np.mean(value))
                elif isinstance(value, list):
                    value = float(np.mean(np.array(value, dtype=np.float32)))
                
            elif key == 'chroma_mean':
                if isinstance(value, np.ndarray):
                    if value.ndim > 1:
                        value = np.mean(value, axis=1)
                elif isinstance(value, list):
                    value = np.array(value, dtype=np.float32)
                
                if isinstance(value, np.ndarray) and value.ndim != 1:
                    value = value.flatten()
            
            converted_feature[key] = value
        
        converted_features.append(converted_feature)
    
    first_converted = converted_features[0]
    print("\nAfter conversion, first item:")
    for key, value in first_converted.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")
    
    print(f"\nSaving converted features to '{output_path}'...")
    with open(output_path, 'wb') as f:
        pickle.dump(converted_features, f)
    
    print(f"Conversion completed: {len(converted_features)} items saved to '{output_path}'.")
    return output_path

async def detect_onsets_and_split(audio_data, sample_rate, original_filename,
                                output_dir=None, save_splitted_files=False,
                                onset_threshold_db=-40, segment_ms=MIN_WINDOW_MS,
                                max_onsets_per_second=4, max_onsets_per_file=25,
                                min_onset_distance_ms=100):
    """
    Use torchaudio to perform GPU-accelerated onset detection and extract segments of minimum window size (MIN_WINDOW_MS) from each onset.
    Limit the maximum number of onsets per second and file to prevent excessive onset detection.
    Set minimum onset interval to consider consecutive onsets as sustained.
    CUDA/MPS optimization applied.
    Short segments are padded and faded.
    
    Args:
        audio_data: Input audio data
        sample_rate: Audio sample rate
        original_filename: Original audio file name
        output_dir: Output directory for split files
        save_splitted_files: Whether to save split files
        onset_threshold_db: Onset detection threshold in dBFS
        segment_ms: Minimum window size in milliseconds
        max_onsets_per_second: Maximum number of onsets per second
        max_onsets_per_file: Maximum number of onsets per file
        min_onset_distance_ms: Minimum onset interval in milliseconds
        
    Returns:
        list: List of split audio segments with metadata
    """
    device = get_device()
    
    if not torch.is_tensor(audio_data):
        audio_data = torch.from_numpy(audio_data).to(device, non_blocking=True)
    else:
        audio_data = audio_data.to(device, non_blocking=True)
    
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)
    
    n_fft = 128
    hop_length = 64
    win_length = 128
    
    window = torch.hann_window(win_length).to(device)
    
    spec = F.spectrogram(
        waveform=audio_data,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,
        normalized=False,
        center=True
    )
    
    spec_magnitude = torch.abs(spec)
    
    energy = torch.sum(spec_magnitude, dim=1)
    energy_diff = torch.diff(energy, dim=1)
    
    threshold = torch.exp(torch.tensor(onset_threshold_db / 20.0, dtype=torch.float32)).to(device)
    onset_frames = torch.where(energy_diff > threshold)[1]
    
    onset_samples = onset_frames * hop_length
    onset_samples = onset_samples.cpu().numpy()
    
    original_onset_count = len(onset_samples)
    excessive_onsets = False
    
    if len(onset_samples) > 1:
        min_distance_samples = int(sample_rate * (min_onset_distance_ms / 1000.0))
        
        filtered_onsets = [onset_samples[0]]
        
        for i in range(1, len(onset_samples)):
            if onset_samples[i] - filtered_onsets[-1] >= min_distance_samples:
                filtered_onsets.append(onset_samples[i])
        
        onset_samples = np.array(filtered_onsets)
    
    audio_duration = audio_data.size(-1) / sample_rate
    
    onsets_per_second = original_onset_count / audio_duration if audio_duration > 0 else 0
    
    if original_onset_count > max_onsets_per_file or onsets_per_second > max_onsets_per_second:
        excessive_onsets = True
    
    if len(onset_samples) > 0:
        if len(onset_samples) > max_onsets_per_file:
            indices = np.linspace(0, len(onset_samples) - 1, max_onsets_per_file, dtype=int)
            onset_samples = onset_samples[indices]
        else:
            if onsets_per_second > max_onsets_per_second:
                target_onsets = min(int(max_onsets_per_second * audio_duration), max_onsets_per_file)
                
                if target_onsets > 0:
                    indices = np.linspace(0, len(onset_samples) - 1, target_onsets, dtype=int)
                    onset_samples = onset_samples[indices]
    
    segment_samples = MIN_WINDOW_SAMPLES
    
    splitted_buffers = []
    total_length = audio_data.size(-1)
    
    if len(onset_samples) == 0 or onset_samples[0] > 0:
        onset_samples = np.concatenate(([0], onset_samples))
    
    audio_cpu = audio_data.cpu().numpy()
    
    for i, start in enumerate(onset_samples):
        end = start + segment_samples
        
        if end > total_length:
            available_samples = total_length - start
            if available_samples > 0:
                chunk_audio = apply_replication_padding_and_fade(audio_cpu[0, start:total_length], segment_samples)
            else:
                chunk_audio = np.zeros(segment_samples)
        else:
            chunk_audio = audio_cpu[0, start:end]
        
        chunk_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_onset_{i}.wav"
        
        if save_splitted_files and output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            chunk_path = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_path, chunk_audio, sample_rate)
        
        metadata = {
            'excessive_onsets': excessive_onsets,
            'original_onset_count': original_onset_count,
            'onsets_per_second': onsets_per_second,
            'audio_duration': audio_duration
        }
        
        splitted_buffers.append((chunk_audio, sample_rate, chunk_filename, metadata))
    
    del spec, spec_magnitude, energy, energy_diff, window, audio_data
    clear_memory(device)
    
    return splitted_buffers

async def readfile_async(directory_path, filename,
                        detect_onset=False,
                        output_dir=None,
                        save_splitted_files=False,
                        fixed_segment_ms=1000,
                        samples_per_segment=1,
                        energy_threshold_dbfs=-20):
    """
    Asynchronously read audio files from a directory.
    Select the highest energy segment from each file and extract a single sample from the highest energy position.
    Process all files without skipping.
    
    Args:
        directory_path: Path to the directory containing audio files
        filename: Name of the dataset folder
        detect_onset: Whether to detect onsets
        output_dir: Output directory for split files
        save_splitted_files: Whether to save split files
        fixed_segment_ms: Length of each segment in milliseconds
        samples_per_segment: Number of samples to extract per segment
        energy_threshold_dbfs: Energy threshold in dBFS
        
    Returns:
        list: List of extracted audio samples with metadata
    """
    start_time = time.time()
    
    os.makedirs('dataset', exist_ok=True)
    
    buffers_pkl_path = 'dataset/' + filename + '.pkl'
    buffers = []
    
    if os.path.exists(buffers_pkl_path):
        with open(buffers_pkl_path, 'rb') as f:
            buffers = pickle.load(f)
        print(f"Loaded {len(buffers)} buffers from {buffers_pkl_path}.")
        return buffers
    
    print(f"No existing buffer file found. Processing audio files from {directory_path}.")
    audio_files = []
    for root, _, files in os.walk(directory_path):
        for fname in files:
            if fname.endswith(".wav") or fname.endswith(".mp3"):
                audio_files.append(os.path.join(root, fname))
    print(f"Found {len(audio_files)} audio files. Starting to load them...")
    
    total_batches = math.ceil(len(audio_files) / BATCH_SIZE)
    
    excessive_onset_files = []
    
    processed_files = 0
    successful_files = 0
    
    with tqdm(total=len(audio_files), desc='Audio processing progress', unit='files') as pbar:
        async def load_batch(batch_idx, pbar):
            nonlocal processed_files, successful_files
            
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(audio_files))
            batch_files = audio_files[start_idx:end_idx]
            
            all_segments = []
            for file_path in batch_files:
                file_name = os.path.basename(file_path)
                pbar.set_postfix({'Current file': file_name})
                
                segments = await load_single_file(file_path, fixed_segment_ms, samples_per_segment, energy_threshold_dbfs)
                processed_files += 1
                
                if segments is not None and len(segments) > 0:
                    all_segments.extend(segments)
                    successful_files += 1
                
                pbar.update(1)
            
            if detect_onset:
                all_onset_buffers = []
                for audio_data, sr, segment_filename in all_segments:
                    pbar.set_postfix({'Current segment': segment_filename})
                    
                    onset_buffers = await detect_onsets_and_split(
                        audio_data, sr, segment_filename,
                        output_dir=output_dir,
                        save_splitted_files=save_splitted_files,
                        max_onsets_per_second=4,
                        max_onsets_per_file=25,
                        min_onset_distance_ms=100
                    )
                    
                    for _, _, _, metadata in onset_buffers:
                        if metadata['excessive_onsets']:
                            original_filename = os.path.basename(segment_filename).split('_best')[0]
                            if original_filename not in excessive_onset_files:
                                excessive_onset_files.append(original_filename)
                    
                    all_onset_buffers.extend(onset_buffers)
                    pbar.set_postfix({'Detected segments': len(all_onset_buffers), 'Current segment': segment_filename})
                
                return all_onset_buffers
            else:
                samples_with_metadata = []
                for audio_data, sr, segment_filename in all_segments:
                    metadata = {
                        'excessive_onsets': False,
                        'original_onset_count': 0,
                        'onsets_per_second': 0,
                        'audio_duration': len(audio_data) / sr
                    }
                    samples_with_metadata.append((audio_data, sr, segment_filename, metadata))
                return samples_with_metadata
        
        batch_indices = range(total_batches)
        
        tasks = [load_batch(i, pbar) for i in batch_indices]
        results = await asyncio.gather(*tasks)
        for batch_results in results:
            buffers.extend(batch_results)
    
    print(f"Processed files: {processed_files}/{len(audio_files)} (Success rate: {successful_files/processed_files*100:.1f}%)")
    print(f"Extracted segments: {len(buffers)}")
    
    if excessive_onset_files:
        print("\nExcessive onset detection occurred in files:")
        for i, filename in enumerate(excessive_onset_files, 1):
            print(f"{i}. {filename}")
        print(f"Total {len(excessive_onset_files)} files had excessive onset detection.")
    
    with open(buffers_pkl_path, 'wb') as f:
        pickle.dump(buffers, f)
    print(f"Buffers have been saved to {buffers_pkl_path}.")
    
    end_time = time.time()
    print(f"readfile processing time: {end_time - start_time:.2f} seconds")
    return buffers

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        first_arg = sys.argv[1]
        
        if first_arg.endswith('.pkl') and os.path.exists(first_arg):
            print(f"Converting feature file '{first_arg}'...")
            convert_features(first_arg)
        else:
            folder = first_arg
            folder_name = os.path.basename(os.path.normpath(folder))
            
            print(f"Processing folder: {folder}")
            print(f"System configuration:")
            print(f"- CPU cores: {NUM_CPU_CORES}")
            print(f"- Workers: {NUM_WORKERS}")
            print(f"- Batch size: {BATCH_SIZE}")
            
            async def main():
                buffers = await readfile_async(
                    directory_path=folder,
                    filename=folder_name,
                    detect_onset=False,
                    output_dir="splitted_files",
                    save_splitted_files=False,
                    fixed_segment_ms=1000,
                    samples_per_segment=1,
                    energy_threshold_dbfs=-60
                )
                await async_featureExtract(folder_name)
                
                features_pkl_path = f'dataset/{folder_name}_features.pkl'
                check_features(features_pkl_path)
                
                print("\nAutomatically converting extracted feature file to 1D format...")
                convert_features(features_pkl_path)
            
            asyncio.run(main())
    else:
        try:
            import curses_interface
            curses_interface.select_dataset()
        except ImportError:
            print("curses_interface.py module not found.")
            print("Usage: python featureExtractor_torch.py [folderpath]")
            print("      python featureExtractor_torch.py [featurefile.pkl] (feature file conversion)")
            print("Ensure curses_interface.py is in the same directory.") 