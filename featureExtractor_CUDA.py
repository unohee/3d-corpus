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
import librosa  # 상단에 추가
import torch.nn.functional as TF  # 이름 변경
import psutil
from contextlib import nullcontext

# 시스템 설정
NUM_CPU_CORES = cpu_count()  # 실제 CPU 코어 수
BATCH_SIZE = 128  # 64에서 128로 증가
SUB_BATCH_SIZE = 32  # 16에서 32로 증가
NUM_WORKERS = min(NUM_CPU_CORES - 1, 8)  # 하나의 코어는 메인 프로세스용으로 남김

# 분석을 위한 최소 윈도우 크기 정의 (샘플 단위)
MIN_WINDOW_MS = 32  # 최소 32ms 윈도우 (MFCC 분석에 필요한 최소 길이)
MIN_WINDOW_SAMPLES = int(16000 * (MIN_WINDOW_MS / 1000.0))  # 16kHz 샘플링 레이트 기준
FADE_MS = 5  # 페이드 인/아웃 길이 (ms)
FADE_SAMPLES = int(16000 * (FADE_MS / 1000.0))  # 페이드 샘플 수

# 시스템 메모리 확인 (GB 단위)
def get_system_memory():
    return psutil.virtual_memory().total / (1024 ** 3)

# 사용 가능한 메모리와 워커 수에 따라 동시 처리 배치 수 자동 조정
SYSTEM_MEMORY = get_system_memory()
MEMORY_PER_WORKER = SYSTEM_MEMORY / NUM_WORKERS  # 워커당 사용 가능한 메모리
MAX_CONCURRENT_BATCHES = NUM_WORKERS  # 워커 수에 맞춰 동시 처리 배치 수 설정

# 가용한 GPU 백엔드 사용 설정 - 프로세스별로 초기화
def get_device():
    if not hasattr(get_device, "device"):
        if torch.cuda.is_available():
            get_device.device = torch.device("cuda")
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)  # 가용 메모리의 80%만 사용
            if current_process().name == 'MainProcess':
                print(f"CUDA 메모리 최적화 설정 적용됨 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            get_device.device = torch.device("cpu")
            if current_process().name == 'MainProcess':
                print("GPU를 찾을 수 없어 CPU를 사용합니다.")
    return get_device.device

# 디바이스 유형에 따른 메모리 정리 함수
def clear_memory(device):
    """
    디바이스 유형(CUDA, MPS, CPU)에 따라 적절한 메모리 정리를 수행합니다.
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()  # 모든 디바이스 유형에 대해 가비지 컬렉션 실행

# 변환기 캐시
transforms_cache = {}

def get_transforms(device):
    """
    변환기를 캐시하여 재사용합니다.
    FFT 256포인트(16ms), hop 256포인트(16ms)로 설정하여 처리 속도 최대화
    """
    if device not in transforms_cache:
        transforms_cache[device] = {
            'mfcc': torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=13,
                melkwargs={
                    'n_fft': 256,  # 16ms 윈도우
                    'hop_length': 256,  # 16ms 스텝 (0% 오버랩)
                    'n_mels': 13,  # mel 필터뱅크 수 최적화
                    'f_min': 0,  # 전체 주파수 범위 사용
                    'f_max': 8000  # 최대 주파수
                }
            ).to(device),
            'spectral': torchaudio.transforms.SpectralCentroid(
                sample_rate=16000,
                n_fft=128,  # 8ms 윈도우로 축소
                hop_length=64  # 4ms 스텝 (50% 오버랩)
            ).to(device),
            'spec': torchaudio.transforms.Spectrogram(
                n_fft=256,  # 16ms 윈도우
                hop_length=256,  # 16ms 스텝 (0% 오버랩)
                power=None
            ).to(device)
        }
    return transforms_cache[device]

# 리플리케이션 패딩 및 페이딩 함수 추가
def apply_replication_padding_and_fade(audio_data, target_length):
    """
    오디오 데이터에 리플리케이션 패딩과 페이딩을 적용합니다.
    짧은 세그먼트를 반복하여 패딩하고, 부드러운 전환을 위해 페이드 인/아웃을 적용합니다.
    빈 오디오 데이터(길이가 0인 배열)도 안전하게 처리합니다.
    """
    original_length = len(audio_data)
    
    # 원본 데이터가 비어있는 경우 (길이가 0인 배열)
    if original_length == 0:
        # 0으로 채워진 배열 반환
        return np.zeros(target_length)
    
    if original_length >= target_length:
        return audio_data[:target_length]
    
    # 리플리케이션 패딩 적용
    padded_audio = np.zeros(target_length)
    
    # 원본 데이터 복사
    padded_audio[:original_length] = audio_data
    
    # 남은 부분을 원본 데이터 반복으로 채움
    remaining = target_length - original_length
    repetitions_needed = math.ceil(remaining / original_length)
    
    for i in range(repetitions_needed):
        start_idx = original_length + (i * original_length)
        end_idx = min(start_idx + original_length, target_length)
        copy_length = end_idx - start_idx
        padded_audio[start_idx:end_idx] = audio_data[:copy_length]
    
    # 페이드 인/아웃 적용
    fade_samples = min(FADE_SAMPLES, original_length // 4)  # 원본 길이의 1/4까지만 페이드 적용
    
    # 페이드 샘플이 0보다 큰 경우에만 페이딩 적용
    if fade_samples > 0:
        # 페이드 인
        fade_in = np.linspace(0, 1, fade_samples)
        padded_audio[:fade_samples] *= fade_in
        
        # 페이드 아웃
        fade_out = np.linspace(1, 0, fade_samples)
        padded_audio[-fade_samples:] *= fade_out
        
        # 원본 데이터와 반복 데이터 사이의 연결 부분에 크로스페이드 적용
        for i in range(repetitions_needed):
            crossfade_start = original_length + (i * original_length) - fade_samples
            if crossfade_start >= 0 and crossfade_start + (2 * fade_samples) <= target_length:
                fade_out_curve = np.linspace(1, 0, fade_samples)
                fade_in_curve = np.linspace(0, 1, fade_samples)
                
                # 크로스페이드 적용
                crossfade_region = padded_audio[crossfade_start:crossfade_start + (2 * fade_samples)]
                first_half = crossfade_region[:fade_samples] * fade_out_curve
                second_half = crossfade_region[fade_samples:] * fade_in_curve
                
                # 크로스페이드 결과 적용
                padded_audio[crossfade_start:crossfade_start + fade_samples] = first_half
                padded_audio[crossfade_start + fade_samples:crossfade_start + (2 * fade_samples)] = second_half
    
    return padded_audio

async def load_single_file(path, fixed_segment_ms=1000, samples_per_segment=4, energy_threshold_dbfs=-60):  # 에너지 임계값을 매우 낮게 설정
    """
    단일 오디오 파일을 비동기적으로 로드하고 일정한 길이로 분할한 후,
    각 세그먼트의 에너지를 계산하여 가장 에너지가 높은 세그먼트 하나만 선택합니다.
    선택된 세그먼트에서 가장 에너지가 높은 위치에서 하나의 샘플만 추출합니다.
    모든 파일을 처리하며 건너뛰지 않습니다.
    최소 윈도우 크기(MIN_WINDOW_SAMPLES) 미만의 세그먼트는 리플리케이션 패딩 및 페이딩하여 사용합니다.
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
        if waveform_max > 1e-6:  # 충분히 큰 신호가 있는 경우에만 정규화
            waveform = waveform / waveform_max
        
        # 오디오 데이터 준비
        audio_data = waveform.squeeze().numpy()
        total_samples = len(audio_data)
        
        # 빈 오디오 파일 처리 (길이가 0인 경우)
        if total_samples == 0:
            print(f"파일 '{path}'가 비어 있습니다. 더미 데이터로 대체합니다.")
            dummy_sample = np.zeros(MIN_WINDOW_SAMPLES)
            sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_dummy.wav"
            return [(dummy_sample, 16000, sample_filename)]
        
        # 너무 짧은 오디오 파일은 리플리케이션 패딩 및 페이딩으로 처리
        if total_samples < MIN_WINDOW_SAMPLES:
            # 리플리케이션 패딩 및 페이딩 적용
            audio_data = apply_replication_padding_and_fade(audio_data, MIN_WINDOW_SAMPLES)
            total_samples = len(audio_data)
            print(f"파일 '{path}'가 너무 짧아 {MIN_WINDOW_SAMPLES} 샘플({MIN_WINDOW_MS}ms)로 리플리케이션 패딩 및 페이딩되었습니다.")
        
        # 고정 길이 세그먼트로 분할
        segment_samples = int(16000 * (fixed_segment_ms / 1000.0))
        
        # 각 세그먼트의 에너지 계산
        segment_energies = []
        segment_positions = []
        
        # dBFS 임계값을 선형 스케일로 변환
        energy_threshold = 10 ** (energy_threshold_dbfs / 10)
        
        for seg_idx, start in enumerate(range(0, total_samples, segment_samples)):
            end = min(start + segment_samples, total_samples)
            segment_length = end - start
            
            # 세그먼트 길이가 0인 경우 건너뛰기 (이론적으로는 발생하지 않아야 함)
            if segment_length == 0:
                continue
            
            # 최소 윈도우 크기보다 짧은 세그먼트는 리플리케이션 패딩 및 페이딩
            if segment_length < MIN_WINDOW_SAMPLES:
                # 리플리케이션 패딩 및 페이딩 적용
                segment_data = apply_replication_padding_and_fade(audio_data[start:end], MIN_WINDOW_SAMPLES)
            else:
                segment_data = audio_data[start:end]
            
            # RMS 에너지 계산 후 dBFS로 변환
            rms_energy = np.mean(segment_data ** 2)
            energy_dbfs = 10 * np.log10(rms_energy) if rms_energy > 0 else -100
            
            # 모든 세그먼트 저장 (에너지 임계값 무시)
            segment_energies.append(rms_energy)
            segment_positions.append((seg_idx, start, end))
        
        # 세그먼트가 없는 경우 (매우 드문 경우) 전체 오디오를 하나의 세그먼트로 사용
        if not segment_energies:
            # 전체 오디오를 하나의 세그먼트로 사용
            segment_data = audio_data
            if len(segment_data) < MIN_WINDOW_SAMPLES:
                # 리플리케이션 패딩 및 페이딩 적용
                segment_data = apply_replication_padding_and_fade(segment_data, MIN_WINDOW_SAMPLES)
            
            sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_best.wav"
            return [(segment_data, 16000, sample_filename)]
        
        # 가장 에너지가 높은 세그먼트 선택
        max_energy_idx = np.argmax(segment_energies)
        best_seg_idx, start, end = segment_positions[max_energy_idx]
        
        # 선택된 세그먼트 내에서 가장 에너지가 높은 위치 찾기
        segment_data = audio_data[start:end]
        segment_length = end - start
        
        # 세그먼트가 최소 윈도우 크기보다 작으면 리플리케이션 패딩 및 페이딩
        if segment_length < MIN_WINDOW_SAMPLES:
            segment_data = apply_replication_padding_and_fade(segment_data, MIN_WINDOW_SAMPLES)
            segment_length = MIN_WINDOW_SAMPLES
        
        # 세그먼트를 작은 윈도우로 나누어 에너지 계산
        window_size = MIN_WINDOW_SAMPLES
        window_energies = []
        window_positions = []
        
        # 윈도우 크기보다 작은 세그먼트는 이미 패딩되어 있으므로 여기서는 처리하지 않음
        
        for window_start in range(0, segment_length - window_size + 1, window_size // 2):  # 50% 오버랩
            window_end = window_start + window_size
            window = segment_data[window_start:window_end]
            energy = np.mean(window ** 2)  # RMS 에너지
            window_energies.append(energy)
            window_positions.append(start + window_start)
        
        # 윈도우가 없으면 세그먼트 시작 위치 사용
        if not window_energies:
            best_pos = start
        else:
            # 가장 에너지가 높은 윈도우 선택
            max_window_idx = np.argmax(window_energies)
            best_pos = window_positions[max_window_idx]
        
        # 선택된 위치에서 샘플 추출 (항상 MIN_WINDOW_SAMPLES 길이 보장)
        end_pos = min(best_pos + MIN_WINDOW_SAMPLES, total_samples)
        if end_pos - best_pos < MIN_WINDOW_SAMPLES:
            # 리플리케이션 패딩 및 페이딩 적용
            sample = apply_replication_padding_and_fade(audio_data[best_pos:end_pos], MIN_WINDOW_SAMPLES)
        else:
            sample = audio_data[best_pos:end_pos]
        
        sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_best.wav"
        return [(sample, 16000, sample_filename)]
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        # 오류 발생 시에도 빈 결과 대신 최소한의 더미 데이터 반환
        dummy_sample = np.zeros(MIN_WINDOW_SAMPLES)  # 최소 윈도우 크기의 더미 샘플
        sample_filename = f"{os.path.splitext(os.path.basename(path))[0]}_dummy.wav"
        print(f"파일 '{path}' 처리 중 오류 발생, 더미 데이터로 대체합니다.")
        return [(dummy_sample, 16000, sample_filename)]

async def load_audio_batch(file_paths):
    """
    배치 단위로 오디오 파일을 비동기적으로 로드합니다.
    """
    # 병렬로 파일 로드
    tasks = [load_single_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    
    # None이 아닌 결과만 반환
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
    
    # CUDA 최적화: 텐서 생성 및 디바이스 이동
    audio_tensors = []
    for audio_data in audio_arrays:
        tensor = torch.from_numpy(audio_data).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        pad_size = max_len - tensor.size(-1)
        if pad_size > 0:
            tensor = TF.pad(tensor, (0, pad_size))
        
        # 신호 정규화
        rms = torch.sqrt(torch.mean(tensor ** 2))
        if rms > 1e-6:
            tensor = tensor / rms
        
        audio_tensors.append(tensor)
    
    # 배치 텐서를 CUDA로 이동
    batch_tensor = torch.stack(audio_tensors).to(device, non_blocking=True)
    
    transforms = get_transforms(device)
    
    async def extract_features():
        with torch.no_grad():
            # 자동 혼합 정밀도 활성화 (CUDA에서만)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    batch_tensor_cont = batch_tensor.contiguous()
                    
                    # CUDA 최적화: 배치 처리
                    mfccs = transforms['mfcc'](batch_tensor_cont)
                    spectral = transforms['spectral'](batch_tensor_cont)
                    spec = transforms['spec'](batch_tensor_cont)
            else:
                batch_tensor_cont = batch_tensor.contiguous()
                mfccs = transforms['mfcc'](batch_tensor_cont)
                spectral = transforms['spectral'](batch_tensor_cont)
                spec = transforms['spec'](batch_tensor_cont)
            
            # MFCCs 처리 (차원이 [배치, 1, 특성, 시간] 형태임)
            # 예: [32, 1, 13, time]
            mfccs = mfccs.squeeze(1)  # [배치, 특성, 시간]
            
            # 시간 축에 대한 평균 계산
            mfcc_means = torch.mean(mfccs, dim=2)  # [배치, 특성]
            
            # Spectral Centroid 처리
            # spectral 형태: [배치, 1, 시간]
            spectral = spectral.squeeze(1)  # [배치, 시간]
            
            # NaN 제거 및 안전한 평균 계산
            # 1. NaN 값을 0으로 대체
            spectral = torch.nan_to_num(spectral, nan=0.0, posinf=4000.0, neginf=0.0)
            
            # 2. 시간 축 평균 계산
            # 모든 값이 NaN인 경우 0을 반환하도록 처리
            spectral_means = torch.mean(spectral, dim=1, keepdim=True)
            
            # NaN 확인 및 처리
            spectral_means = torch.nan_to_num(spectral_means, nan=0.0, posinf=4000.0, neginf=0.0)
            
            # 크로마 특성 계산
            magnitude = torch.abs(spec)
            magnitude = magnitude.squeeze(1)  # [배치, 주파수, 시간]
            
            # 크로마 계산 - CPU에서 수행
            magnitude_cpu = magnitude.cpu()
            n_fft = 256
            sample_rate = 16000
            
            bin_frequencies = torch.arange(magnitude_cpu.shape[1]) * (sample_rate / n_fft)
            
            min_freq = 32.7  # C1
            max_freq = 4186.0  # C8
            valid_freq_mask = (bin_frequencies >= min_freq) & (bin_frequencies <= max_freq)
            
            valid_frequencies = bin_frequencies[valid_freq_mask]
            magnitude_valid = magnitude_cpu[:, valid_freq_mask, :]
            
            midi_notes = 12 * torch.log2(valid_frequencies / 440.0) + 69
            chroma_bins = torch.remainder(midi_notes.round(), 12).long()
            
            chroma = torch.zeros(len(batch_tensor_cont), 12, magnitude_cpu.shape[-1])
            
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
            
            chroma_means = torch.mean(chroma, dim=2)  # [배치, 12]
            
            # NaN 확인 및 처리
            chroma_means = torch.nan_to_num(chroma_means, nan=0.0)
            
            # CPU로 이동
            mfcc_means = mfcc_means.cpu().float()
            spectral_means = spectral_means.cpu().float()
            
            # 마지막 NaN 검사
            mfcc_means = torch.nan_to_num(mfcc_means, nan=0.0)
            spectral_means = torch.nan_to_num(spectral_means, nan=0.0)
            
            return mfcc_means, spectral_means, chroma_means
    
    # 특징 추출 실행
    mfcc_means, spectral_means, chroma_means = await extract_features()
    
    # 결과 처리 - CPU 텐서를 numpy로 변환
    results = []
    for i in range(len(audio_batch)):
        # 모든 텐서를 numpy로 안전하게 변환
        mfcc_np = mfcc_means[i].detach().numpy()
        spectral_np = spectral_means[i].detach().numpy()
        chroma_np = chroma_means[i].detach().numpy()
        
        # NumPy 배열의 NaN 값을 0으로 대체
        mfcc_np = np.nan_to_num(mfcc_np, nan=0.0)
        spectral_np = np.nan_to_num(spectral_np, nan=0.0)
        chroma_np = np.nan_to_num(chroma_np, nan=0.0)
        
        results.append({
            'mfccs_mean': mfcc_np,
            'spectral_centroid_mean': spectral_np,
            'chroma_mean': chroma_np
        })
    
    # CUDA 메모리 정리
    clear_memory(device)
    
    return results

async def process_features_async(buffers):
    """
    전체 버퍼를 비동기적으로 처리합니다.
    CUDA 활용을 최대화하기 위해 최적화되었습니다.
    """
    device = get_device()
    all_features = []
    
    # 과도한 온셋 검출 파일 추적
    excessive_onset_files = []
    
    # 프로그레스바 업데이트 간격 계산
    total_batches = math.ceil(len(buffers) / BATCH_SIZE)
    
    # CUDA 최적화를 위한 동시 처리 배치 수 조정
    if device.type == 'cuda':
        # CUDA 디바이스 메모리 확인
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        if free_memory <= 0:  # 초기 상태에서 메모리가 할당되지 않은 경우
            free_memory = total_memory * 0.8  # 총 메모리의 80%를 사용 가능하다고 가정
        memory_per_batch = 50 * 1024 * 1024  # 배치당 메모리 사용량 50MB로 감소
        max_batches_by_memory = int(free_memory / memory_per_batch * 0.8)
        concurrent_batches = max(4, min(MAX_CONCURRENT_BATCHES, max_batches_by_memory))  # 최소 4개 배치 처리
        
        print(f"CUDA 디바이스 감지: 동시 처리 배치 수를 {concurrent_batches}로 최적화")
        print(f"CUDA 메모리: 총 {total_memory/(1024**3):.2f}GB, 가용 {free_memory/(1024**3):.2f}GB")
    else:
        concurrent_batches = MAX_CONCURRENT_BATCHES
    
    print(f"동시 처리 배치 수: {concurrent_batches} (워커당 메모리: {MEMORY_PER_WORKER:.1f}GB)")
    
    async def process_batch_group(start_batch_idx, pbar):
        group_results = []
        group_tasks = []
        
        # 현재 그룹에서 처리할 배치 인덱스 계산
        for batch_idx in range(start_batch_idx, min(start_batch_idx + concurrent_batches, total_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(buffers))
            batch = buffers[start_idx:end_idx]
            
            audio_batch = [(audio_data, sr) for audio_data, sr, _, _ in batch]
            filenames = [fname for _, _, fname, _ in batch]
            metadata_list = [metadata for _, _, _, metadata in batch]
            
            # 배치 처리 태스크 생성 및 즉시 실행
            task = asyncio.create_task(process_audio_batch(audio_batch, device))
            group_tasks.append((batch_idx, filenames, metadata_list, task))
        
        # 모든 태스크를 동시에 실행하고 결과 수집
        try:
            # 모든 태스크의 완료를 기다림
            completed_tasks = await asyncio.gather(*(task for _, _, _, task in group_tasks))
            
            # 결과 처리
            for (batch_idx, filenames, metadata_list, _), batch_results in zip(group_tasks, completed_tasks):
                batch_features = []
                for filename, metadata, result in zip(filenames, metadata_list, batch_results):
                    feature_data = {'filename': filename}
                    feature_data.update(result)
                    feature_data.update(metadata)  # 메타데이터 추가
                    
                    # 과도한 온셋 검출 파일 추적
                    if metadata['excessive_onsets']:
                        original_filename = os.path.basename(filename).split('_onset_')[0]
                        if original_filename not in excessive_onset_files:
                            excessive_onset_files.append(original_filename)
                    
                    batch_features.append(feature_data)
                    # 현재 처리 중인 파일 이름 표시
                    file_name = os.path.basename(filename)
                    pbar.set_postfix({'현재 파일': file_name})
                    pbar.update(1)  # 각 파일마다 진행률 업데이트
                group_results.append((batch_idx, batch_features))
                
        except Exception as e:
            print(f"Error in batch group processing: {e}")
            # 실패한 경우 빈 결과 반환
            for batch_idx, _, _, _ in group_tasks:
                group_results.append((batch_idx, []))
        
        # 배치 그룹 처리 후 메모리 정리
        clear_memory(device)
        
        return group_results
    
    # 배치 그룹 단위로 처리
    with tqdm(total=len(buffers), desc='특징 추출 진행률', unit='files') as pbar:
        for start_idx in range(0, total_batches, concurrent_batches):
            group_results = await process_batch_group(start_idx, pbar)
            
            # 결과를 배치 인덱스 순서대로 정렬
            group_results.sort(key=lambda x: x[0])
            
            # 결과 저장
            for _, batch_features in group_results:
                all_features.extend(batch_features)
            
            # 메모리 정리
            gc.collect()
            clear_memory(device)
    
    # 과도한 온셋 검출 파일 보고
    if excessive_onset_files:
        print("\n과도한 온셋 검출이 발생한 파일:")
        for i, filename in enumerate(excessive_onset_files, 1):
            print(f"{i}. {filename}")
        print(f"총 {len(excessive_onset_files)}개 파일에서 과도한 온셋 검출이 발생했습니다.")
    
    return all_features

async def async_featureExtract(path):
    """
    비동기적으로 전체 버퍼를 처리합니다.
    MPS 활용을 최대화하기 위해 최적화되었습니다.
    """
    import time
    start_time = time.time()
    
    # dataset 폴더가 없는 경우 생성
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
    
    # 비동기 처리 실행
    all_features = await process_features_async(buffers)
    
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Features extracted and saved to {features_pkl_path}.")
    
    end_time = time.time()
    print(f"async_featureExtract 처리 시간: {end_time - start_time:.2f}초")

def check_features(features_pkl_path):
    """
    특성 추출 데이터를 통계적으로 분석하고 검증합니다.
    - 기본 통계 (평균, 표준편차, 중앙값, 사분위수 등)
    - 이상치 탐지
    - 특성 간 상관관계
    - 분포 분석
    - 0값과 NaN 분석
    """
    try:
        with open(features_pkl_path, 'rb') as f:
            features = pickle.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {features_pkl_path}")
        return

    print(f"\n특성 추출 데이터 통계 분석 결과:")
    print(f"총 오디오 유닛 개수: {len(features)}")

    if not features or not isinstance(features, list) or not isinstance(features[0], dict):
        print("오류: 데이터 구조가 예상과 다릅니다.")
        return

    # 통계 데이터 수집을 위한 딕셔너리 초기화
    stats = {}
    for key in features[0].keys():
        if key not in ['filename', 'excessive_onsets', 'original_onset_count', 'onsets_per_second', 'audio_duration']:
            stats[key] = {
                'values': [],           # 모든 값을 저장
                'nan_count': 0,         # NaN 개수
                'zero_count': 0,        # 0 값 개수
                'total_elements': 0,    # 전체 요소 수
                'shape_counts': {},     # shape 분포
                'percentiles': {},      # 백분위수
                'outliers': [],         # 이상치 정보
            }

    # 데이터 수집
    for idx, feature in enumerate(features):
        for key, value in feature.items():
            if key in stats:
                if isinstance(value, np.ndarray):
                    # shape 통계
                    shape_str = str(value.shape)
                    stats[key]['shape_counts'][shape_str] = stats[key]['shape_counts'].get(shape_str, 0) + 1
                    
                    # 1차원으로 변환하여 통계 계산
                    flat_value = value.flatten()
                    stats[key]['values'].extend(flat_value)
                    stats[key]['nan_count'] += np.isnan(flat_value).sum()
                    stats[key]['zero_count'] += np.sum(np.abs(flat_value) < 1e-6)
                    stats[key]['total_elements'] += flat_value.size

    # 통계 계산 및 출력
    print("\n=== 특성별 상세 통계 분석 ===")
    for key in stats.keys():
        values = np.array(stats[key]['values'])
        non_nan_values = values[~np.isnan(values)]
        
        print(f"\n{key}:")
        print(f"  1. 기본 정보:")
        print(f"    - 총 요소 수: {stats[key]['total_elements']:,}")
        print(f"    - NaN 비율: {(stats[key]['nan_count'] / stats[key]['total_elements'] * 100):.2f}%")
        print(f"    - 0값 비율: {(stats[key]['zero_count'] / stats[key]['total_elements'] * 100):.2f}%")
        
        if len(non_nan_values) > 0:
            # 기본 통계량
            percentiles = np.percentile(non_nan_values, [0, 25, 50, 75, 100])
            mean = np.mean(non_nan_values)
            std = np.std(non_nan_values)
            
            print(f"  2. 분포 통계:")
            print(f"    - 평균: {mean:.3f}")
            print(f"    - 표준편차: {std:.3f}")
            print(f"    - 중앙값: {percentiles[2]:.3f}")
            print(f"    - 최소값: {percentiles[0]:.3f}")
            print(f"    - 최대값: {percentiles[4]:.3f}")
            print(f"    - IQR: {(percentiles[3] - percentiles[1]):.3f}")
            
            # 이상치 탐지 (IQR 방법)
            q1, q3 = percentiles[1], percentiles[3]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = non_nan_values[(non_nan_values < lower_bound) | (non_nan_values > upper_bound)]
            outlier_ratio = len(outliers) / len(non_nan_values) * 100
            
            print(f"  3. 이상치 분석:")
            print(f"    - 이상치 비율: {outlier_ratio:.2f}%")
            print(f"    - 이상치 범위: [{lower_bound:.3f}, {upper_bound:.3f}] 외")
            
            # 분포 특성
            skewness = np.mean(((non_nan_values - mean) / std) ** 3) if std > 0 else 0
            kurtosis = np.mean(((non_nan_values - mean) / std) ** 4) - 3 if std > 0 else 0
            
            print(f"  4. 분포 특성:")
            print(f"    - 왜도(Skewness): {skewness:.3f}")
            print(f"    - 첨도(Kurtosis): {kurtosis:.3f}")
        
        print(f"  5. 데이터 형태:")
        for shape, count in stats[key]['shape_counts'].items():
            print(f"    - Shape {shape}: {count:,}개 ({count/len(features)*100:.1f}%)")

    # 특성 간 상관관계 분석
    print("\n=== 특성 간 상관관계 분석 ===")
    feature_means = {}
    for key in stats.keys():
        if len(stats[key]['values']) > 0:
            feature_means[key] = []
            for feature in features:
                if isinstance(feature[key], np.ndarray):
                    # 각 특성의 평균값 계산 방식 수정
                    if key == 'mfccs_mean':
                        # MFCC는 각 계수별 평균을 사용
                        mean_value = np.mean(feature[key])
                    elif key == 'spectral_centroid_mean':
                        # Spectral Centroid는 전체 평균 사용
                        mean_value = np.mean(feature[key])
                    elif key == 'chroma_mean':
                        # Chroma는 12개 음높이 클래스의 평균 사용
                        mean_value = np.mean(feature[key])
                    feature_means[key].append(mean_value)
    
    if len(feature_means) >= 2:
        correlation_matrix = np.zeros((len(feature_means), len(feature_means)))
        feature_keys = list(feature_means.keys())
        
        print("\n특성별 평균값 통계:")
        for key in feature_keys:
            values = np.array(feature_means[key])
            print(f"\n{key}:")
            print(f"  - 평균: {np.mean(values):.3f}")
            print(f"  - 표준편차: {np.std(values):.3f}")
            print(f"  - 범위: [{np.min(values):.3f}, {np.max(values):.3f}]")
        
        print("\n상관계수 행렬:")
        for i, key1 in enumerate(feature_keys):
            print(f"\n{key1}:")
            for j, key2 in enumerate(feature_keys):
                if i != j:  # 자기 자신과의 상관관계는 제외
                    correlation = np.corrcoef(
                        feature_means[key1],
                        feature_means[key2]
                    )[0, 1]
                    correlation_matrix[i, j] = correlation
                    strength = "강한" if abs(correlation) > 0.7 else "중간" if abs(correlation) > 0.3 else "약한"
                    direction = "양" if correlation > 0 else "음"
                    print(f"  - {key2}와의 상관계수: {correlation:.3f} ({strength} {direction}의 상관관계)")

    # 첫 번째 샘플의 구조 출력
    print("\n=== 첫 번째 샘플의 상세 구조 ===")
    first_feature = features[0]
    for key, value in first_feature.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    - shape: {value.shape}")
            print(f"    - dtype: {value.dtype}")
            print(f"    - 평균: {np.nanmean(value):.3f}")
            print(f"    - 표준편차: {np.nanstd(value):.3f}")
        else:
            print(f"  {key}: {value}")

async def detect_onsets_and_split(audio_data, sample_rate, original_filename,
                                output_dir=None, save_splitted_files=False,
                                onset_threshold_db=-40, segment_ms=MIN_WINDOW_MS,  # 최소 윈도우 크기로 세그먼트 길이 설정
                                max_onsets_per_second=4, max_onsets_per_file=25,
                                min_onset_distance_ms=100):  # 최소 온셋 간격 파라미터 추가
    """
    torchaudio를 사용하여 GPU 가속이 가능한 온셋 검출을 수행하고,
    각 온셋에서 최소 윈도우 크기(MIN_WINDOW_MS) 길이의 세그먼트를 추출합니다.
    초당 최대 온셋 수와 파일당 최대 온셋 수를 제한하여 과도한 온셋 검출을 방지합니다.
    최소 온셋 간격을 설정하여 연속적인 온셋을 서스테인으로 간주합니다.
    CUDA/MPS 최적화가 적용되었습니다.
    짧은 세그먼트는 리플리케이션 패딩 및 페이딩을 적용합니다.
    """
    device = get_device()
    
    # 텐서로 변환 (non_blocking=True로 비동기 전송)
    if not torch.is_tensor(audio_data):
        audio_data = torch.from_numpy(audio_data).to(device, non_blocking=True)
    else:
        audio_data = audio_data.to(device, non_blocking=True)
    
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)
    
    # 온셋 검출을 위한 파라미터 설정
    n_fft = 128  # 8ms 윈도우
    hop_length = 64  # 4ms 스텝 (50% 오버랩)
    win_length = 128  # 윈도우 크기와 동일하게 설정
    
    # 윈도우 함수를 디바이스에 미리 생성
    window = torch.hann_window(win_length).to(device)
    
    # 스펙트로그램 계산
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
    
    # 진폭 스펙트로그램으로 변환
    spec_magnitude = torch.abs(spec)
    
    # 시간에 따른 에너지 변화 계산
    energy = torch.sum(spec_magnitude, dim=1)
    energy_diff = torch.diff(energy, dim=1)
    
    # 온셋 임계값 설정 및 적용
    threshold = torch.exp(torch.tensor(onset_threshold_db / 20.0, dtype=torch.float32)).to(device)
    onset_frames = torch.where(energy_diff > threshold)[1]
    
    # 샘플 단위로 변환 (CPU로 이동)
    onset_samples = onset_frames * hop_length
    onset_samples = onset_samples.cpu().numpy()
    
    # 원래 검출된 온셋 수 저장
    original_onset_count = len(onset_samples)
    excessive_onsets = False
    
    # 최소 온셋 간격 적용 (서스테인 처리)
    if len(onset_samples) > 1:
        # 최소 간격을 샘플 단위로 변환
        min_distance_samples = int(sample_rate * (min_onset_distance_ms / 1000.0))
        
        # 충분한 간격이 있는 온셋만 유지
        filtered_onsets = [onset_samples[0]]  # 첫 번째 온셋은 항상 유지
        
        for i in range(1, len(onset_samples)):
            # 이전 온셋과의 간격이 최소 간격보다 크면 유지
            if onset_samples[i] - filtered_onsets[-1] >= min_distance_samples:
                filtered_onsets.append(onset_samples[i])
        
        onset_samples = np.array(filtered_onsets)
    
    # 오디오 길이(초)
    audio_duration = audio_data.size(-1) / sample_rate
    
    # 현재 초당 온셋 수
    onsets_per_second = original_onset_count / audio_duration if audio_duration > 0 else 0
    
    # 과도한 온셋 검출 여부 확인
    if original_onset_count > max_onsets_per_file or onsets_per_second > max_onsets_per_second:
        excessive_onsets = True
    
    # 온셋 수 제한 적용
    if len(onset_samples) > 0:
        # 파일당 최대 온셋 수 제한
        if len(onset_samples) > max_onsets_per_file:
            # 균등한 간격으로 온셋 선택
            indices = np.linspace(0, len(onset_samples) - 1, max_onsets_per_file, dtype=int)
            onset_samples = onset_samples[indices]
        else:
            # 초당 온셋 수가 제한을 초과하는 경우
            if onsets_per_second > max_onsets_per_second:
                # 필요한 총 온셋 수 계산
                target_onsets = min(int(max_onsets_per_second * audio_duration), max_onsets_per_file)
                
                # 균등한 간격으로 온셋 선택
                if target_onsets > 0:
                    indices = np.linspace(0, len(onset_samples) - 1, target_onsets, dtype=int)
                    onset_samples = onset_samples[indices]
    
    # 세그먼트 길이 계산 (최소 윈도우 크기 사용)
    segment_samples = MIN_WINDOW_SAMPLES
    
    # 오디오 분할
    splitted_buffers = []
    total_length = audio_data.size(-1)
    
    # 첫 번째 온셋이 0에서 시작하지 않는 경우 추가
    if len(onset_samples) == 0 or onset_samples[0] > 0:
        onset_samples = np.concatenate(([0], onset_samples))
    
    # 한 번에 모든 세그먼트 추출 (메모리 효율성 향상)
    audio_cpu = audio_data.cpu().numpy()
    
    for i, start in enumerate(onset_samples):
        end = start + segment_samples
        
        # 세그먼트가 오디오 길이를 초과하면 리플리케이션 패딩 및 페이딩 처리
        if end > total_length:
            # 리플리케이션 패딩 및 페이딩 적용
            available_samples = total_length - start
            if available_samples > 0:
                chunk_audio = apply_replication_padding_and_fade(audio_cpu[0, start:total_length], segment_samples)
            else:
                # 시작 위치가 오디오 길이를 초과하는 경우 (매우 드문 경우)
                chunk_audio = np.zeros(segment_samples)
        else:
            chunk_audio = audio_cpu[0, start:end]
        
        chunk_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_onset_{i}.wav"
        
        if save_splitted_files and output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            chunk_path = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_path, chunk_audio, sample_rate)
        
        # 과도한 온셋 검출 정보 추가
        metadata = {
            'excessive_onsets': excessive_onsets,
            'original_onset_count': original_onset_count,
            'onsets_per_second': onsets_per_second,
            'audio_duration': audio_duration
        }
        
        splitted_buffers.append((chunk_audio, sample_rate, chunk_filename, metadata))
    
    # 메모리 정리
    del spec, spec_magnitude, energy, energy_diff, window, audio_data
    clear_memory(device)
    
    return splitted_buffers

async def readfile_async(directory_path, filename,
                        detect_onset=False,  # 온셋 검출 비활성화
                        output_dir=None,
                        save_splitted_files=False,
                        fixed_segment_ms=1000,
                        samples_per_segment=1,   # 샘플 수를 1로 변경
                        energy_threshold_dbfs=-20):  # 에너지 임계값을 매우 낮게 설정
    """
    디렉토리에서 오디오 파일을 비동기적으로 읽어들입니다.
    각 오디오 파일에서 가장 에너지가 높은 세그먼트 하나만 선택하고,
    해당 세그먼트에서 가장 에너지가 높은 위치에서 하나의 샘플만 추출합니다.
    모든 파일을 처리하며 건너뛰지 않습니다.
    """
    import time
    start_time = time.time()
    
    # dataset 폴더가 없는 경우 생성
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
    
    # 전체 파일을 여러 배치로 나누어 병렬 처리
    total_batches = math.ceil(len(audio_files) / BATCH_SIZE)
    
    # 과도한 온셋 검출 파일 추적
    excessive_onset_files = []
    
    # 처리된 파일 수와 성공적으로 처리된 파일 수 추적
    processed_files = 0
    successful_files = 0
    
    # 전체 진행률 표시 (전체 파일 수 기준)
    with tqdm(total=len(audio_files), desc='오디오 처리 진행률', unit='files') as pbar:
        async def load_batch(batch_idx, pbar):  # pbar 파라미터 추가
            nonlocal processed_files, successful_files
            
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(audio_files))
            batch_files = audio_files[start_idx:end_idx]
            
            all_segments = []
            for file_path in batch_files:
                # 현재 처리 중인 파일 이름 표시
                file_name = os.path.basename(file_path)
                pbar.set_postfix({'현재 파일': file_name})
                
                # 파일에서 가장 에너지가 높은 세그먼트 하나만 선택
                segments = await load_single_file(file_path, fixed_segment_ms, samples_per_segment, energy_threshold_dbfs)
                processed_files += 1
                
                if segments is not None and len(segments) > 0:
                    all_segments.extend(segments)
                    successful_files += 1
                
                pbar.update(1)  # 각 파일마다 진행률 업데이트
            
            if detect_onset:
                all_onset_buffers = []
                for audio_data, sr, segment_filename in all_segments:
                    # 현재 처리 중인 세그먼트 이름 표시
                    pbar.set_postfix({'현재 세그먼트': segment_filename})
                    
                    onset_buffers = await detect_onsets_and_split(
                        audio_data, sr, segment_filename,
                        output_dir=output_dir,
                        save_splitted_files=save_splitted_files,
                        max_onsets_per_second=4,  # 초당 최대 4개의 온셋만 유지
                        max_onsets_per_file=25,   # 파일당 최대 25개의 온셋만 유지
                        min_onset_distance_ms=100  # 최소 100ms 간격의 온셋만 유지 (서스테인 처리)
                    )
                    
                    # 과도한 온셋 검출 파일 추적
                    for _, _, _, metadata in onset_buffers:
                        if metadata['excessive_onsets']:
                            original_filename = os.path.basename(segment_filename).split('_best')[0]
                            if original_filename not in excessive_onset_files:
                                excessive_onset_files.append(original_filename)
                    
                    all_onset_buffers.extend(onset_buffers)
                    # 검출된 온셋 수와 세그먼트 이름 함께 표시
                    pbar.set_postfix({'검출된 세그먼트': len(all_onset_buffers), '현재 세그먼트': segment_filename})
                
                return all_onset_buffers
            else:
                # 온셋 검출 없이 직접 추출한 샘플 사용
                # 메타데이터 추가
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
        
        # 배치 단위로 병렬 처리
        batch_indices = range(total_batches)
        
        # 전체 진행률 표시 (전체 파일 수 기준)
        tasks = [load_batch(i, pbar) for i in batch_indices]
        results = await asyncio.gather(*tasks)
        for batch_results in results:
            buffers.extend(batch_results)
    
    print(f"처리된 파일: {processed_files}/{len(audio_files)} (성공률: {successful_files/processed_files*100:.1f}%)")
    print(f"추출된 세그먼트: {len(buffers)}")
    
    # 과도한 온셋 검출 파일 보고
    if excessive_onset_files:
        print("\n과도한 온셋 검출이 발생한 파일:")
        for i, filename in enumerate(excessive_onset_files, 1):
            print(f"{i}. {filename}")
        print(f"총 {len(excessive_onset_files)}개 파일에서 과도한 온셋 검출이 발생했습니다.")
    
    with open(buffers_pkl_path, 'wb') as f:
        pickle.dump(buffers, f)
    print(f"Buffers have been saved to {buffers_pkl_path}.")
    
    end_time = time.time()
    print(f"readfile 처리 시간: {end_time - start_time:.2f}초")
    return buffers

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        folder = sys.argv[1]
    else:
        folder = "./dataset/FSD50K.eval_audio"
        
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
            detect_onset=False,  # 온셋 검출 비활성화
            output_dir="splitted_files",
            save_splitted_files=False,
            fixed_segment_ms=1000,  # 1초 세그먼트
            samples_per_segment=1,   # 각 세그먼트에서 1개 샘플만 추출
            energy_threshold_dbfs=-60  # 에너지 임계값을 매우 낮게 설정 (-60dBFS)
        )
        await async_featureExtract(folder_name)
        
        # 특성 추출 결과 체크
        features_pkl_path = f'dataset/{folder_name}_features.pkl'
        check_features(features_pkl_path)
    
    asyncio.run(main()) 