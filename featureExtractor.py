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

# 유틸리티 함수들
def load_audio_file(file_path):
    """
    Helper function to load an audio file with a fixed sample rate (16k).
    노멀라이제이션도 수행하여 출력합니다.
    Returns the normalized audio data and its sample rate.
    """
    audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    # 음원을 최대 절대값 기준으로 정규화 (범위: [-1, 1])
    audio_data = librosa.util.normalize(audio_data)
    return audio_data, sample_rate

def detect_onsets_and_split(audio_data, sample_rate, original_filename,
                            output_dir=None, save_splitted_files=False, 
                            min_length_ms=500, onset_threshold_db=-40):
    """
    노멀라이즈된 오디오 데이터에서 온셋을 탐지하고, 
    각 온셋에 해당하는 구간을 분할하여 (chunk_audio, sample_rate, chunk_filename, start_sample, end_sample) 튜플로 반환합니다.
    분할된 구간의 길이가 min_length_ms보다 작으면 무시합니다.
    
    onset_threshold_db : dBFS 단위의 threshold. 음원이 노멀라이즈되어 있으므로, 일반적으로 -40 dB 정도가 적당합니다.
    """
    # dBFS threshold를 amplitude 차이로 변환 후 np.float32로 캐스팅
    delta_val = np.float32(librosa.db_to_amplitude(onset_threshold_db, ref=1.0))
    
    # 최소 길이를 500ms에 해당하는 샘플 수로 계산
    min_length_samples = int(sample_rate * (min_length_ms / 1000.0))
    
    # librosa의 onset 탐지 (Bello et al., 2005): delta 파라미터를 적용
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, delta=delta_val)
    
    # 프레임 단위로 검출되므로, 샘플 단위 위치로 변환
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)  

    splitted_buffers = []
    total_length = len(audio_data)

    # tqdm를 적용하여 온셋 진행 상황 표시
    for i in tqdm(range(len(onset_samples)), desc='Splitting onsets', leave=False):
        start = onset_samples[i]
        end = onset_samples[i+1] if i+1 < len(onset_samples) else total_length

        # 오디오 조각 추출
        chunk_audio = audio_data[start:end]
        # 최소 길이 미만이면 해당 청크는 건너뜁니다.
        if len(chunk_audio) < min_length_samples:
            continue

        # 마커용 파일명 생성
        chunk_filename = f"{os.path.splitext(os.path.basename(original_filename))[0]}_onset_{i}.wav"

        # 필요 시 파일로 저장
        if save_splitted_files and output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            chunk_path = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_path, chunk_audio, sample_rate)

        splitted_buffers.append((chunk_audio, sample_rate, chunk_filename, start, end))

    return splitted_buffers
# 파일 로딩 및 온셋 분할 함수
def readfile(directory_path, filename,
                detect_onset=True,     # 온셋 분할 여부
                output_dir=None,       # 분할된 파일 저장 경로
                save_splitted_files=False):
    """
    디렉토리에서 오디오 파일을 읽어들여 buffers 리스트 생성.
    """
    import time
    start_time = time.time()

    buffers_pkl_path = filename + '.pkl'
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

        # 워커 수를 2로 제한하여 메모리 사용량 조절
        with Pool(processes=4) as pool:
            loaded_results = list(tqdm(pool.imap(load_audio_file, audio_files),
                                        total=len(audio_files),
                                        desc="Loading audio files"))
        print(f"Loaded {len(loaded_results)} audio buffers in memory.")

        if detect_onset:
            print("Onset detection is enabled. Splitting audio by detected onsets...")
            # ThreadPoolExecutor를 사용하여 onset detection을 병렬 처리
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
    print(f"readfile 처리 시간: {end_time - start_time:.2f}초")
    return buffers

def process_buffer(audio_data, sample_rate, filename):
    """
    각 오디오 버퍼(또는 분할된 구간)에 대해 MFCC, Spectral Centroid, Chroma를 순차적으로 추출.
    """
    # 최소 오디오 신호 길이 체크
    min_fft = 256
    if np.max(np.abs(audio_data)) < 1e-4 or len(audio_data) < min_fft:
        return {
            'filename': filename,
            'mfccs_mean': [],
            'spectral_centroid_mean': 0.0,
            'chroma_mean': []
        }

    # 적절한 fmax 설정 (Nyquist 주파수 이하로)
    fmax_value = min(sample_rate / 2, 8000)  # 8kHz 이상이면 8000으로 제한
    n_mels_value = 40  # 기본값(128)보다 낮게 조정

    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate,
                                 n_mfcc=13, n_fft=256, hop_length=256,
                                 fmax=fmax_value, n_mels=n_mels_value)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Spectral Centroid 추출
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate,
                                                          n_fft=256, hop_length=256)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Chroma 추출 (tuning 파라미터를 명시적으로 설정하여 피치 튜닝을 건너뜀)
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

# 청크 단위 특징 추출을 위한 헬퍼 함수
def chunked_feature_extract(buffers, chunk_size=1000):
    """
    일정 크기(chunk_size)씩 나누어 feature extraction을 수행할 때 사용.
    """
    for i in range(0, len(buffers), chunk_size):
        yield buffers[i:i+chunk_size]

# 비동기 특징 추출 함수
async def async_featureExtract(path):
    """
    비동기적으로 전체 버퍼를 청크 단위로 처리한 후 최종 결과를 하나의 파일에 저장.
    """
    import time, gc
    start_time = time.time()

    buffers_pkl_path = path + '.pkl'
    features_pkl_path = path + '_features.pkl'

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
    chunk_size = 500  # 한 번에 500개씩 처리
    total_chunks = (len(buffers) + chunk_size - 1) // chunk_size
    all_features = []

    for chunk_idx in tqdm(range(total_chunks), desc='전체 청크 처리'):
        chunk = buffers[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        # 워커 수를 8로 제한 (메모리 오버헤드 최소화)
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
        del chunk, tasks, results, chunk_features
        gc.collect()

    # 최종 결과만 저장
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Features extracted and saved to {features_pkl_path}.")

    end_time = time.time()
    print(f"async_featureExtract 처리 시간: {end_time - start_time:.2f}초")

# 메인 실행 블록
if __name__ == "__main__":
    # 1) 디렉토리 내 오디오 파일 로딩 + 온셋 분할 + 피클 저장
    #    'your_audio_directory' 위치에 있는 .wav/.mp3 파일을 모두 읽은 뒤, 온셋 단위로 분할하여 저장
    readfile("./FSD50k.dev_audio", "FSD50k.dev_audio",
             detect_onset=True,            # 온셋 분할 여부
             output_dir="",  # 분할된 파일을 저장할 폴더(생략 가능)
             save_splitted_files=False)     # 실제 파일로도 저장

    # 2) 특징 추출 (비동기로 실행)
    #    분할된 후의 버퍼를 불러와 feature extraction
    asyncio.run(async_featureExtract("FSD50k.dev_audio"))

#!/usr/bin/env python3
# filepath: /Volumes/Workroom_Studio/python/3d-corpus/featureExtractor_test.py

