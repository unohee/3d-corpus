import os
import sys
import asyncio
from feature_extractor import readfile, async_featureExtract

def main():
    if len(sys.argv) >= 2:
        folder = sys.argv[1]
    else:
        folder = input("분석할 폴더 경로를 입력하세요: ")

    # 폴더 이름을 기반으로 피클 파일명을 결정 (예: my_folder.pkl, my_folder_features.pkl)
    folder_name = os.path.basename(os.path.normpath(folder))
    
    print(f"Processing folder: {folder}")
    # 1) 폴더 내 오디오 파일 로딩 + 온셋 분할 + 피클 저장
    readfile(
        directory_path=folder,
        filename=folder_name,
        detect_onset=True,
        output_dir="splitted_files",
        save_splitted_files=True
    )
    
    # 2) 분할된 버퍼에 대해 특징 추출 (비동기)
    asyncio.run(async_featureExtract(folder_name))

if __name__ == "__main__":
    main()
