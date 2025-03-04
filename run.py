import sys
import os
import asyncio
from featureExtractor import readfile, async_featureExtract

def main():
    try:
        if len(sys.argv) >= 2:
            folder = sys.argv[1]
        else:
            folder = "./FSD50k.dev_audio"

        # 폴더 이름을 기반으로 피클 파일명을 결정 (예: my_folder.pkl, my_folder_features.pkl)
        folder_name = os.path.basename(os.path.normpath(folder))
        
        print(f"Processing folder: {folder}")
        # 1) 폴더 내 오디오 파일 로딩 + 온셋 분할 + 피클 저장
        readfile(
            directory_path=folder,
            filename=folder_name,
            detect_onset=True,
            output_dir="splitted_files",
            save_splitted_files=False
        )
        
        # 2) 분할된 버퍼에 대해 특징 추출 (비동기)
        asyncio.run(async_featureExtract(folder_name))
    except Exception as e:
        print("프로그램 실행 중 예외 발생:", e)
        raise e

if __name__ == "__main__":
    main()