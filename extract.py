# extract.py 
# batch process to perform audio feature extracts

import os
import time
import importlib
from tkinter import filedialog
import featureExtract_n as ext

root = Tk()
root.withdraw()

# 폴더 선택 다이얼로그 표시
selected_directory = filedialog.askdirectory(title="Select Folder")

#get the timestamp from time module.
current_time = time.strftime("%Y%m%d-%H%M%S")

print(selected_directory)
#extract folder name
folder_name = os.path.basename(os.path.normpath(selected_directory))

file_name = f"{folder_name}_{current_time}"
buffer = ext.readfile(selected_directory, file_name)
ext.featureExtract(file_name)