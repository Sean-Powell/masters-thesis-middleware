import os

import numpy as np
import scipy.io
import cv2
import os

# Load MATLAB file

def convertMatFileToBmp(path, file):
    mat = scipy.io.loadmat(path + "/" + file)
    mat_resolution = mat['resolution']
    mat_gaze = mat['gaze']

    img = np.zeros((mat_resolution[0][0], mat_resolution[0][1]))
    for n in range(len(mat_gaze) - 1):
        gaze_data = mat_gaze[n]
        fixations = gaze_data['fixations'][0]
        for i in range(int(len(fixations) / 2) - 1):
            try:
                x, y = fixations[i]
                img[y][x] = 255
            except:
                continue

    bmp_file = path + "/" + file.split(".")[0] + ".bmp"
    cv2.imwrite(bmp_file, img)
    os.remove(path + "/" + file)

train_path = "datasets/train/train_fixation"
val_path = "datasets/val/val_fixation"

for f in os.listdir(train_path):
    if ".bmp" in f:
        continue
    convertMatFileToBmp(train_path, f)

for f in os.listdir(val_path):
    if ".bmp" in f:
        continue
    convertMatFileToBmp(val_path, f)

print("done")