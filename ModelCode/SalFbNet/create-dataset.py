import os
import shutil

DATASET_INPUT = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
OUTPUT = "H:/Masters-Thesis/Models/SalFBNet/Datasets/CAT2000"
for category in os.listdir(DATASET_INPUT):
    for image in os.listdir(DATASET_INPUT + "/" + category):
        if image == "Output":
            continue
        print("processing image: " + category + "/" + image)
        src = DATASET_INPUT + "/" + category + "/" + image
        out = OUTPUT + "/" + category + "_" + image
        shutil.copy(src, out)