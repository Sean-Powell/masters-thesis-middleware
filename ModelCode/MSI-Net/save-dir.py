import os
import shutil

DATASET_PATH = "results/images"
OUTPUT_FILE = "H:/Masters-Thesis/ModelOutputs/MSI-Net-Trained"

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

makeDir(OUTPUT_FILE)
for f in os.listdir(DATASET_PATH):
    name = f.split("_")[1]
    category = f.split("_")[0]
    print("Moving: " + f)
    makeDir(OUTPUT_FILE + "/" + category)
    shutil.copy(DATASET_PATH + "/" + f, OUTPUT_FILE + "/" + category + "/" + name)