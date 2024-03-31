import os
import shutil

parseRes18 = False

if parseRes18:
    DATASET_FILE = "results_Res18/Test_CAT2000"
    OUTPUT_FILE = "H:/Masters-Thesis/ModelOutputs/SalFBNet-Res18"
else:
    DATASET_FILE = "results_Res18Fixed/Test_CAT2000"
    OUTPUT_FILE = "H:/Masters-Thesis/ModelOutputs/SalFBNet-Res18Fixed"


def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


makeDir(OUTPUT_FILE)
for f in os.listdir(DATASET_FILE):
    name = f.split("_")[1]
    category = f.split("_")[0]
    print("Moving: " + f)
    makeDir(OUTPUT_FILE + "/" + category)
    shutil.copy(DATASET_FILE + "/" + f, OUTPUT_FILE + "/" + category + "/" + name)


