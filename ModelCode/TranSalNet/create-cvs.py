import os

OUTPUT="datasets"

def createTrainFile():
    f = open(OUTPUT + "/train_ids.csv", "w")
    f.write("image,map,fixation\n")
    for file in os.listdir("datasets/train/train_fixation"):
        file_name = file.split(".")[0]
        f.write(file_name + ".jpg," + file_name + ".png," + file_name + ".bmp\n")
    f.close()

def createValFile():
    f = open(OUTPUT + "/val_ids.csv", "w")
    f.write("image,map,fixation\n")
    for file in os.listdir("datasets/val/val_fixation"):
        file_name = file.split(".")[0]
        f.write(file_name + ".jpg," + file_name + ".png," + file_name + ".bmp\n")
    f.close()

createTrainFile()
createValFile()