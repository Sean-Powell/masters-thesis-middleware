import os
import cv2
DATASET = "H:/Masters-Thesis/ModelOutputs/MSI-Net-Trained"
for category in os.listdir(DATASET):
    for image in os.listdir(DATASET + "/" + category):
        img = cv2.imread(DATASET + "/" + category + "/" + image)
        newName = image.split(".")[0] + ".jpg"
        cv2.imwrite(DATASET + "/" + category + "/" + newName, img)
        os.remove(DATASET + "/" + category + "/" + image)
