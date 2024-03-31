import cv2
import os
train_path = "datasets/train/train_fixation"
val_path = "datasets/val/val_fixation"

for f in os.listdir(train_path):
    img = cv2.imread(train_path + "/" + f, 0)
    cv2.imwrite(train_path + "/" + f, img)

for f in os.listdir(val_path):
    img = cv2.imread(val_path + "/" + f, 0)
    cv2.imwrite(val_path + "/" + f, img)