import os
import shutil
import random
from math import floor

TRAINING_AMOUNT = 70
TESTING_AMOUNT = 20
VALIDATION_AMOUNT = 10

OUTPUT = "SPLIT_DATASET"
INPUT = "Dataset"

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def createSets (path):
    makeDir(path)
    makeDir(path + "/fixations")
    makeDir(path + "/saliency")
    makeDir(path + "/stimuli")

def createCategory (root_path, category):
    makeDir(root_path + "/val/fixations/" + category)
    makeDir(root_path + "/val/saliency/" + category)
    makeDir(root_path + "/val/stimuli/" + category)
    makeDir(root_path + "/train/fixations/" + category)
    makeDir(root_path + "/train/saliency/" + category)
    makeDir(root_path + "/train/stimuli/" + category)
    makeDir(root_path + "/test/fixations/" + category)
    makeDir(root_path + "/test/saliency/" + category)
    makeDir(root_path + "/test/stimuli/" + category)

def copyToDataset(input_path, output_path, img, category,  dataset):
    stimuli_path = input_path + "/stimuli/" + category + "/"+ img
    fixation_path = input_path + "/fixations/" + category + "/" + img.split(".")[0] + ".mat"
    saliency_path = input_path + "/saliency/" + category + "/" + img

    if dataset == "val":
        stimuli_output = output_path + "/val/stimuli/" + category + "/" + img
        fixation_output = output_path + "/val/fixations/" + category + "/" + img.split(".")[0] + ".mat"
        saliency_output = output_path + "/val/saliency/" + category + "/" + img
    elif dataset == "train":
        stimuli_output = output_path + "/train/stimuli/" + category + "/" + img
        fixation_output = output_path + "/train/fixations/" + category + "/" + img.split(".")[0] + ".mat"
        saliency_output = output_path + "/train/saliency/" + category + "/" + img
    else:
        stimuli_output = output_path + "/test/stimuli/" + category + "/" + img
        fixation_output = output_path + "/test/fixations/" + category + "/" + img.split(".")[0] + ".mat"
        saliency_output = output_path + "/test/saliency/" + category + "/" + img

    shutil.copy(stimuli_path, stimuli_output)
    shutil.copy(fixation_path, fixation_output)
    shutil.copy(saliency_path, saliency_output)


makeDir(OUTPUT)
for dataset in os.listdir(INPUT):
    source_path = INPUT + "/" + dataset
    current_path = OUTPUT + "/" + dataset
    makeDir(current_path)
    createSets(current_path + "/val/")
    createSets(current_path + "/train/")
    createSets(current_path + "/test/")
    root_path = current_path
    for category in os.listdir(source_path + "/stimuli"):
        createCategory(root_path, category)
        current_source = source_path + "/stimuli/" + category
        images = []
        for img in os.listdir(current_source):
            images.append(img)

        size = len(images)

        if size == 1:
            copyToDataset(source_path, root_path, images[0], category, "train")
        elif size == 2:
            copyToDataset(source_path, root_path, images[0], category, "train")
            copyToDataset(source_path, root_path, images[1], category, "test")
        elif size == 3:
            copyToDataset(source_path, root_path, images[0], category, "train")
            copyToDataset(source_path, root_path, images[1], category, "train")
            copyToDataset(source_path, root_path, images[2], category, "train")
        else:
            train_size = floor(size * (TRAINING_AMOUNT / 100))
            test_size = floor(size * (TESTING_AMOUNT / 100))
            val_size = floor(size * (VALIDATION_AMOUNT / 100))

            for i in range(train_size):
                index = random.randint(0, len(images) - 1)
                copyToDataset(source_path, root_path,  images[index], category, "train")
                images.remove(images[index])

            for i in range(test_size):
                index = random.randint(0, len(images) - 1)
                copyToDataset(source_path, root_path, images[index], category, "test")
                images.remove(images[index])

            for i in range(val_size):
                index = random.randint(0, len(images) - 1)
                copyToDataset(source_path, root_path, images[index], category, "val")
                images.remove(images[index])

            while len(images) is not 0:
                index = random.randint(0, len(images) - 1)
                copyToDataset(source_path, root_path,  images[index], category, "train")
                images.remove(images[index])


