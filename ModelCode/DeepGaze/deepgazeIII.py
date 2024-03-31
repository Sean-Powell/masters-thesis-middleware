import numpy as np
import torch
import imageio
import cv2
from scipy.ndimage import zoom
from scipy.special import logsumexp
import deepgaze_pytorch
import os

DEVICE = 'cuda'

DATASET_INPUT = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
OUTPUT_DIR = "H:/Masters-Thesis/ModelOutputs/DeepGazeIII"

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def createSaleincyMap(image_path, output_path):
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
    image = imageio.imread(image_path)
    # location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
    fixation_history_x = np.array([1024 // 2, 300, 500, 200, 200, 700])
    fixation_history_y = np.array([768 // 2, 300, 100, 300, 100, 500])

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
    centerbias_template = np.load('centerbias_mit1003.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template,
                      (image.shape[0] / centerbias_template.shape[0], image.shape[1] / centerbias_template.shape[1]),
                      order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
    x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_y[model.included_fixations]]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    x = log_density_prediction.detach().cpu().numpy()[0, 0]
    img = (255 * (x - np.min(x)) / np.ptp(x)).astype(int)
    cv2.imwrite(output_path, img)


for category in os.listdir(DATASET_INPUT):
    current_output_dir = OUTPUT_DIR + "/" + category
    makeDir(current_output_dir)
    for image in os.listdir(DATASET_INPUT + "/" + category):
        if image == "Output":
            continue
        print("processing image: " + category + "/" + image)
        image_path = DATASET_INPUT + category + "/" + image
        output_path = current_output_dir + "/" + image
        try:
            createSaleincyMap(image_path, output_path)
        except:
            print("Failed to process: " + category + "/" + image)