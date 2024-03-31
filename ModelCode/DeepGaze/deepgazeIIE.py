import numpy as np
import torch
import imageio
import cv2
from scipy.ndimage import zoom
from scipy.special import logsumexp
import deepgaze_pytorch
import os

DEVICE = 'cuda'
#DEVICE = 'cpu'

DATASET_INPUT = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
OUTPUT_DIR = "H:/Masters-Thesis/ModelOutputs/DeepGazeIIE"

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def createSaleincyMap(image_path, output_path):
    centerbias_template = np.load('centerbias_mit1003.npy')
    centerbias_template_shape_zero = centerbias_template.shape[0]
    centerbias_template_shape_one = centerbias_template.shape[1]

    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
    image = imageio.imread(image_path)

    centerbias = zoom(centerbias_template, (image.shape[0] / centerbias_template_shape_zero,
                                            image.shape[1] / centerbias_template_shape_one),
                      order=0,
                      mode='nearest')

    centerbias -= logsumexp(centerbias)
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)
    x = log_density_prediction.detach().cpu().numpy()[0, 0]
    img = (255 * (x - np.min(x)) / np.ptp(x)).astype(int)
    cv2.imwrite(output_path, img)


for category in os.listdir(DATASET_INPUT):
    current_output_dir = OUTPUT_DIR + "/" + category
    makeDir(current_output_dir)
    for image in os.listdir(DATASET_INPUT + "/" + category):
        if image == "Output":
            continue
        image_path = DATASET_INPUT + category + "/" + image
        output_path = current_output_dir + "/" + image
        try:
            createSaleincyMap(image_path, output_path)
        except:
            print("Failed to process: " + category + "/" + image)