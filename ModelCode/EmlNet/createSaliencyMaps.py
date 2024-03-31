import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from skimage import filters
import skimage.io as sio

import resnet
import decoder
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DATASET_INPUT = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
OUTPUT_DIR = "H:/Masters-Thesis/ModelOutputs/EML-Net"

def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred):
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def createSaleincyMap(image_path, output_path):
    start = time.time()
    pil_img = Image.open(image_path).convert('RGB')
    processed = preprocess(pil_img).unsqueeze(0).cuda()

    with torch.no_grad():
        img_feat = img_model(processed, decode=True)
        pla_feat = pla_model(processed, decode=True)
        pred = decoder_model([img_feat, pla_feat])

    pred = pred.squeeze().detach().cpu().numpy()
    pred = post_process(pred)

    print ("Saving prediction", output_path)
    sio.imsave(output_path, pred)

    end = time.time()
    length = end - start
    print("Took", length, "seconds!")

preprocess = transforms.Compose([
    transforms.Resize((480, 640)),
transforms.ToTensor(),
])

img_model = resnet.resnet50("backbone/res_imagenet.pth").cuda().eval()
pla_model = resnet.resnet50("backbone/res_places.pth").cuda().eval()
decoder_model = decoder.build_decoder("backbone/res_decoder.pth", (480, 640), 5, 5).cuda().eval()

print("loaded models")

for category in os.listdir(DATASET_INPUT):
    current_output_dir = OUTPUT_DIR + "/" + category
    makeDir(current_output_dir)
    for image in os.listdir(DATASET_INPUT + "/" + category):
        if image == "Output":
            continue
        print("processing image: " + category + "/" + image)
        image_path = DATASET_INPUT + category + "/" + image
        output_path = current_output_dir + "/" + image
        createSaleincyMap(image_path, output_path)
