import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.data_process import preprocess_img, postprocess_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_INPUT = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

flag = 1 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

if flag:
    from TranSalNet_Res import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Res.pth'))
    OUTPUT_DIR = "H:/Masters-Thesis/ModelOutputs/TrainSalNet_Res"
else:
    from TranSalNet_Dense import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Dense.pth'))
    OUTPUT_DIR = "H:/Masters-Thesis/ModelOutputs/TrainSalNet_Dense"

model = model.to(device)
model.eval()

def createSaleincyMap(input_path, output_path):
    img = preprocess_img(input_path) # padding and resizing input image into 384x288
    img = np.array(img)/255.
    img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
    img = torch.from_numpy(img)
    img = img.type(torch.cuda.FloatTensor).to(device)
    pred_saliency = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())
    pred_saliency = postprocess_img(pic, input_path) # restore the image to its original size as the result
    cv2.imwrite(output_path, pred_saliency, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) # save the result


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