import os
import numpy as np
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt

#DATASET_PATH = "C:/Users/seanp/Desktop/MIT2000/FIXATIONMAPS"
PREDICTION_PATH = "H:\Masters-Thesis\ModelOutputs\DeepGazeIIE"
DATASET_PATH = "H:/Masters-Thesis/masters-thesis-middleware/Dataset/Yes"

# from https://github.com/MemoonaTahira/Visual-Saliency-Metrics-for-Evaluating-Deep-Learning-Model-performance/blob/main/_visual_attention_metrics.py
def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = np.nan
        return score

    # make sure maps have the same shape
    #from scipy.misc import imresize

    new_size = np.shape(fixationMap)
    map1 = np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

    #map1 = imresize(saliencyMap, np.shape(fixationMap))
    if not map1.max() == 0:
        map1 = map1.astype(float) / map1.max()

    # normalize saliency map
    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1)

    # mean value at fixation locations
    score = map1[fixationMap.astype(bool)].mean()

    return score


def AUC_Judd(saliencyMap, fixationMap, plotName, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap
    new_size = np.shape(fixationMap)
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        #from scipy.misc import imresize
        new_size = np.shape(fixationMap)
        np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

        #saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plotName = plotName.replace("/", "_")
        plt.savefig("Results/" + plotName)

    return score

def binaryFixationMap(path):
    image = cv2.imread(path, 0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < 127:
                image[i, j] = 0
            else:
                image[i, j] = 1
    return image

def calculateNSS(path):
    try:
        saliencyMap = cv2.imread(PREDICTION_PATH + "/" + path, 0)
        fixationMap = DATASET_PATH + "/" + path
        fixationMap = binaryFixationMap(fixationMap)
        nssScore = NSS(saliencyMap, fixationMap)
        print(path + ": " + str(nssScore))
        return nssScore
    except:
        return 0

def calculateAUC(path):
    saliencyMap = cv2.imread(PREDICTION_PATH + "/" + path, 0)
    fixationMap = DATASET_PATH + "/" + path
    fixationMap = binaryFixationMap(fixationMap)
    aucScore = AUC_Judd(saliencyMap, fixationMap, path)
    print(path + ": " + str(aucScore))
    return aucScore


def scoreDataset():
    start = time.time()
    f = open("deepgaze-IIE-result-Yes.txt", "w")
    f.write(PREDICTION_PATH + "\n")
    nssScores = []
    for category in os.listdir(PREDICTION_PATH):
        for image in os.listdir(PREDICTION_PATH + "/" + category):
            nssScore = calculateNSS(category + "/" + image)
            nssScores.append(nssScore)
            f.write(category + "/" + image + "," + str(nssScore) + "\n")

    aucScores = []
    #aucScore = calculateAUC("Action/001.jpg")
    #aucScores.append(aucScore)

    end = time.time()
    print("NSS: " + str(np.mean(nssScores)))
    #print("AUC: " + str(np.mean(aucScores)))
    timeTaken = end - start
    print("Time taken:", timeTaken, "seconds")
    f.write("NSS: " + str(np.mean(nssScores)) + "\n")
    #f.write("AUC: " + str(np.mean(aucScores)) + "\n")
    f.close()

scoreDataset()