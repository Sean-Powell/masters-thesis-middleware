import boto3
import numpy as np
import cv2
import json
import os
import scipy
import shutil

USER_TABLE = "masters-thesis-backend-UserTable-JWKQOMYY4MXJ"

SURVEY_TABLE = "masters-thesis-backend-SurveyTable-8YOCOAJ0ILX5"
SURVEY_BUCKET = "masters-thesis-backend-surveybuc-309171779279"

RESULT_TABLE = "masters-thesis-backend-ResultTable-1SZF9195KO5WN"
RESULT_BUCKET = "masters-thesis-backend-resultbuc-309171779279"

IMAGE_TABLE = "masters-thesis-backend-ImageTable-1U8PS97CO9G2P"
IMAGE_PATH = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
OUTPUT_PATH = "Dataset"
FIXATION_PATH = "fixations"

s3 = boto3.client("s3")
dynamo = boto3.client("dynamodb")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


class FixationPoints:
    def __init__(self, startTime, fixationPoints, imgData):
        self.startTime = startTime
        self.fixationPoints = fixationPoints
        self.imgData = imgData


class SurveyData:
    def __init__(self, gender, age, hasImpairment, impairmentType):
        self.gender = gender
        self.age = age
        self.hasImpairment = hasImpairment

        if hasImpairment != 'No':
            self.nearsighted = impairmentType[0]
            self.farsighted = impairmentType[1]
            self.Presbyopia = impairmentType[2]
            self.Astigmatism = impairmentType[3]
            self.DiabeticRetinopathy = impairmentType[4]
            self.AMD = impairmentType[5]
            self.Glaucoma = impairmentType[6]
            self.Cataract = impairmentType[7]
            self.ColourBlindness = impairmentType[8]
            self.Other = impairmentType[9]


def readFixationPoints(content):
    fixationPoints = content.split('},{')

    # extract the start time
    startTime = fixationPoints[0].split(":[{")[0]
    startTime = int(startTime[13:len(startTime) - 17])

    # extract the img data
    imgData = fixationPoints[len(fixationPoints) - 1]
    fixationPoints[len(fixationPoints) - 1] = imgData.split("}],")[0]
    imgData = imgData.split("}],")[1].split(":{")[1]
    imgData = imgData[:len(imgData) - 2]
    imgData = json.loads('{{{}}}'.format(imgData))
    fixationPoints[0] = fixationPoints[0].split(":[{")[1]

    points = []
    for fixationPoint in fixationPoints:
        points.append(json.loads('{{{}}}'.format(fixationPoint)))

    return FixationPoints(startTime, points, imgData)


def parseFixationPoints(data: FixationPoints):
    points = data.fixationPoints
    parsedPoints = []
    currentTime = data.startTime
    for point in points:
        pointTime = int(point["t"])
        timeAtPoint = pointTime - currentTime  # time in ms
        currentTime = pointTime
        # if (timeAtPoint < 2):
        #     continue
        parsedPoint = {}
        parsedPoint["x"] = int(point["x"])
        parsedPoint["y"] = int(point["y"])
        parsedPoint["t"] = timeAtPoint
        parsedPoints.append(parsedPoint)

    return parsedPoints


def scaleFixationPoints(parsedPoints, originalImageDate, displayedImageData):
    widthScale = originalImageDate[1] / displayedImageData['width']
    heightScale = originalImageDate[0] / displayedImageData['height']

    scaledPoints = []

    for point in parsedPoints:
        scaledPoints.append({
            'x': int(point['x'] * widthScale),
            'y': int(point['y'] * heightScale),
            't': point['t']
        })

    return scaledPoints


def createSaliencyMap(parsedPoints, path, imageData, outputPath, noUsers):
    img = cv2.imread(path, 0)
    height, width = img.shape
    img_size = (height, width)

    scaledPoints = scaleFixationPoints(parsedPoints, img_size, imageData)

    saliency_map = np.zeros(img_size)

    for i in range(len(scaledPoints)):
        x = scaledPoints[i]["x"]
        y = scaledPoints[i]["y"]
        duration = scaledPoints[i]["t"]

        radius = int((duration / 80) / noUsers)
        brightness = (duration / 20) / noUsers
        saliency_map = cv2.circle(saliency_map, (x, y), radius, brightness, -1)

    saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigmaX=40, sigmaY=40)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(outputPath, saliency_map)


def getBucketKeys(bucket):
    response = s3.list_objects_v2(Bucket=bucket)
    keys = []
    if 'Contents' in response:
        for obj in response['Contents']:
            keys.append(obj['Key'])
    else:
        return keys

    while response.get('NextContinuationToken'):
        response = s3.list_objects_v2(
            Bucket=bucket,
            ContinuationToken=response['NextContinuationToken']
        )
        for obj in response['Contents']:
            keys.append(obj['Key'])

    return keys


def getBucketItem(bucket, key):
    objData = s3.get_object(Bucket=bucket, Key=key)
    return objData['Body'].read().decode('utf-8')


def getAssociatedUserUUID(key):
    response = dynamo.scan(
        TableName=SURVEY_TABLE,
        FilterExpression=f'#attr = :val',
        ExpressionAttributeNames={'#attr': 'Key'},
        ExpressionAttributeValues={':val': {'S': key}}
    )

    items = response['Items']
    if items:
        for item in items:
            return item['UUID']["S"]
    else:
        return None


def parseSurveyResults(keys):
    parsedData = {}
    for k in keys:
        data = json.loads('{}'.format(getBucketItem(SURVEY_BUCKET, k)))
        uuid = getAssociatedUserUUID(k)
        parsed = SurveyData(data['gender'], data["age"], data["doYouHaveVisualImpairment"],
                            data["visualImpairmentType"])
        parsedData[uuid] = parsed
    return parsedData


def getResultsFromUser(user):
    queryParams = {
        'TableName': RESULT_TABLE,
        'KeyConditionExpression': f'#att = :val',
        'ExpressionAttributeNames': {'#att': 'UUIID'},
        'ExpressionAttributeValues': {':val': {'S': user}}
    }
    response = dynamo.query(**queryParams)

    allItems = []
    if 'Items' in response:
        allItems.extend(response['Items'])
    else:
        print("User " + user + " did not provide any results")
        return allItems

    while 'LastEvaluatedKey' in response:
        queryParams['ExclusiveStartKey'] = response['LastEvaluatedKey']
        response = dynamo.query(**queryParams)
        if 'Items' in response:
            allItems.extend(response['Items'])

    return allItems


def getResultsByImageId(imageId):
    queryParams = {
        'TableName': RESULT_TABLE,
        'IndexName': 'ImageId-index',
        'KeyConditionExpression': f'#att = :val',
        'ExpressionAttributeNames': {'#att': 'ImageId'},
        'ExpressionAttributeValues': {':val': {'S': imageId}}
    }

    response = dynamo.query(**queryParams)
    items = response['Items']
    returnItems = []
    for item in items:
        key = item['Key']['S']
        uuid = item['UUIID']['S']
        returnItems.append((key, uuid))

    return returnItems


def getImagePath(imageId):
    queryParams = {
        'TableName': IMAGE_TABLE,
        'KeyConditionExpression': f'#att = :val',
        'ExpressionAttributeNames': {'#att': 'ImageId'},
        'ExpressionAttributeValues': {':val': {'S': imageId}}
    }
    response = dynamo.query(**queryParams)

    items = response['Items']
    for item in items:
        return item['key']['S']


def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def createTagFile(path, surveyResults: SurveyData):
    filePath = path + "/tags.txt"
    f = open(filePath, "w")

    f.write("Age: " + surveyResults.age + "\n")
    f.write("Gender: " + surveyResults.gender + "\n")

    if surveyResults.hasImpairment != "No":
        f.write("Nearsighted: " + str(surveyResults.nearsighted) + "\n")
        f.write("Farsighted: " + str(surveyResults.farsighted) + "\n")
        f.write("Presbyopia: " + str(surveyResults.Presbyopia) + "\n")
        f.write("Astigmatism: " + str(surveyResults.Astigmatism) + "\n")
        f.write("DiabeticRetinopathy: " + str(surveyResults.DiabeticRetinopathy) + "\n")
        f.write("Age-Related Macular Degeneration: " + str(surveyResults.AMD) + "\n")
        f.write("Glaucoma: " + str(surveyResults.Glaucoma) + "\n")
        f.write("Cataract: " + str(surveyResults.Cataract) + "\n")
        f.write("ColourBlindness: " + str(surveyResults.ColourBlindness) + "\n")
        if surveyResults.Other == "":
            f.write("Other: None\n")
        else:
            f.write("Other: " + surveyResults.Other + "\n")
    f.close()


def createFixationPoints(fixationPoints, path, outputPath):
    category = path.split("/")[0]
    image = path.split("/")[1]
    fixationPointsArray = np.zeros((1080, 1920))
    outputPath = outputPath + category
    makeDir(outputPath)
    outputPath = outputPath + "/" + image + ".mat"
    for fixation in fixationPoints:
        fixationPointsArray[fixation['y'], fixation['x']] = 1

    fixationPointsArray = fixationPointsArray.astype(bool)
    scipy.io.savemat(outputPath, {'fixLocs': fixationPointsArray})

def copyStimuli(imagePath, outputPath):
    shutil.copy(imagePath, outputPath)


def createSaliencyMaps():
    surveyResultKeys = getBucketKeys(SURVEY_BUCKET)
    parsedSurveyResults = parseSurveyResults(surveyResultKeys)

    for i in range(1, 2000):
        results = getResultsByImageId(str(i))
        imageName = getImagePath(str(i))
        imagePath = IMAGE_PATH + imageName

        if len(results) == 0:
            print("no results for " + str(i))
            continue
        yesFixationPoints = []
        noFixationPoints = []
        combinedFixationPoints = []
        yesUsers = 0
        noUsers = 0
        imgData = ""
        for result in results:
            key, uuid = result
            try:
                resultData = getBucketItem(RESULT_BUCKET, key)
                fixationPoints = readFixationPoints(resultData)
                imgData = fixationPoints.imgData
                parsedPoints = parseFixationPoints(fixationPoints)
                surveyResult = "No"
                try:
                    surveyResult = parsedSurveyResults[uuid].hasImpairment
                except:
                    surveyResult = "No"

                if surveyResult == "Yes":
                    yesUsers += 1
                    yesFixationPoints.extend(parsedPoints)
                else:
                    noUsers += 1
                    noFixationPoints.extend(parsedPoints)
            except:
                print("error in fixation points")
                continue

        image = imageName.split(".")[0]
        if len(noFixationPoints) != 0:
            print("has no fixation points")
            currentPath = OUTPUT_PATH + "/No/"
            makeDir(currentPath)
            saliencyPath = currentPath + "saliency/"
            makeDir(saliencyPath)
            fixationPath = currentPath + "fixations/"
            makeDir(fixationPath)
            stimuliPath = currentPath + "stimuli/"
            makeDir(stimuliPath)
            imageFolder = imageName.split("/")[0]
            saliencyPath = saliencyPath + imageFolder
            stimuliPath = stimuliPath + imageFolder
            makeDir(saliencyPath)
            makeDir(stimuliPath)
            createSaliencyMap(noFixationPoints, imagePath, imgData, saliencyPath + "/" + imageName.split("/")[1],
                              noUsers)
            createFixationPoints(noFixationPoints, image, fixationPath)
            copyStimuli(imagePath, stimuliPath + "/" + imageName.split("/")[1])
        if len(yesFixationPoints) != 0:
            print("has yes fixation points")
            currentPath = OUTPUT_PATH + "/Yes/"
            makeDir(currentPath)
            saliencyPath = currentPath + "saliency/"
            makeDir(saliencyPath)
            fixationPath = currentPath + "fixations/"
            makeDir(fixationPath)
            stimuliPath = currentPath + "stimuli/"
            makeDir(stimuliPath)
            imageFolder = imageName.split("/")[0]
            saliencyPath = saliencyPath + imageFolder
            stimuliPath = stimuliPath + imageFolder
            makeDir(saliencyPath)
            makeDir(stimuliPath)

            createSaliencyMap(yesFixationPoints, imagePath, imgData, saliencyPath + "/" + imageName.split("/")[1],
                              yesUsers)#
            createFixationPoints(yesFixationPoints, image, fixationPath)
            copyStimuli(imagePath, stimuliPath + "/" + imageName.split("/")[1])





createSaliencyMaps()
