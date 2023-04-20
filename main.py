import os
from os.path import isfile, join
import cv2
import pandas as pd
import configparser

import gazedata_preprocession
from GazeBehaviourProcession import GazeBehaviourProcession
import ittiKochFrameProcession
import video_procession

config = configparser.ConfigParser()
config.read('config.properties')

projectPath = os.getcwd() + "\\"
dataPath = projectPath + "venv\\data\\"
gazeDataPath = dataPath + "gazedata\\"
videoDataPath = dataPath + "videos\\"
matchedDataPath = gazeDataPath + "combined_gazedata\\video_matched_data\\"
ittiKochImagesPath = videoDataPath + "IttiKochImages\\"
videonames = [f.split(".")[0] for f in os.listdir(videoDataPath) if isfile(join(videoDataPath, f))]
videofpscount = [round(cv2.VideoCapture(videoDataPath + f + ".mp4").get(cv2.CAP_PROP_FPS)) for f in videonames]

eyeTrackerSamplingRate = 300  #default setting
validityPercentile = float(config['SETTINGS']['Cutoff'])
kernelsigma = float(config['SETTINGS']['KernelSigma'])


if __name__ == '__main__':
    if config['SETTINGS']['DataPreProcession'] == "True":
        print("Processing raw gazedata ...")
        gazedata_preprocession.pipeline()
        print("Gazedata processed.")

    if config['SETTINGS']['SplitVideo'] == "True":
        print("Splitting videos into frames...")
        video_procession.splitVideosToFrames()
        print("Video splitting finished.")


    if config['SETTINGS']['CreateIttiKochFrames'] == "True":
        print("Creating saliency maps...")
        ittiKochFrameProcession.ittiKochFrameProcession()
        print("Creation of saliency maps finished.")


    filesToProcess = [f for f in os.listdir(matchedDataPath) if isfile(join(matchedDataPath, f))]

    keyList = [f.split(".")[0].split("_")[-1] for f in filesToProcess]
    keyList = list(dict.fromkeys(keyList))

    videoframes=[]
    for key in keyList:
        frames = [f for f in os.listdir(ittiKochImagesPath) if "_"+key+"_" in f]
        videoframes.append([key,frames])

    ittiKochFrames = []
    finalFrameList = []

    for i in range(len(keyList)):
        for f in videoframes[i][1]:
            ittiKochFrames.append(cv2.imread(ittiKochImagesPath + "\\" + f))
            print("Adding", f, "to the list.")
        finalFrameList.append([keyList[i], ittiKochFrames])
        ittiKochFrames = []



    filesToProcesssize=len(filesToProcess)


    for item in filesToProcess: #indent verändert
        matchedDataName = matchedDataPath + item
        print(matchedDataName)
        proc = GazeBehaviourProcession(matchedDataName)
        df = pd.read_csv(matchedDataName, header=None)
        print(videofpscount[videonames.index(matchedDataName.split("_")[-1].split(".")[0])])
        fpscount = videofpscount[videonames.index(matchedDataName.split("_")[-1].split(".")[0])]
        length = len(df.index)/(eyeTrackerSamplingRate / fpscount)
        proc.matchGazeDataToFrame(finalFrameList[keyList.index(item.split(".")[0].split("_")[-1])][1],0,int(length))
