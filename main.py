import os
from os.path import isfile, join
import cv2
import pandas as pd
import configparser
import time

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

videonames = ["childrenballbath","childrenparty","childrenplay","childrenplaying","S54surprised"]
videofpscount = [50, 30, 30, 25, 25]
eyeTrackerSamplingRate = 300  #default setting
validityPercentile = float(config['SETTINGS']['Cutoff'])
kernelsigma = float(config['SETTINGS']['KernelSigma'])


if __name__ == '__main__':
    start_time = time.time()
    if config['SETTINGS']['DataPreProcession'] == "True":
        print("Gazedata Vorverarbeitung läuft...")
        gazedata_preprocession.pipeline()
        print("Gazedata Vorverarbeitung abgeschlossen.")
    end_time = time.time()
    execution_time = end_time - start_time
    with open('execution_time_gazedata_preprocession.txt', 'a') as file:
        file.write(f'{gazedata_preprocession}: {execution_time:.6f} seconds\n')

    start_time = time.time()
    if config['SETTINGS']['SplitVideo'] == "True":
        print("Video wird in Frames aufgeteilt...")
        video_procession.splitVideosToFrames()
        print("Aufteilung in Frames abgeschlossen.")
    end_time = time.time()
    execution_time = end_time - start_time
    with open('execution_time_video_splitting.txt', 'a') as file:
        file.write(f'{video_splitting}: {execution_time:.6f} seconds\n')
    
    start_time = time.time()
    if config['SETTINGS']['CreateIttiKochFrames'] == "True":
        print("Saliency Maps werden erstellt...")
        ittiKochFrameProcession.ittiKochFrameProcession()
        print("Erstellen der Saliency Maps abgeschlossen.")
    end_time = time.time()
    execution_time = end_time - start_time
    with open('execution_time_saliency_maps.txt', 'a') as file:
        file.write(f'{saliency_maps}: {execution_time:.6f} seconds\n')
    
    start_time = time.time() 
    
    filesToProcess = [f for f in os.listdir(matchedDataPath) if isfile(join(matchedDataPath, f))]
    keyList = [f.split(".")[0].split("_")[-1] for f in filesToProcess]
    keyList = list(dict.fromkeys(keyList))
    print(filesToProcess)
    videoframes=[]
    for key in keyList:
        frames = [f for f in os.listdir(ittiKochImagesPath) if "_"+key+"_" in f]
        videoframes.append([key,frames])

    #print(videoframes[keyList.index("childrenplaying")])

    ittiKochFrames = []
    finalFrameList = []

    for i in range(len(keyList)):
        for f in videoframes[i][1]:
            ittiKochFrames.append(cv2.imread(ittiKochImagesPath + "\\" + f))
            print("Füge", f, "zur Liste hinzu.")
        finalFrameList.append([keyList[i], ittiKochFrames])
        ittiKochFrames = []

    filesToProcess=filesToProcess[:10]

    for item in filesToProcess: #indent verändert
        matchedDataName = matchedDataPath + item
        print(matchedDataName)
        proc = GazeBehaviourProcession(matchedDataName)
        df = pd.read_csv(matchedDataName, header=None)
        print(videofpscount[videonames.index(matchedDataName.split("_")[-1].split(".")[0])])
        fpscount = videofpscount[videonames.index(matchedDataName.split("_")[-1].split(".")[0])]
        length = len(df.index)/(eyeTrackerSamplingRate / fpscount)
        proc.matchGazeDataToFrame(finalFrameList[keyList.index(item.split(".")[0].split("_")[-1])][1],0,int(length))
    
    end_time = time.time()
    execution_time = end_time - start_time
    with open('execution_time_10_result_files.txt', 'a') as file:
        file.write(f'{result_files}: {execution_time:.6f} seconds\n')
    
