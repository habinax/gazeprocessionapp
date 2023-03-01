import pandas as pd
import configparser
import scipy.stats as st

import cv2
import numpy as np

import gazedata_preprocession
from GazeBehaviourProcession import GazeBehaviourProcession
import ittiKochFrameProcession
import pySaliencyMapDefs
import video_procession

config = configparser.ConfigParser()
config.read('config.properties')

projectPath = config['SETTINGS']['ProjectPath']
dataPath = projectPath + "venv\\data\\"
gazeDataPath = dataPath + "gazedata\\"
videoDataPath = dataPath + "videos\\"
matchedDataPath = gazeDataPath + "combined_gazedata\\video_matched_data\\"

combineFiles = config['SETTINGS']['CombineFiles']

videonames = ["childrenballbath","childrenparty","childrenplay","childrenplaying","S54surprised"]
videofpscount = [50, 30, 30, 25, 25]
eyeTrackerSamplingRate = 300  #default setting
validityPercentile = float(config['SETTINGS']['Cutoff'])
kernelsigma = float(config['SETTINGS']['KernelSigma'])


if __name__ == '__main__':
    if config['SETTINGS']['DataPreProcession'] == "True":
        print("Gazedata Vorverarbeitung läuft...")
        gazedata_preprocession.pipeline()
        print("Gazedata Vorverarbeitung abgeschlossen.")

    if config['SETTINGS']['SplitVideo'] == "True":
        print("Video wird in Frames aufgeteilt...")
        video_procession.splitVideosToFrames()
        print("Aufteilung in Frames abgeschlossen.")

    if config['SETTINGS']['CreateIttiKochFrames'] == "True":
        print("Saliency Maps werden erstellt...")
        ittiKochFrameProcession.ittiKochFrameProcession()
        print("Erstellen der Saliency Maps abgeschlossen.")

    print("Gude")
    matchedDataName = matchedDataPath + "005_t4_childrenballbath.csv"
    proc = GazeBehaviourProcession(matchedDataName)
    df = pd.read_csv(matchedDataName, header=None)
    length = len(df.index)/(eyeTrackerSamplingRate / videofpscount[videonames.index(matchedDataName.split("_")[-1].split(".")[0])])
    proc.matchGazeDataToFrame(0,int(length))