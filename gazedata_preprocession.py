from os import listdir
from os.path import isfile, join
import itertools
import pandas as pd
from pathlib import Path
import cv2
import main

timeStampIndex = 26  #Stelle der timeStamp im CSV, startend bei index 0

def pipeline():
    path = main.gazeDataPath
    combineColumns(path)
    matchEvent(path)



def matchEvent(path):
    eyeTrackerFrames = getVideoFrameNumbersWithSampleRate(main.videoDataPath)
    combinedDataPath = path + "combined_gazedata\\"

    combinedFiles = [f for f in listdir(combinedDataPath) if isfile(join(combinedDataPath, f))] # Aggregiert alle kombinierten Files zusammen in eine Liste

    files = [f for f in listdir(path) if isfile(join(path, f))]  # Aggregiert alle nicht kombinierten Files zusammen in eine Liste, um eventfiles zu extrahieren
    eventFiles = [f for f in files if f.endswith("event.csv")]

    pathToWrite = combinedDataPath + "video_matched_data\\"
    Path(pathToWrite).mkdir(parents=True, exist_ok=True)

    for var in range(len(eventFiles)):
        event = pd.read_csv(path + eventFiles[var], header = None)
        comb = pd.read_csv(combinedDataPath + combinedFiles[var], header = None)

        print("Suche in: " + combinedFiles[var])
        searchVar="natorient_"
        natorientEvent = []

        for name in main.videonames:
            fullSearchVar = searchVar+name
            print("Suche nach: " + fullSearchVar)
            for index in event.index:
                if event.iloc[index, 0].endswith(fullSearchVar):
                    print("Gefunden in Zeile: " + str(index))
                    natorientEvent.append([fullSearchVar, event.iloc[index, 1]])

        for x in range(len(natorientEvent)):
            videoNametoSearch=natorientEvent[x][0].split("_")[1]+".mp4"
            a,b=index_2d(eyeTrackerFrames,videoNametoSearch)
            framelen=eyeTrackerFrames[a][b+1]
            print(framelen)
            for row in range(len(comb)):
                if natorientEvent[x][1] < comb.iloc[row,timeStampIndex]:
                    startRow=row
                    endRow=row+framelen
                    print(startRow,endRow)
                    natorientEvent[x][1]=startRow
                    natorientEvent[x].append(endRow)
                    print(natorientEvent)
                    break

        writeEventDataToCSV(natorientEvent, comb, pathToWrite + combinedFiles[var].partition("_")[2])

def writeEventDataToCSV(eventList, gazeDataFile, dataPath):
    for event in eventList:
        if len(event) == 3:
            vidName, start, end = event[0],event[1],event[2]
            writePath = dataPath.split(".")[0] + "_" + vidName.split("_")[1] + ".csv"
            print(writePath)

            gazeDataFile.iloc[start:end,:].to_csv(writePath,header=None) #Extrahiert Daten von Start- bis Endpunkt und erstellt und schreibt sie in eine CSV Datei
        else:
            print("Fehler beim Prozessieren von", dataPath, "aufgetreten. Proband wird übersprungen.")


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, x.index(v)

def combineColumns(path):
    gazefiles = [f for f in listdir(path) if isfile(join(path, f))]     #Aggregiert alle Files zusammen in eine Liste
    groups = [list(g) for _, g in itertools.groupby(sorted(gazefiles), lambda x: x[0:6])]    #Gruppiert die Daten anhand der ersten 6 Buchstaben des Namens (siehe letzte Expression x[0:6])

    combined_data_path=path+"combined_gazedata\\" #Ort, an dem die Daten abgespeichert werden
    Path(combined_data_path).mkdir(parents=True, exist_ok=True) #Erstellt Ordner, falls nicht bereits vorhanden

    for g in groups:
        if len(g)==4: # 4 falls calibpoints Datei vorhanden ist
            filelist=[pd.read_csv(path+g[2], header = None),pd.read_csv(path+g[3],header = None)] # Liest Daten von gazedata und timestamps Datei ein
            if len(filelist[0].index) == len(filelist[1].index):
                excl_merged = pd.concat(filelist, axis=1)
                print(combined_data_path + "combined_" + g[0].split("_")[0] + "_" + g[0].split("_")[1])
                excl_merged.to_csv(combined_data_path + "combined_" + g[0].split("_")[0] + "_" + g[0].split("_")[1] + ".csv", index=False)
        elif len(g)==3: # 4 falls calibpoints Datei nicht vorhanden ist
            filelist = [pd.read_csv(path + g[1], header = None), pd.read_csv(path + g[2], header = None)] # Liest Daten von gazedata und timestamps Datei ein
            if len(filelist[1].index) == len(filelist[0].index):
                excl_merged = pd.concat(filelist, axis=1)
                print(combined_data_path + g[0].split("_")[0] + "_" + g[0].split("_")[1])
                excl_merged.to_csv(combined_data_path + "combined_" + g[0].split("_")[0] + "_" + g[0].split("_")[1] + ".csv", index=False)

def getVideoFrameNumbersWithSampleRate(path):
    videoLength=[]
    videonames=main.videonames
    for name in videonames:
        cap = cv2.VideoCapture(path + name + ".mp4")
        fps = round(cap.get(cv2.CAP_PROP_FPS))#Anzahl der FPS
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #Anzahl der Frames im Video
        seconds = length/fps
        videoLength.append([name + ".mp4", round(seconds*main.eyeTrackerSamplingRate)]) #2D-List mit 5x2-Tupel: [Videoname, Sekunden]
    return videoLength

