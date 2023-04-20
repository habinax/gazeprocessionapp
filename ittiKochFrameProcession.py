import time
from os import listdir
from os.path import isfile, join

import cv2

import main
import pySaliencyMap
import numpy as np
import pySaliencyMapDefs
from pathlib import Path

def ittiKochFrameProcession():
    framepath = join(main.videoDataPath, "Frames\\")
    ittikochpath = join(main.videoDataPath, "IttiKochImages\\")
    files = [f for f in listdir(framepath) if isfile(join(framepath, f))]  # Aggregiert alle nicht kombinierten Files zusammen in eine Liste, um eventfiles zu extrahieren
    Path(ittikochpath).mkdir(parents=True, exist_ok=True)
    for frame in files:
        print("Processing frame", frame)
        img = cv2.imread(join(framepath,frame))
        # initialize
        imgsize = img.shape
        img_width = imgsize[1]
        img_height = imgsize[0]
        sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

        # computation
        saliency_map = sm.SMGetSM(img)
        #binarized_map = sm.SMGetBinarizedSM(img)
        #salient_region = sm.SMGetSalientRegion(img)

        savestr = f'i{pySaliencyMapDefs.weight_intensity}c{pySaliencyMapDefs.weight_color}o{pySaliencyMapDefs.weight_orientation}m{pySaliencyMapDefs.weight_motion}'

        salmapstr = ittikochpath+savestr+"saliency_map_"+frame
        # binmapstr = "C:\\Users\\inap\\PycharmProjects\\gazeprocession\\venv\\data\\videos\\IttiKochImages\\"+savestr+"binarized_map_"
        # salregstr = "C:\\Users\\inap\\PycharmProjects\\gazeprocession\\venv\\data\\videos\\IttiKochImages\\"+savestr+"salient_region_"

        norm = np.zeros((saliency_map.shape[0],saliency_map.shape[1]))

        final = cv2.normalize(saliency_map, norm, 0, 255, cv2.NORM_MINMAX)

        cv2.imwrite(salmapstr, final)
        # cv2.imwrite(f'{binmapstr}.jpg', binarized_map)
        # cv2.imwrite(f'{salregstr}.jpg', cv2.cvtColor(salient_region, cv2.COLOR_BGR2GRAY))
