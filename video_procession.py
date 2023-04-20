import cv2
from os import listdir
from os.path import isfile, join
from pathlib import Path

import main


def splitVideosToFrames():
    framePath= main.videoDataPath + "Frames\\"
    Path(framePath).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(main.videoDataPath) if isfile(join(main.videoDataPath, f))]
    for item in onlyfiles:
        frameNr = 0
        capture = cv2.VideoCapture(join(main.videoDataPath, item))
        print("Processing ", item)
        while (True):
            success, frame = capture.read()
            if success:
                formattedFrameNr="{:04d}".format(frameNr) # Die Frame Nr wird in das Format von 4 Ziffern gebracht - aus 1 wird 0001
                frameName = (f'{framePath}{item[0:-4]}_{formattedFrameNr}.jpg') # item[0:-4] nimmt den Namen des videos ohne die ".mp4" Endung
                print(frameName)
                cv2.imwrite(frameName, frame)
            else:
                break
            frameNr += 1
        capture.release()
    print("Frames saved.")