import cv2
from os import listdir
from os.path import isfile, join

import main


def splitVideosToFrames():
    framePath= main.videoDataPath + "Frames\\"
    onlyfiles = [f for f in listdir(main.videoDataPath) if isfile(join(main.videoDataPath, f))]
    for item in onlyfiles:
        frameNr = 0
        capture = cv2.VideoCapture(join(main.videoDataPath, item))
        print("Prozessiere ", item)
        while (True):
            success, frame = capture.read()
            if success:
                formattedFrameNr="{:04d}".format(frameNr)
                frameName = (f'{framePath}{item[0:-4]}_{formattedFrameNr}.jpg')
                print(frameName)
                cv2.imwrite(frameName, frame)
            else:
                break
            frameNr += 1
        capture.release()
    print("Frames abgespeichert.")
