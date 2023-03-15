import csv
from os import listdir
import cv2
import numpy as np
import scipy.stats as st
import pandas as pd
from pathlib import Path
import main


class GazeBehaviourProcession:
    csv_file=""
    data = None
    kernel = None

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data=pd.read_csv(csv_file,header=None)
        self.kernel = self.gkern(55 * 2 + 1, main.kernelsigma)

    def matchGazeDataToFrame(self,videoframes,startFrame,endFrame):
        videoName=self.csv_file.split("\\")[-1].split("_")[-1][:-4]
        trackerTics=int(main.eyeTrackerSamplingRate / main.videofpscount[main.videonames.index(videoName)])
        gazeposXL,gazeposYL,gazeposXR,gazeposYR,timestamp=self.getXYPos()
        datapath= main.dataPath + "\\videos\\IttiKochImages"
        videoNameToSeach = "_"+videoName+"_"
        frames = [f for f in listdir(datapath) if videoNameToSeach in f]
        picturelist = []

        if endFrame > int(frames[-1].split("_")[-1].split(".")[0]):
            endFrame = int(frames[-1].split("_")[-1].split(".")[0])

        for r in range(startFrame,endFrame):
            if self.checkDataForValidity(self.csv_file,main.validityPercentile,r*trackerTics,r*trackerTics+trackerTics):
                print("Prozessiere Frame Nr.", r)
                xl=[i for i in gazeposXL[int(r*trackerTics):int((r+1)*trackerTics)] if i>0 and i<1]
                yl=[i for i in gazeposYL[int(r*trackerTics):int((r+1)*trackerTics)] if i>0 and i<1]
                xr=[i for i in gazeposXR[int(r*trackerTics):int((r+1)*trackerTics)] if i>0 and i<1]
                yr=[i for i in gazeposYR[int(r*trackerTics):int((r+1)*trackerTics)] if i>0 and i<1]


                if len(xl)>(trackerTics/(100/main.validityPercentile)) and len(xr)>(trackerTics/(100/main.validityPercentile)) and \
                        len(yl)>(trackerTics/(100/main.validityPercentile)) and len(yr)>(trackerTics/(100/main.validityPercentile)):
                    # print(xl, len(xl))
                    # print(yl, len(yl))
                    # print(xr, len(xr))
                    # print(yr, len(yr))
                    #
                    # print(np.mean(xl))
                    # print(np.mean(yl))
                    # print(np.mean(xr))
                    # print(np.mean(yr))

                    xpos = (np.mean(xl) + np.mean(xr)) / 2
                    ypos = (np.mean(yl) + np.mean(yr)) / 2

                    # print("Xpos: ", xpos)
                    # print("Ypos: ", ypos)

                    imageGrayValue, flag = self.getImageFractal(videoframes[r],xpos,ypos)
                    if flag:
                        picturelist.append(["Frame "+str(r),timestamp[r*trackerTics],imageGrayValue, 1])
                    else:
                        picturelist.append(["Frame "+str(r),timestamp[r*trackerTics],imageGrayValue, 0])


                else:
                    picturelist.append(["Frame "+str(r),timestamp[r*trackerTics],-1,-1])
            else:
                picturelist.append(["Frame "+str(r),timestamp[r*trackerTics],-1, -1])

        resultPath = main.dataPath+"Ergebnisse\\"
        Path(resultPath).mkdir(parents=True, exist_ok=True)
        with open(resultPath+frames[0][:16]+"\\"+self.csv_file.split("\\")[-1], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['Frame Nr','Timestamp','Graustufenwert','Blickpunkt verschoben'])
            writer.writerows(picturelist)

    def getImageFractal(self,image,xpos,ypos):
        # Ermittle die Höhe und Breite des Bildes
        height, width = image.shape[:2]
        flag = False
        x,y=round(int(width*xpos)),round(int(height*ypos))
        radius = 55
        if x>=width-55:
            x=width-56
            flag=1
        elif x<=55:
            x=56
            flag = 1
        if y>=height-55:
            y=height-56
            flag = 1
        elif y<=55:
            y=56
            flag = 1

        result=image[y-radius:y+radius+1,x-radius:x+radius+1]

        # mask = np.zeros(result.shape)
        # mask = mask.astype(np.uint8)
        # mask = cv2.circle(mask, (radius,radius), radius, (255,255,255), -1)
        # dst = cv2.bitwise_and(result, mask)


        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        returnvalue=np.sum(gray*self.kernel)/255

        # cv2.imshow('Result', gray)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()
        #
        # print(returnvalue, flag)
        return returnvalue, flag


    def getXYPos(self):
        df = self.data

        gazeposXL = df.iloc[:, 7]
        gazeposYL = df.iloc[:, 8]
        gazeposXR = df.iloc[:, 20]
        gazeposYR = df.iloc[:, 21]
        timestamp = df.iloc[:, 27]

        gazeposXL = gazeposXL.values.tolist()
        gazeposYL = gazeposYL.values.tolist()
        gazeposXR = gazeposXR.values.tolist()
        gazeposYR = gazeposYR.values.tolist()
        timestamp = timestamp.values.tolist()

        return gazeposXL, gazeposYL, gazeposXR, gazeposYR, timestamp

    def checkDataForValidity(self, data, percentile, optstart=None,optend=None):
        df=self.data
        if optstart and optend is None:
            df=df.iloc[:,1]
        else:
            df=df.iloc[optstart:optend,1]
        df=df.values.tolist()
        if len(df)/(100/percentile) > df.count(0.0): #0.0 gilt als NICHT VALID
            return True

        print("Data invalid -", df.count(0.0), "von", len(df) ,"Einträgen sind ungültig.")
        return False

    def gkern(self, kernlen, nsig=1.0):
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()
