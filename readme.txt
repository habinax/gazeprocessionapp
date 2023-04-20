To setup the application:

1. Download python: https://www.python.org/downloads/ (install pip while downloading, otherwise it has to be added later on manuually)
2. Check if python and pip are installed by the following command: "py --version" und "py -m pip"
3. Execute the setup.bat file in the directory of the application
3.1. If problems with virtualenv occur, run the command "py -m pip install virtualenv" and try executing the setup.bat file again

If the dependencies have been downloaded properly:

4. Save the data (timestamps, gazedata, event, calibpoints (if existing)) in venv/data/gazedata
5. Save all videos, that have been shown to the subjects, in venv/data/videos
6. Set all flags in the config.properties file to "True" (Case-sensitive!)
7. Execute the "Applikation_starten" file to start the program



##############################################################

config.properties explanation: 

DataPreProcession = True/False    --- True (case sensivite!) if the data hasn't been pre processed, otherwise False
SplitVideo = True/False           --- True if the videos haven't been split up to frames, otherwise False
CreateIttiKochFrames = True/False --- True if saliency maps should be created, otherwise False

weight_intensity   = 0.30         --- parameter for the creation of the Itti-Koch saliency map
weight_color       = 0.30         --- parameter for the creation of the Itti-Koch saliency map
weight_orientation = 0.20         --- parameter for the creation of the Itti-Koch saliency map
weight_motion      = 0.20         --- parameter for the creation of the Itti-Koch saliency map
Cutoff = 30                       --- The value in percent, which is defining the cut off. The lower the cutoff value, the higher is the tolerance.
KernelSigma = 1.0                 --- The sigma value for the 2d gauss kernel

##############################################################
