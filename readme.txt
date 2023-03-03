Zum Aufsetzen der Applikation:

1. Python herunterladen: https://www.python.org/downloads/ (am besten bei direkt bei der Installation pip mit installieren, sonst muss es nachträglich hinzugefügt werden)
2. In CMD überprüfen ob Python und pip installiert sind durch folgende Commands: "python --version" und "pip"
3. In dem Verzeichnis, in dem die Applikation cloned wurde, die setup.bat Datei ausführen


Falls die gesamten Dependencies ordnungsgemäß heruntergeladen wurden nun folgende Schritte vollziehen:

4. Die gesamten erhobenen Daten (inkl. calibpoints+event) in .env/data/gazedata abspeichern
5. Die Videos in .env/data/videos abspeichern


##############################################################

config.properties Erklärung: 

DataPreProcession = True/False    --- Wert True (case sensivite!), falls die Daten noch nicht vorverarbeitet worden sind, sonst False
SplitVideo = True/False           --- Wert True, falls die Videos noch nicht in einzelne Frames aufgespalten worden sind, sonst False
CreateIttiKochFrames = True/False --- Wert True, falls Itti-Koch saliency maps erstellt werden sollen, sonst False

weight_intensity   = 0.30         --- Parameter für die Erstellung der Itti-Koch saliency map
weight_color       = 0.30         --- Parameter für die Erstellung der Itti-Koch saliency map
weight_orientation = 0.20         --- Parameter für die Erstellung der Itti-Koch saliency map
weight_motion      = 0.20         --- Parameter für die Erstellung der Itti-Koch saliency map
Cutoff = 30                       --- Der Wert in Prozent, ab dem Daten für ungültig befunden werden - Abhängig von Anzahl der Gaze-Daten pro Frame: Sollten bei beispielsweise 10 Gazedata Einträgen, die zu einem Frame gehören, mehr als [Cutoff]% davon ungültig sein, so werden die Daten als nicht relevant gewertet. Je niedriger der Cutoff, desto höher die Toleranz.
KernelSigma = 1.0                 --- Der Sigma-Wert des 2D-Gauss-Kernels, der die Relevanz der Pixel innerhalb des Blickpunktes bestimmt.

##############################################################
