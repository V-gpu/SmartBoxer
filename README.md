# SmartBoxer
Automated Movement-Pattern Analytics and Longitudinal Performance Tracking of Multiple Boxers in Large-Scale Sparring Videos

Read the following instructions to work with SmartBoxer repository for inferencing and reproducibility testing.
This code is implemented and tested on the Ubuntu 22.04 LTS system.
The complete repository, along with the dataset and pre-trained models, are available at
https://drive.google.com/drive/folders/1zMeZAZI32kszZup85OTsRsr5KrcppYjQ

Note:
To install the end-to-end SmartBoxer module repository using GitHub and run it on your local system, follow the given steps, download all the required files from the Google Drive link, and place them accordingly.

Pre-requisites
Install anaconda, open a terminal, and create a new environment with Python3.6+
   conda create -n smartboxer python=3.6+
   conda activate smartboxer

Installation
1) Download the repository to your local system using git clone https://github.com/V-gpu/SmartBoxer.git
2) Download the yolov5 and yolov7 weights from SLOAN/00boundarydetection/yolov5l.pt, SLOAN/00boundarydetection/yolov5n.pt and SLOAN/1detection/yolov7-seg.pt, respectively.
3) pip install requirements.txt (The requirements file is available in the ‘1detection’ folder) 

Get Started
Automatic Bout Clip Segmentation:
1) Download the input long-term raw video from SLOAN/00boundarydetection/input/Bout-03-Mar-2023 10-35-34.avi
2) Run the following code in the terminal:
   cd SLOAN
   bash smartboxerM1.sh
3) Segmented output bout clips gets saved into SLOAN/0SegmentedVideos/segmentedbouts folder
   
Continuous and Robust Tracking (HistoTrack) + Movement Pattern Analytics
1) Place the bout clip to be tested at SLOAN/0SegmentedVideos/Toprocess/Bout-0_Boxer1_Boxer-2_2023-03-03111.avi
2) Run the following code in the terminal:
   cd SLOAN
   bash smartboxerM2.sh
3) You would be required to select the desired number of individuals to track. Visually look at the colored boxes around each individual from the ‘source’ subfolder of 4Metrics. 
D1 - black, D2- white, D3 - cyan. You can type ‘T’ and press enter accordingly. For instance, let us say that boxers are assigned black and white boxes, while the referee is assigned the cyan box. Then, D1: type ‘T’ and press enter, D2: type ‘T’ and press enter, D3: press enter. This allows the AI model to track only the boxers. 

A new folder with the filename mentioned in the Toprocess folder will be created and four different Movement-Pattern Analytics (Directional histogram, hotspot, Engagement/disengagement, Zone management)  will be created, and the metrics for every 20 sec are derived. A video plotting the  Longitudinal Performance Tracking  is also saved in the 4metrics folder.

For plotting the overall bout analytics and checking the reproducibility of the code, go through and run the Plotting.py file using the terminal.
   cd SLOAN
   python Plotting.py 

Optional: 
To rename the video with specific names, use two names in the player one and player 2 columns and input the row in the renamefile.py (uncomment - python3 $SUBFOLD1 in the smartboxerM1.sh file). You would obtain a folder named ‘renamedsegmentedbouts’ containing the renamed segmented bouts.

