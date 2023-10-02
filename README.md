# SmartBoxer
Automated Movement-Pattern Analytics and Longitudinal Performance Tracking of Multiple Boxers in Large-Scale Sparring Videos

Read the following instructions to work with SmartBoxer repository for inferencing and reproducibility testing.
This code is implemented and tested on the Ubuntu 22.04 LTS system.
The complete repository, along with the dataset and pre-trained models, are available at
https://drive.google.com/drive/folders/1zMeZAZI32kszZup85OTsRsr5KrcppYjQ

To install the end-to-end SmartBoxer module repository using GitHub and run it on your local system, follow the given steps, download all the required files from the Google Drive link, and place them accordingly.

Installation
1) Download the repository to your local system using git clone https://github.com/V-gpu/SmartBoxer.git
2) Download the yolov5 and yolov7 weights from SLOAN/00boundarydetection/yolov5l.pt, SLOAN/00boundarydetection/yolov5n.pt and SLOAN/1detection/yolov7-seg.pt, respectively. 

Get Started
Automatic Bout Clip Segmentation:
1) Download the input long-term raw video from SLOAN/00boundarydetection/input/Bout-03-Mar-2023 10-35-34.avi
2) Run the following code in the terminal:
   cd SLOAN
   bash smartboxerM1.sh

Continuous and Robust Tracking (HistoTrack) + Movement Pattern Analytics
1) Place the bout clip to be tested at SLOAN/0SegmentedVideos/Toprocess/Bout-0_Boxer1_Boxer-2_2023-03-03111.avi
2) Run the following code in the terminal:
   cd SLOAN
   bash smartboxerM2.sh

For plotting the overall bout analytics and checking the reproducibility of the code, go through and run the Plotting.py file using the terminal.
   cd SLOAN
   python Plotting.py 
   
