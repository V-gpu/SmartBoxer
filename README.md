# SmartBoxer
**Automated Movement-Pattern Analytics and Longitudinal Performance Tracking of Multiple Boxers in Large-Scale Sparring Videos**

![GitHub repo size](https://img.shields.io/github/repo-size/V-gpu/SmartBoxer)
![GitHub stars](https://img.shields.io/github/stars/V-gpu/SmartBoxer)

![Alt text](SLOAN/Demonstration_Figure.jpg?raw=true "Movement Pattern Analysis of Two Boxers throughout a Bout using 4 Quantitative Metrics")

## Introduction
SmartBoxer is a powerful tool for analyzing movement patterns and tracking performance in sparring boxing videos. <br> 
This repository provides all the necessary resources to get started. <br>
Read the following instructions to work with the repository for inferencing and reproducibility testing.<br>
This code is implemented and tested on the Ubuntu 22.04 LTS system with Python 3.8.

### Note
To install the end-to-end SmartBoxer module repository using GitHub and run it on your local system, follow the given steps, download all the required files from the [Google Drive link](https://drive.google.com/drive/folders/1zMeZAZI32kszZup85OTsRsr5KrcppYjQ), and place them accordingly.

## System Requirements
- Linux or Ubuntu
- Python 3.6+

## Installation
### Pre-requisites
Install Anaconda, open a terminal, and create a new environment with Python 3.6 or later. <br>
  `conda create -n smartboxer python=3.6+`<br>
  `conda activate smartboxer`
   
1. Clone the repository to your local system:
`git clone https://github.com/V-gpu/SmartBoxer.git`

2. Download the following weights files:
- yolov5l.pt from `SLOAN/00boundarydetection/yolov5l.pt`
- yolov5n.pt from `SLOAN/00boundarydetection/yolov5n.pt`
- yolov7-seg.pt from `SLOAN/1detection/yolov7-seg.pt`

3. Install the required Python packages:<br>
`cd SmartBoxer/SLOAN/1detection/` <br>
`pip install -r requirements.txt\`

## Getting Started
### Automatic Bout Clip Segmentation
1. Download the input long-term raw video from `SmartBoxer/SLOAN/00boundarydetection/input/Bout-03-Mar-2023 10-35-34.avi`.

2. Run the segmentation code:<br>
`cd SmartBoxer/SLOAN`<br>
`bash smartboxerM1.sh`

3. Segmented bout clips will be saved in `SmartBoxer/SLOAN/0SegmentedVideos/segmentedbouts`.

### Continuous and Robust Tracking (HistoTrack) + Movement Pattern Analytics
1. Place the bout clip to be tested at `SmartBoxer/SLOAN/0SegmentedVideos/Toprocess/Bout-0_Boxer1_Boxer-2_2023-03-03111.avi`.

2. Run the tracking and analytics code:<br>
`cd SmartBoxer/SLOAN`<br>
`bash smartboxerM2.sh`

3. Follow the on-screen instructions to select the individuals to track. <br>
Visually look at the colored boxes around each individual from the ‘source’ subfolder of 4Metrics. <br>
D1 - black, D2- white, D3 - cyan. You can type ‘T’ and press enter accordingly. <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For instance, let us say that boxers are assigned black and white boxes, while the referee is assigned the cyan box. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Then, D1: type ‘T’ and press enter, D2: type ‘T’ and press enter, D3: press enter. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This allows the AI model to track only the boxers.

5. A new folder with movement pattern analytics will be created at `SmartBoxer/SLOAN/4Metrics/`

## Reproducibility and Analytics
To plot overall bout analytics and check reproducibility, run the following:<br>
`cd SmartBoxer/SLOAN`<br>
`python Plotting.py`

### Optional
- To rename videos, edit the player names in `renamefile.py` and uncomment the appropriate line in `smartboxerM1.sh`.
- Run `smartboxerM1.sh` to generate renamed segmented bouts.

## Additional Resources
- For the complete repository, including datasets and pre-trained models, visit this [Google Drive link](https://drive.google.com/drive/folders/1zMeZAZI32kszZup85OTsRsr5KrcppYjQ).
- Feel free to mail us at baghelvipul@iitgn.ac.in for any queries about executing the files.
