# SmartBoxer
**Automated Movement-Pattern Analytics and Longitudinal Performance Tracking of Multiple Boxers in Large-Scale Sparring Videos**

![GitHub repo size](https://img.shields.io/github/repo-size/V-gpu/SmartBoxer)
![GitHub stars](https://img.shields.io/github/stars/V-gpu/SmartBoxer)

## Introduction
SmartBoxer is a powerful tool for analyzing movement patterns and tracking performance in sparring boxing videos. This repository provides all the necessary resources to get started.

## System Requirements
- Ubuntu 22.04 LTS
- Anaconda with Python 3.6+

## Installation
1. Clone the repository to your local system:
`git clone https://github.com/V-gpu/SmartBoxer.git`

2. Download the following weights files:
- yolov5l.pt from `SLOAN/00boundarydetection/yolov5l.pt`
- yolov5n.pt from `SLOAN/00boundarydetection/yolov5n.pt`
- yolov7-seg.pt from `SLOAN/1detection/yolov7-seg.pt`

3. Install the required Python packages:
`cd SmartBoxer/SLOAN/1detection/` 
`pip install -r requirements.txt\`

## Getting Started
### Automatic Bout Clip Segmentation
1. Download the input long-term raw video from `SmartBoxer/SLOAN/00boundarydetection/input/Bout-03-Mar-2023 10-35-34.avi`.

2. Run the segmentation code:
`cd SmartBoxer/SLOAN`
`bash smartboxerM1.sh`

3. Segmented bout clips will be saved in `SmartBoxer/SLOAN/0SegmentedVideos/segmentedbouts`.

### Continuous and Robust Tracking (HistoTrack) + Movement Pattern Analytics
1. Place the bout clip to be tested at `SmartBoxer/SLOAN/0SegmentedVideos/Toprocess/Bout-0_Boxer1_Boxer-2_2023-03-03111.avi`.

2. Run the tracking and analytics code:
`cd SmartBoxer/SLOAN`
`bash smartboxerM2.sh`

3. Follow the on-screen instructions to select the individuals to track.

4. A new folder with movement pattern analytics will be created.

## Reproducibility and Analytics
To plot overall bout analytics and check reproducibility, run the following:
`cd SmartBoxer/SLOAN`
`python Plotting.py`

### Optional
- To rename videos, edit the player names in `renamefile.py` and uncomment the appropriate line in `smartboxerM1.sh`. Run `smartboxerM1.sh` to generate renamed segmented bouts.

## Additional Resources
- For the complete repository, including datasets and pre-trained models, visit [this Google Drive link](https://drive.google.com/drive/folders/1zMeZAZI32kszZup85OTsRsr5KrcppYjQ).
