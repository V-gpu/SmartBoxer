##!/bin/bash


read -p "Enter a directory: " ${PWD}

SUBFOLD0=${PWD%%/}/0SegmentedVideos/Toprocess/
SUBFOLD1=${PWD%%/}/1detection/runs/predict-seg/*
SUBFOLD11=${PWD%%/}/1detection/runs/predict-seg/
SUBFOLD2=${PWD%%/}/1detection/segment/predict.py
SUBFOLD3=${PWD%%/}/1detection/yolov7-seg.pt
SUBFOLD5=${PWD%%/}/2Tracking/histotracker.py
SUBFOLD6=${PWD%%/}/3Trajectory/trajectory.py

#rm -rf $SUBFOLD1 

if [ -d "$SUBFOLD0" ]; then
    echo "Number of files is $(find "$SUBFOLD0" -type f | wc -l)"
	#Totalvideofiles = $(find "$SUBFOLD0" -type f | wc -l)
    echo "Number of directories is $(find "$SUBFOLD0" -type d | wc -l)"
else
    echo "[ERROR]  Please provide a directory."
    exit 1
fi

start=0
end=$(find "$SUBFOLD0" -type f | wc -l) 
#echo 'ENG' $end


OIFS="$IFS"
IFS=$'\n'
#for filename in `find "$SUBFOLD0" -type f -name "*.avi"`
for filename in "$SUBFOLD0"/*
do
	videoarr[$index]="$filename"
	index=$(($index+1))
done;	
IFS="$OIFS"


for ((i=start; i<=end-1; i++))
do
	echo 'NAMECHECK', "${videoarr[i]}"
	#python3 $SUBFOLD2 --weights $SUBFOLD3 --source "${videoarr[i]}" --class 0 --imgsz 640 --max-det=100 --iou-thres=0.7 --conf-thres=0.7
done


Totalfilesexp="$(ls ${PWD%%/}/1detection/runs/predict-seg/)"

for filenameexp in "$SUBFOLD11"/*
do
	SUBFOLD44="${filenameexp}"
	exparr[$index1]=$SUBFOLD44
	index1=$(($index1+1))
	echo 'check', $index1
done;


for ((i=start; i<=end-1; i++))
do
	echo '12', "${exparr[i]}"
	echo '123', "${videoarr[i]}"
	#python3 $SUBFOLD5 --fullpathpickle "${exparr[i]}" --filelist "${videoarr[i]}" 
	#sleep 5
	python3 $SUBFOLD6 --filelist "${videoarr[i]}"
done


