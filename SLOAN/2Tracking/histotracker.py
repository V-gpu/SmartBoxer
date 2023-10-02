import numpy as np

#contents = np.load('runs/detect/detectionthreshold80/object_tracking/1.npy', allow_pickle=True).reshape(1)
#print('contents', contents[0][2])
# contents = np.load('runs/detect/50/object_tracking/1.npy', allow_pickle=True).reshape(1)
# #print('contents', contents[0])
# A = contents[0].keys()
# AB = contents[0][1]
# print(AB)

import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from numpy import pi
import cv2
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
import pathlib



outputsimilarity = []
similarityinformation = []
refID = []
xaxisframe = []
count=0
framesnumber=0
P=[]
PP=[]
PPP=[]
R=[]
G=[]
B=[]
AssumingR_1 = []
AssumingG_1 = []
AssumingB_1 = []
AssumingR_2 = []
AssumingG_2 = []
AssumingB_2 = []
AssumingR_3 = []
AssumingG_3 = []
AssumingB_3 = []
intermediateOP = []
anchorboxes1 = []

ConScoID1 = []
ConScoID2 = []
ConScoID3 = []
TotalCheck=[]
ReSim = []
AssumedID1_1 = []
AssumedID2_1 = []
AssumedID3_1 = []
AssumedID4_1 = []
AssumedID5_1 = []
AssumedID6_1 = []
AssumedID7_1 = []
AssumedID8_1 = []
AssumedID9_1 = []
AssumedID10_1 = []
AssumedID1_2 = []
AssumedID2_2 = []
AssumedID3_2 = []
AssumedID4_2 = []
AssumedID5_2 = []
AssumedID6_2 = []
AssumedID7_2 = []
AssumedID8_2 = []
AssumedID9_2 = []
AssumedID10_2 = []

AssumedID1_3 = []
AssumedID2_3 = []
AssumedID3_3 = []
AssumedID4_3 = []
AssumedID5_3 = []
AssumedID6_3 = []
AssumedID7_3 = []
AssumedID8_3 = []
AssumedID9_3 = []
AssumedID10_3 = []

ratiofD1C1_R = []
ratiofD1C2_R = []
ratiofD1C3_R = []
ratiofD1C1_G = []
ratiofD1C2_G = []
ratiofD1C3_G = []
ratiofD1C1_B = []
ratiofD1C2_B = []
ratiofD1C3_B = []

insidebacktrack = 0
totatAssumedID1 = []
totatAssumedID2 = []
totatAssumedID3 = []
TotalhistID1 = []
TotalhistID2 = []
TotalhistID3 = []
actualframenumber = []
frameswap = 0
Countcheck = 0
bigboxcheck = []
global video1
# def euclidian_distance(y1, y2):
#     return np.sqrt(np.sum((y1-y2)**2)) # based on pythagorean
from math import sqrt

def euclidian_distance(x, y):
    return sqrt(sum((px - py)**2  for px, py in zip(x,y)))

# import required libraries
import numpy as np
from numpy.linalg import norm

TotalCheck = []
Totalassumed  = []

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

save_path_new = str(pathlib.Path(__file__).parent.resolve())

CWD = str(pathlib.Path(__file__).parent.parent)
print('1234CWD',CWD)
#CWD = os.getcwd()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fullpathpickle")
parser.add_argument("--filelist")
args = parser.parse_args()
#config = vars(args)
#print(f'{args.fullpathpickle}')

fullpathpickle = args.fullpathpickle
print('-------------------fullpathpickle-----------------------',fullpathpickle)
#fullpathpickle = '1detection/runs/predict-seg/exp'
fullpath = '0SegmentedVideos'
fullpathprocess = 'Toprocess'

# contents = load_object(CWD+'/'+fullpathpickle+'/'+'BB.pkl')
# maskingid = load_object(CWD+'/'+fullpathpickle+'/'+'mask.pkl')
contents = load_object(fullpathpickle+'/'+'BB.pkl')
maskingid = load_object(fullpathpickle+'/'+'mask.pkl')

control = 10
Count = 0

videofilename = args.filelist
print('-------------------filelist-----------------------',videofilename)

#Load the video
#filelist = glob.glob(CWD+'/'+fullpath+'/'+ fullpathprocess + '/' + "*.avi")  # Get all pdf files in the current folder

# for file in filelist:
#     videofilename = file

nameofvideo = os.path.split(videofilename)[-1]
nameoffolder,ext = os.path.splitext(nameofvideo)

metricfolder = '4Metrics'

nameofvideopath = CWD+'/'+ metricfolder+'/' + nameoffolder
isExist = os.path.exists(nameofvideopath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(nameofvideopath)
   print("The video directory is created!")

Sourcepath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "Sourcefolder"
isExist = os.path.exists(Sourcepath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(Sourcepath)
   print("The Sourcepath directory is created!")

#video1=cv2.VideoCapture('/home/monsley/Downloads/yolov7-segmentation-main/videos/20sec/221.mp4')
video1=cv2.VideoCapture(videofilename)
total_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
#print('total_frames',total_frames)


width  = video1.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = video1.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print(width,height)

def compute_histogram(image):
    """Returns histogram for image region defined by mask for each channel
    Params:
        image (numpy array) - original image. Shape (H, W, C).
        mask (numpy array) - boolean mask. Shape (H, W).
    Output:
        list of tuples, each tuple (each channel) contains 2 arrays: first - computed histogram, the second - bins.

    """
    # Apply binary mask to your array, you will get array with shape (N, C)
    #region = image[mask]
	
    region = image

    red = np.histogram(region[..., 0].ravel(), bins=256, range=[0, 256])
    green = np.histogram(region[..., 1].ravel(), bins=256, range=[0, 256])
    blue = np.histogram(region[..., 2].ravel(), bins=256, range=[0, 256])

    return [red, green, blue]

def plot_histogram(histograms):
    """Plots histogram computed for each channel.
    Params:
        histogram (list of tuples) - [(red_ch_hist, bins), (green_ch_hist, bins), (green_ch_hist, bins)]
    """

    colors = ['r', 'g', 'b']
    for hist, ch in zip(histograms, colors):
        plt.bar(hist[1][:256], hist[0], color=ch)
        plt.show()


def plot_histogram_folder(img,checkframenumbers,detectionsinframes):
    color = ('r', 'g', 'b')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[1,256])
        plt.plot(histr,color = col)
        plt.xlim([1,256])
    #plt.show()
    plt.savefig(f'/home/monsley/Downloads/yolov7-segmentation-main/histogramsss/' + str(checkframenumbers) + '_' + str(detectionsinframes) + '.png')
    plt.close()



from collections import Counter
def leastFrequencyElement(inputList, listLength):
   # getting the frequency of all elements of a list
   hashTable = Counter(inputList)
   # Setting the minimum frequency(minimumCount) as length of list + 1
   minimumCount = listLength + 1
   # Variable to store the resultant least frequent element
   resultElement = -1
   # iterating the hash table
   for k in hashTable:
         # Check if the minimum count is greater or equal to the frequency of the key
            if (minimumCount >= hashTable[k]):
               # If it is true then this key will be the current least frequent element
                  resultElement = k
            # Set the minimum count as the current key frequency value
            minimumCount = hashTable[k]
# returning the least frequent element
   return resultElement

def croppedreg(frame, mask):
    im = frame
    imre=cv2.resize(im, [640,640])
    mask1 = mask.astype(np.uint8)
    masked = cv2.bitwise_and(imre, imre, mask=mask1) 
    points = masked
    zero_rows = (points == 0).all(1)
    first_invalid = np.where(zero_rows)[0][0]
    points[:first_invalid]
    positions = np.nonzero(points)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    croppedregion = masked[top:bottom, left:right]
    #croppedregion = points
    return croppedregion

def most_frequent(List):
    return max(set(List), key = List.count)

def sum_lists(*args):
    return list(map(sum, zip(*args)))

def aIOU(a, b):  # returns None if rectangles don't intersect
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[4], b[4]) - max(a[2], b[2])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
	    return 0

templateanchorboxes1 = []
checkanchorboxes1 = []
controllingfactor = 1
forbacktracking = []
boundingboxofassumedID3 = []
TotalassumeBBox = []
# the flow is like, i will look at the number of detections in a frame, and the detected anchor boxes > 0.5 CS, seperately
#if the number of detections in a frame is  ==3 i will take that as a reference.  
F2sub=0
DetectionThreshold = 0.5
#A111 = [1,2,3,4,5,6,7,8,9,10,11,12]
#Determining how many number of frames to be crosschecked
count = 0
ret, frame = video1.read()
#while ret == True:
ID1 = {}
ID2 = {}
ID3 = {}
NEWF1 = 0
NEWF1hist = 0
for framesnumber in range(1,(total_frames)-1):
	#framesnumber+=1
	#video1.set(cv2.CAP_PROP_FRAME_COUNT, 100)
	ret, frame = video1.read()
	sameframe =frame.copy()
	#print(framesnumber)

	height, width, channels = frame.shape
	#print('height, width, channels',height, width, channels)
	blank = np.zeros(frame.shape[:2], dtype='uint8')
	sameblank = blank.copy()
	# To check whether the 1st frame has only two detections
	#this will complete for the 1st frame
	for detectionsinframes in range(len(contents[0][framesnumber])):
		A11 = contents[0][framesnumber][detectionsinframes]
		#print((A11))
		if len(A11)!=0:# and detectionsinframes < 3:
			count+=1
		Tcheckmodule = [framesnumber, detectionsinframes, count]
		#print('Tcheckmodule',(Tcheckmodule),count)
		forbacktracking.append(Tcheckmodule)
		#print('forbacktracking',forbacktracking)
	count = 0
	#print('==============================================len(forbacktracking)===============================',(forbacktracking))
	if len(TotalassumeBBox)==0 and (NEWF1) ==0 and Tcheckmodule[2]==3:
		if len(forbacktracking) <3:
			takingtheinitialframe = forbacktracking[0][0]
			Nextframe = framesnumber# checkframe <
			F2sub = Nextframe - takingtheinitialframe+1
            #print('check after skip',framesnumber-F2sub)        
			#print('-------------------------------F2sub---------------------------------',framesnumber, F2sub)
			for detectionsinframes in range(len(contents[0][framesnumber-F2sub])):
				A11 = contents[0][framesnumber-F2sub][detectionsinframes]
				if len(A11)!=0:
					count+=1
				Tcheckmodule = [framesnumber-F2sub, detectionsinframes, count]
                #print('Tassumemoduleskipped',framesnumber-F2sub, Tcheckmodule)
			count = 0
			forbacktracking = []
            #frameswap=1
		elif len(forbacktracking) == 3:
			F2sub = 0
            #print('checkingwhether it takes the correct framenumber',framesnumber-F2sub)
			for detectionsinframes in range(len(contents[0][framesnumber-F2sub])):
				A11 = contents[0][framesnumber-F2sub][detectionsinframes]
				if len(A11)!=0:
					count+=1
				Tcheckmodule = [framesnumber-F2sub, detectionsinframes, count]
				#print('Tassumemodulecorrectframe',framesnumber-F2sub, Tcheckmodule)			
			count = 0
			forbacktracking = []
        #forbacktracking = []
        #print('-------------------------------F2sub---------------------------------',framesnumber-F2sub,framesnumber)    
		if Tcheckmodule[0]==framesnumber-F2sub and Tcheckmodule[2]==3:
			#print('framesnumbiii',framesnumber-F2sub)
			for detectionsinframes in range(len(contents[0][framesnumber-F2sub])):
				A11 = contents[0][framesnumber-F2sub][detectionsinframes]
				x = round(int(A11[0]))
				y = round(int(A11[1]))
				w = round(int(A11[2]))
				h = round(int(A11[3]))					
				im = frame
				mask0 = (maskingid[0][framesnumber][detectionsinframes])				
				croppedregion = croppedreg(im, mask0)
				#cv2.imwrite("/home/monsley/Downloads/yolov7-object-tracking-main/assumedboximage/image"+str(framesnumber-F2sub)+ '_' +str(detectionsinframes)+".jpg", croppedregion)
				normchannel0 = []
				if detectionsinframes == 0:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				#normchannel0 = max(normchannel0)
				elif detectionsinframes == 1:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				#normchannel0 = max(normchannel0)						
				elif detectionsinframes == 2:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				normchannel01 = max(normchannel0)
				color = ['r','g','b']
				for i,col in enumerate(color):
					histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
					#print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-histr--XXXXXXXXXXXXXXXXXXXXXXXXXXX',len(histr), np.amax(histr),np.amin(histr) )
					if detectionsinframes == 0 and i == 0 and col == 'r':#ID1
						AssumingR_1 = histr
						AssumingR_1 = histr/normchannel01
						plt.plot(AssumingR_1,color = col)
						#AssumingR_1 = cv2.normalize(AssumingR_1, AssumingR_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					if detectionsinframes == 0 and i == 1 and col == 'g':
						AssumingG_1 = histr
						AssumingG_1 = histr/normchannel01
						plt.plot(AssumingG_1,color = col)						
						#AssumingG_1 = cv2.normalize(AssumingG_1, AssumingG_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					if detectionsinframes == 0 and i == 2 and col == 'b':
						AssumingB_1 = histr
						AssumingB_1 = histr/normchannel01
						plt.plot(AssumingB_1,color = col)						
						#AssumingB_1 = cv2.normalize(AssumingB_1, AssumingB_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)				
                    #print('R,G,B ID1', len(AssumingR_1),len(AssumingG_1),len(AssumingB_1))# individual 1
					if detectionsinframes == 1 and i == 0 and col == 'r':#ID2
						AssumingR_2 = histr
						AssumingR_2 = histr/normchannel01
						plt.plot(AssumingR_2,color = col)						
						#AssumingR_2 = cv2.normalize(AssumingR_2, AssumingR_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					if detectionsinframes == 1 and i == 1 and col == 'g':
						AssumingG_2 = histr
						AssumingG_2 = histr/normchannel01
						plt.plot(AssumingG_2,color = col)						
						#AssumingG_2 = cv2.normalize(AssumingG_2, AssumingG_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					if detectionsinframes == 1 and i == 2 and col == 'b':
						AssumingB_2 = histr
						AssumingB_2 = histr/normchannel01
						plt.plot(AssumingB_2,color = col)						
						#AssumingB_2 = cv2.normalize(AssumingB_2, AssumingB_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)				
					#print('R,G,B, ID2', len(AssumingR_2),len(AssumingG_2),len(AssumingB_2))# individual 2
					if detectionsinframes == 2 and i == 0 and col == 'r':#ID3
						AssumingR_3 = histr
						AssumingR_3 = histr/normchannel01
						plt.plot(AssumingR_3,color = col)
						#plt.savefig(f'histograms/' + str(framesnumber) + '_' + str(detectionsinframes) + '.png')						
						#AssumingR_3 = cv2.normalize(AssumingR_3, AssumingR_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)	
					if detectionsinframes == 2 and i == 1 and col == 'g':
						AssumingG_3 = histr
						AssumingG_3 = histr/normchannel01
						plt.plot(AssumingG_3,color = col)						
						#AssumingG_3 = cv2.normalize(AssumingG_3, AssumingG_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					if detectionsinframes == 2 and i == 2 and col == 'b':
						AssumingB_3 = histr
						AssumingB_3 = histr/normchannel01
						plt.plot(AssumingB_3,color = col)						
						#AssumingB_3 = cv2.normalize(AssumingB_3, AssumingB_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					#plt.savefig(f'histograms/' + str(framesnumber) + '_' + str(detectionsinframes) + '.png')
				templateanchorboxes1.append([framesnumber-F2sub,x, y, w, h])
				#print('templateanchorboxes1',templateanchorboxes1)
			Template1 = [AssumingR_1,AssumingG_1,AssumingB_1]
			Template2 = [AssumingR_2,AssumingG_2,AssumingB_2]
			Template3 = [AssumingR_3,AssumingG_3,AssumingB_3]
			boundingboxofassumedID1 = templateanchorboxes1[0]
			boundingboxofassumedID2 = templateanchorboxes1[1]
			boundingboxofassumedID3 = templateanchorboxes1[2]
			Totalassumed = [[AssumingR_1],[AssumingG_1], [AssumingB_1],[AssumingR_2],[AssumingG_2],[AssumingB_2], [AssumingR_3],[AssumingG_3],[AssumingB_3]]
			TotalassumeBBox = [[boundingboxofassumedID1], [boundingboxofassumedID2], [boundingboxofassumedID3]]
			plt.close()	
			print('F1',Tcheckmodule[0],boundingboxofassumedID1, boundingboxofassumedID2,boundingboxofassumedID3)
			insidebacktrack=0
	elif NEWF1 !=0:
	    print('---------------------------------------------------------------------------------------------')
	    TotalassumeBBox = NEWF1
	    Totalassumed = NEWF1hist
	    #print('TotalassumeBBox',TotalassumeBBox)

	if len(TotalassumeBBox)!=0:
		insidebacktrack+=1
		#print('insideF2framesnumber',framesnumber,insidebacktrack)
		checkframenumber = framesnumber + controllingfactor
		#print('insideF2checkframenumber',checkframenumber)
		for detectionsinframes in range(len(contents[0][checkframenumber])):
			A11 = contents[0][checkframenumber][detectionsinframes]
			if len(A11)!=0:	
				Countcheck+=1
			checkmodule = [checkframenumber, detectionsinframes, Countcheck]
			#print('insideF2checkframenumbercheckmodule',checkmodule)
		Countcheck = 0
        #print('framesnumber',framesnumber)
		if checkmodule[0]==checkframenumber and checkmodule[2]==3:
			for detectionsinframes in range(len(contents[0][checkframenumber])):
				A11 = contents[0][checkframenumber][detectionsinframes]
				x = round(int(A11[0]))
				y = round(int(A11[1]))
				w = round(int(A11[2]))
				h = round(int(A11[3]))	
				im = frame
				mask0 = (maskingid[0][checkframenumber][detectionsinframes])				
				croppedregion = croppedreg(im, mask0)	
				#cv2.imwrite("/home/monsley/Downloads/yolov7-segmentation-main/image/"+str(checkframenumber)+ '_' +str(detectionsinframes)+".jpg", croppedregion)# writing the croppedregion)
				#plot_histogram_folder(croppedregion,checkframenumber,detectionsinframes)
				normchannel0 = []
				if detectionsinframes == 0:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				#normchannel0 = max(normchannel0)
				elif detectionsinframes == 1:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				#normchannel0 = max(normchannel0)						
				elif detectionsinframes == 2:
					color = ['r','g','b']
					for i,col in enumerate(color):
						histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
						normchannel0.append(max(histr))
						#print('normchannel0',normchannel0, detectionsinframes)
				normchannel01 = max(normchannel0)
				color = ['r','g','b']
				for i,col in enumerate(color):
					histr = cv2.calcHist([croppedregion],[i],None,[255],[1,255])
					#print('normchannel01',normchannel01, detectionsinframes)
					#maY, maX, _ = plt.hist(croppedregion)
					if detectionsinframes == 0 and i == 0 and col == 'r':#ID?
						CheckingR_1 = histr
						CheckingR_1 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingR_1,color = col)					
						#CheckingR_1 = cv2.normalize(CheckingR_1, CheckingR_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)# per channel normalization
					elif detectionsinframes == 0 and i == 1 and col == 'g':
						CheckingG_1 = histr
						CheckingG_1 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingG_1,color = col)																
						#CheckingG_1 = cv2.normalize(CheckingG_1, CheckingG_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					elif detectionsinframes == 0 and i == 2 and col == 'b':
						CheckingB_1 = histr
						CheckingB_1 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingB_1,color = col)																	
						#CheckingB_1 = cv2.normalize(CheckingB_1, CheckingB_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)				
					#print('R,G,B', len(CheckingR_1),len(CheckingG_1),len(CheckingB_1))# individual 1
					elif detectionsinframes == 1 and i == 0 and col == 'r':#ID?
						CheckingR_2 = histr
						CheckingR_2 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingR_2,color = col)					 												
						#CheckingR_2 = cv2.normalize(CheckingR_2, CheckingR_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					elif detectionsinframes == 1 and i == 1 and col == 'g':
						CheckingG_2 = histr
						CheckingG_2 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingG_2,color = col)																	
						#CheckingG_2 = cv2.normalize(CheckingG_2, CheckingG_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					elif detectionsinframes == 1 and i == 2 and col == 'b':
						CheckingB_2 = histr
						CheckingB_2 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingB_2,color = col)																	
						#CheckingB_2 = cv2.normalize(CheckingB_2, CheckingB_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)					
					#print('R,G,B', len(CheckingR_2),len(CheckingG_2),len(CheckingB_2))# individual 2
					elif detectionsinframes == 2 and i == 0 and col == 'r':#ID?
						CheckingR_3 = histr
						CheckingR_3 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingR_3,color = col)																
						#CheckingR_3 = cv2.normalize(CheckingR_3, CheckingR_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					elif detectionsinframes == 2 and i == 1 and col == 'g':
						CheckingG_3 = histr
						CheckingG_3 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingG_3,color = col)																		
						#CheckingG_3 = cv2.normalize(CheckingG_3, CheckingG_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					elif detectionsinframes == 2 and i == 2 and col == 'b':
						CheckingB_3 = histr
						CheckingB_3 = histr/normchannel01
						#print('detectionsinframes, i, col, normchannel01',detectionsinframes, i, col, normchannel01)
						plt.plot(CheckingB_3,color = col)																	
						#CheckingB_3 = cv2.normalize(CheckingB_3, CheckingB_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
					#plt.savefig(f'histograms/' + str(checkframenumber) + '_' + str(detectionsinframes) + '.png')
				checkanchorboxes1.append([checkframenumber,x, y, w, h])
                #print('checkanchorboxes1',checkanchorboxes1)
			CheckID1 = [CheckingR_1,CheckingG_1,CheckingB_1]
			CheckID2 = [CheckingR_2,CheckingG_2,CheckingB_2]
			CheckID3 = [CheckingR_3,CheckingG_3,CheckingB_3]
			boundingboxofcheckID1 = checkanchorboxes1[0]
			boundingboxofcheckID2 = checkanchorboxes1[1]
			boundingboxofcheckID3 = checkanchorboxes1[2]			
			TotalCheck = [[CheckingR_1],[CheckingG_1],[CheckingB_1], [CheckingR_2],[CheckingG_2],[CheckingB_2], [CheckingR_3],[CheckingG_3],[CheckingB_3]]
			#TotalCheck = [AB[0], AB[1], AB[2]]
			TotalCheckBBox = [[boundingboxofcheckID1],[boundingboxofcheckID2],[boundingboxofcheckID3]]
			checkanchorboxes1 = []
			templateanchorboxes1 = []#added030723
			print('F2',checkframenumber,TotalCheckBBox)
			plt.close()
            #print('-----------------assume and check',TotalassumeBBox,TotalCheckBBox)
			actualframenumber.append(checkframenumber)
			A0 = TotalCheckBBox[0]
			A1 = TotalCheckBBox[1]
			A2 = TotalCheckBBox[2]
			print(A0[0],A1[0],A2[0])
			areaofoverlap01 = aIOU(A0[0],A1[0])#01
			areaofoverlap02 = aIOU(A0[0],A2[0])#02
			areaofoverlap12 = aIOU(A1[0],A2[0])#12
			print(checkframenumber,areaofoverlap01,areaofoverlap02,areaofoverlap12)
			if (areaofoverlap01 + areaofoverlap02 + areaofoverlap12) > 600:
				TotalCheck = []
				TotalCheckBBox = []
			else:
				TotalCheck = TotalCheck
				TotalCheckBBox = TotalCheckBBox	
				for bigbox in TotalCheckBBox:
					Fxywh=bigbox[0][0]
					xx = bigbox[0][1]
					yy = bigbox[0][2]
					ww = bigbox[0][3]
					hh = bigbox[0][4]
					hofbbox = hh-yy
					wofbbox = ww-xx
					bigboxcheck.append([hofbbox,wofbbox])
				print('bigboxcheck',bigboxcheck)
				detlarg = sorted([bigboxcheck[0][0],bigboxcheck[0][1],bigboxcheck[1][0],bigboxcheck[1][1],bigboxcheck[2][0],bigboxcheck[2][1]])
				print('detlarg',detlarg)
				if detlarg[0] <= 50:
					TotalCheck = []
					TotalCheckBBox = []
				else:
					TotalCheck = TotalCheck
					TotalCheckBBox = TotalCheckBBox
			bigboxcheck = []		
			backupTotalCheckBBox = TotalCheckBBox.copy()
			backupTotalCheck = TotalCheck.copy()
		if len(TotalCheck)==9 and len(Totalassumed) == 9:
			M = 0
			N = 0
			for k in range(3):
				if k==0:
					smart = 0
					M = 0
					N = 0
				elif k==1:
					smart = 1
					M = 1
					N = 1
				elif k==2:
					smart = 2
					M = 2
					N = 2	
				for l in range(3):
                    #GT = TotalAssumed[M]			
					for m in range(3):
						#print('M,N',k, M,N)
						PR = TotalCheck[N]
						GT = Totalassumed[M]
						#print(len(PR),len(GT))
						A = np.squeeze(np.array(GT))
						B = np.squeeze(np.array(PR))
						cosinesimilarity = np.dot(A,B)/(norm(A)*norm(B))
						#cosinesimilarity= wasserstein_distance(A,B)
						#_,cosinesimilarity = ks_2samp(A, B)
						ReSim.append(cosinesimilarity)
						N+=3
					N=smart
					M+=3
					outputsimilarity.append([checkframenumber, cosinesimilarity])#across temporal frames.
					#print('outputsimilarity',outputsimilarity)
			print('ReSim',(ReSim))

			R_D1C1 = ReSim[0]
			R_D1C2 = ReSim[1]
			R_D1C3 = ReSim[2]
			R_D2C1 = ReSim[3]
			R_D2C2 = ReSim[4]
			R_D2C3 = ReSim[5]
			R_D3C1 = ReSim[6]
			R_D3C2 = ReSim[7]
			R_D3C3 = ReSim[8]
			
			G_D1C1 = ReSim[9]
			G_D1C2 = ReSim[10]
			G_D1C3 = ReSim[11]			
			G_D2C1 = ReSim[12]
			G_D2C2 = ReSim[13]
			G_D2C3 = ReSim[14]
			G_D3C1 = ReSim[15]
			G_D3C2 = ReSim[16]
			G_D3C3 = ReSim[17]
			
			B_D1C1 = ReSim[18]
			B_D1C2 = ReSim[19]
			B_D1C3 = ReSim[20]
			B_D2C1 = ReSim[21]
			B_D2C2 = ReSim[22]
			B_D2C3 = ReSim[23]
			B_D3C1 = ReSim[24]
			B_D3C2 = ReSim[25]
			B_D3C3 = ReSim[26]
						
			D1C1 = [R_D1C1,G_D1C1, B_D1C1]
			D1C2 = [R_D1C2, G_D1C2, B_D1C2]
			D1C3 = [R_D1C3, G_D1C3, B_D1C3]                    
            
			D2C1 = [R_D2C1,G_D2C1, B_D2C1]
			D2C2 = [R_D2C2, G_D2C2, B_D2C2]
			D2C3 = [R_D2C3, G_D2C3, B_D2C3]           
                        
			D3C1 = [R_D3C1,G_D3C1, B_D3C1]
			D3C2 = [R_D3C2, G_D3C2, B_D3C2]
			D3C3 = [R_D3C3, G_D3C3, B_D3C3]
			
			M1 = norm([ReSim[0], ReSim[9], ReSim[18]])
			M2 = norm([ReSim[1], ReSim[10], ReSim[19]])
			M3 = norm([ReSim[2], ReSim[11], ReSim[20]])
			M4 = norm([ReSim[3], ReSim[12], ReSim[21]])
			M5 = norm([ReSim[4], ReSim[13], ReSim[22]])
			M6 = norm([ReSim[5], ReSim[14], ReSim[23]])
			M7 = norm([ReSim[6], ReSim[15],  ReSim[24]])
			M8 = norm([ReSim[7], ReSim[16], ReSim[25]])
			M9 = norm([ReSim[8], ReSim[17], ReSim[26]])
			print('[ReSim[0],ReSim[9], ReSim[18]]',ReSim[0],ReSim[9], ReSim[18],ReSim[0]+ReSim[9]+ReSim[18], norm([ReSim[0],ReSim[9], ReSim[18]]) )
			print('[ReSim[1],ReSim[10], ReSim[19]]',ReSim[1],ReSim[10], ReSim[19],ReSim[1]+ReSim[10]+ReSim[19], norm([ReSim[1],ReSim[10], ReSim[19]]) )
			print('[ReSim[2],ReSim[11], ReSim[20]]',ReSim[2],ReSim[11], ReSim[20],ReSim[2]+ReSim[11]+ReSim[20], norm([ReSim[2],ReSim[11], ReSim[20]]) )
			
			
			print('M1,M2,M3,M4,M5,M6,M7, M8, M9',M1,M2,M3,M4,M5,M6,M7, M8, M9)
			R1 = max(M1, M2, M3) / min(M1, M2, M3)
			R2 = max(M4, M5, M6) / min(M4, M5, M6)
			R3 = max(M7, M8, M9) / min(M7, M8, M9)
			print('R1,R2,R3',R1,R2,R3)
			MAX1 = [M1, M2, M3]
			MAX2 = [M4, M5, M6]
			MAX3 = [M7, M8, M9]
			#print('MAX1,MAX2,MAX3',MAX1,MAX2,MAX3)
			index1ofMAX1 = MAX1.index(max(MAX1))
			index1ofMAX2 = MAX2.index(max(MAX2))
			index1ofMAX3 = MAX3.index(max(MAX3))
			highestorder = [R1, R2, R3]
			FirstAssign = highestorder.index(max(highestorder))
			print('FirstAssign',FirstAssign)
				
			if FirstAssign == 0 and index1ofMAX1 == 0:
					print('0,0')
					#ID1.append(TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					CS1 = max(M1, M2, M3)/ (M1 + M2 + M3)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 0,1,2
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
					#del TotalCheckBBox[0]#C1
					TotalCheckBBox.remove(TotalCheckBBox[0])
					TotalassumeBBox.remove(TotalassumeBBox[0])
					indices2 = 0,1,2
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					# indicestotalTotalassumeBBox = 0
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 0
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]									
					indices3 = 0,1,2,3,6,9,10,11,12,15,18,19,20,21,24
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					print('-----------------------D1C1 is assigned--------------------------------')
			elif FirstAssign == 0 and index1ofMAX1 == 1:
					#print('0,1')
					#ID1.append(TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					CS1 = max(M1, M2, M3)/ (M1 + M2 + M3)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 3,4,5
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
					#del TotalCheckBBox[1]#C1
					indices2 = 0,1,2
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[0]
					indices3 = 0,1,2,4,7,9,10,11,13,16,18,19,20,22,25
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[1])
					TotalassumeBBox.remove(TotalassumeBBox[0])
					print('-----------------------D1C2 is assigned--------------------------------')										
					# indicestotalTotalassumeBBox = 0
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 1
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]										
			elif FirstAssign == 0 and index1ofMAX1 == 2:
					#print('0,2')
					#ID1.append(TotalCheckBBox[2])
					ID1[checkframenumber] = (TotalCheckBBox[2])
					CS1 = max(M1, M2, M3)/ (M1 + M2 + M3)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 6,7,8
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
					#del TotalCheckBBox[2]#C1
					indices2 = 0,1,2
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[0]
					indices3 = 0,1,2,5,8,9,10,11,14,17,18,19,20,23,26
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[2])
					TotalassumeBBox.remove(TotalassumeBBox[0])
					print('-----------------------D1C3 is assigned--------------------------------')										
					# indicestotalTotalassumeBBox = 0
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 2
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]									  			
			elif FirstAssign == 1 and index1ofMAX2 == 0:
					#print('1,0')
					#ID2.append(TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					CS1 = max(M4, M5, M6)/ (M4 + M5 + M6)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 0,1,2
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
					#del TotalCheckBBox[0]#C1
					indices2 = 3,4,5
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[1]
					indices3 = 0,3,4,5,6,9,12,13,14,15,18,21,22,23,24
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[0])
					TotalassumeBBox.remove(TotalassumeBBox[1])
					print('-----------------------D2C1 is assigned--------------------------------')										
					# indicestotalTotalassumeBBox = 1
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 0
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]											
			elif FirstAssign == 1 and index1ofMAX2 == 1:
					#print('1,1')
					#ID2.append(TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					CS1 = max(M4, M5, M6)/ (M4 + M5 + M6)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 3,4,5
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
                    #print('TotalCheck',len(TotalCheck))
					#del TotalCheckBBox[1]#C1
                    #print('TotalCheckBBox',len(TotalCheckBBox))
                    #del Totalassumed[6:]#D3
					indices2 = 3,4,5
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[1]
					indices3 = 1,3,4,5,7,10,12,13,14,16,19,21,22,23,25
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[1])
					TotalassumeBBox.remove(TotalassumeBBox[1])
					print('-----------------------D2C2 is assigned--------------------------------')									
					# indicestotalTotalassumeBBox = 1
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 1
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]															
			elif FirstAssign == 1 and index1ofMAX2 == 2:
					#print('1,2')
					#ID2.append(TotalCheckBBox[2])
					ID2[checkframenumber] = (TotalCheckBBox[2])
					CS1 = max(M4, M5, M6)/ (M4 + M5 + M6)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 6,7,8
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
                    #print('TotalCheck',len(TotalCheck))
					#del TotalCheckBBox[2]#C1
                    #print('TotalCheckBBox',len(TotalCheckBBox))
                    #del Totalassumed[6:]#D3
					indices2 = 3,4,5
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[1]
					indices3 = 2,3,4,5,8,11,12,13,14,17,20,21,22,23,26
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[2])
					TotalassumeBBox.remove(TotalassumeBBox[1])	
					print('-----------------------D2C3 is assigned--------------------------------')									
					# indicestotalTotalassumeBBox = 1
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 2
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]												
			elif FirstAssign == 2 and index1ofMAX3 == 0:
					#print('2,0')
					#ID3.append(TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					CS1 = max(M7, M8, M9)/ (M7 + M8 + M9)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 0,1,2
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
                    #print('TotalCheck',len(TotalCheck))
					#del TotalCheckBBox[0]#C1
                    #print('TotalCheckBBox',len(TotalCheckBBox))
                    #del Totalassumed[6:]#D3
					indices2 = 6,7,8
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[2]
					indices3 = 0,3,6,7,8,9,12,15,16,17,18,21,24,25,26
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[0])
					TotalassumeBBox.remove(TotalassumeBBox[2])
					print('-----------------------D3C1 is assigned--------------------------------')																				
					# indicestotalTotalassumeBBox = 2
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 0
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]											
			elif FirstAssign == 2 and index1ofMAX3 == 1:
					#print('2,1')
					#ID3.append(TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					CS1 = max(M7, M8, M9)/ (M7 + M8 + M9)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 3,4,5
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
                    #print('TotalCheck',len(TotalCheck))
					#del TotalCheckBBox[1]#C1
                    #print('TotalCheckBBox',len(TotalCheckBBox))
                    #del Totalassumed[6:]#D3
					indices2 = 6,7,8
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[2]
					indices3 = 1,4,6,7,8,10,13,15,16,17,19,22,24,25,26
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[1])
					TotalassumeBBox.remove(TotalassumeBBox[2])
					print('-----------------------D3C2 is assigned--------------------------------')									
					# indicestotalTotalassumeBBox = 2
					# TotalassumeBBox = [i for j, i in enumerate(TotalassumeBBox) if j not in indicestotalTotalassumeBBox]
					# indicestotalcheckbox = 1
					# TotalCheckBBox = [i for j, i in enumerate(TotalCheckBBox) if j not in indicestotalcheckbox]									
			elif FirstAssign == 2 and index1ofMAX3 == 2:
					#print('2,2')
					#ID3.append(TotalCheckBBox[2])
					ID3[checkframenumber] = (TotalCheckBBox[2])
					CS1 = max(M7, M8, M9)/ (M7 + M8 + M9)
					#print('1----------------------------CS1',CS1)
					ConScoID1.append(CS1)
					xaxisframe.append(framesnumber)
					indices1 = 6,7,8
					TotalCheck = [i for j, i in enumerate(TotalCheck) if j not in indices1]
                    #print('TotalCheck',len(TotalCheck))
					#del TotalCheckBBox[2]#C1
                    #print('TotalCheckBBox',len(TotalCheckBBox))
                    #del Totalassumed[6:]#D3
					indices2 = 6,7,8
					Totalassumed = [i for j, i in enumerate(Totalassumed) if j not in indices2]				
					#del TotalassumeBBox[2]
					indices3 = 2,5,6,7,8,11,14,15,16,17,20,23,24,25,26
					ReSim = [i for j, i in enumerate(ReSim) if j not in indices3]
					TotalCheckBBox.remove(TotalCheckBBox[2])
					TotalassumeBBox.remove(TotalassumeBBox[2])
					print('-----------------------D3C3 is assigned--------------------------------')

			ReSim1 = []
			if len(TotalCheck)==6 and len(Totalassumed) == 6 and len(TotalCheckBBox) == 2 and len(TotalassumeBBox) == 2:
				M = 0
				N = 0
				for k in range(3):
					if k==0:
						smart = 0
						M = 0
						N = 0
					elif k==1:
						smart = 1
						M = 1
						N = 1
					elif k==2:
						smart = 2
						M = 2
						N = 2	
					for l in range(2):
						for m in range(2):
							#print('Mafter,Nafter',k, M,N)
							PR = TotalCheck[N]
							GT = Totalassumed[M]
							#print(len(PR),len(GT))
							A = np.squeeze(np.array(GT))
							B = np.squeeze(np.array(PR))
							cosinesimilarity = np.dot(A,B)/(norm(A)*norm(B))
                            #cosinesimilarity= wasserstein_distance(A,B)
                            #_,cosinesimilarity = ks_2samp(A, B)
							ReSim1.append(cosinesimilarity)
							N+=3
						N=smart
						M+=3
						outputsimilarity.append([checkframenumber, cosinesimilarity])#across temporal frames.
					#print('outputsimilarity',outputsimilarity)
					print('----------------ReSimafter------------',(ReSim1))


			#print('ReSim',len(ReSim1))
			#TotalCheckBBox = [TotalCheckBBox[i] for i in [0,1]]#swapping
			R_D1C1 = ReSim[0]#(0,0)
			R_D1C2 = ReSim[1]#(0,3)
			R_D2C1 = ReSim[2]#(3,0)
			R_D2C2 = ReSim[3]#(3,3)
			G_D1C1 = ReSim[4]#(1,1)
			G_D1C2 = ReSim[5]#(1,4)
			G_D2C1 = ReSim[6]#(4,1)
			G_D2C2 = ReSim[7]#(4,4)
			B_D1C1 = ReSim[8]#(2,2)
			B_D1C2 = ReSim[9]#(2,5)
			B_D2C1 = ReSim[10]#(5,2)
			B_D2C2 = ReSim[11]#(5,5)
			
			MM1 = norm([ReSim[0], ReSim[4], ReSim[8]])
			MM2 = norm([ReSim[1], ReSim[5], ReSim[9]])
			MM3 = norm([ReSim[2], ReSim[6], ReSim[10]])
			MM4 = norm([ReSim[3], ReSim[7], ReSim[11]])


			print('AssumeBox', TotalassumeBBox)
			print('M1,M2,M3,M4',checkframenumber,MM1,MM2,MM3,MM4)
			MR1 = max(MM1, MM2) / min(MM1, MM2)
			MR2 = max(MM3, MM4) / min(MM3, MM4)
			print('R1,R2',MR1,MR2)
			Mhighestorder = [MR1,MR2]
			MFirstAssign = Mhighestorder.index(max(Mhighestorder))
			print('MFirstAssign',MFirstAssign,checkframenumber)
			Actual = [MM1, MM2, MM3, MM4]
			Switched = [MM1, MM3, MM2, MM4]
			print('Actual,',Actual)
			#print('Switched,',Switched)
			MMAX1 = [MM1, MM2]
			MMAX2 = [MM3, MM4]
			index1ofMMAX1 = MMAX1.index(max(MMAX1))
			index1ofMMAX2 = MMAX2.index(max(MMAX2))	
			print('index1ofMMAX1,index1ofMMAX2',index1ofMMAX1,index1ofMMAX2)
			print('FirstAssign, index1ofMAX1, index1ofMAX2,index1ofMAX3, MFirstAssign, index1ofMMAX1, index1ofMMAX2',FirstAssign, index1ofMAX1, index1ofMAX2,index1ofMAX3, MFirstAssign, index1ofMMAX1, index1ofMMAX2)
			print('backupTotalCheckBBox',backupTotalCheckBBox)
			print('TotalCheckBBox',TotalCheckBBox)
			print(FirstAssign,index1ofMAX1,MFirstAssign)
			
			if FirstAssign == 0:
				if index1ofMAX1 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 0 0 0 0')
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX1 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 0 0 0 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX1 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 0 0 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX1 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 0 0 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX1 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 0 1 0 0')									
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX1 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 0 1 0 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX1 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==0:  
					print('imhere 0 1 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]  
				elif index1ofMAX1 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 0 1 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]  
				elif index1ofMAX1 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 0 2 0 0')									
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]  
				elif index1ofMAX1 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 0 2 0 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]         
				elif index1ofMAX1 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 0 2 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]                 
				elif index1ofMAX1 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 0 2 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]                 
			elif FirstAssign == 1:
				if index1ofMAX2 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 1 0 0 0')
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX2 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 1 0 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX2 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 1 0 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX2 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 1 0 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX2 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 1 1 0 0')									
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]
				elif index1ofMAX2 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 1 1 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX2 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==0:  
					print('imhere 1 1 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]  
				elif index1ofMAX2 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 1 1 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]  
				elif index1ofMAX2 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 1 2 0 0')									
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID3[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]  
				elif index1ofMAX2 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 1 2 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID3[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]         
				elif index1ofMAX2 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 1 2 1 0')									
					ID3[checkframenumber] = (TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]         
				elif index1ofMAX2 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 1 2 1 1')									
					ID3[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]                 
			elif FirstAssign == 2:
				if index1ofMAX3 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 2 0 0 0')
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX3 == 0 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 2 0 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX3 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 2 0 1 0')									
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[1],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX3 == 0 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 2 0 1 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[2],backupTotalCheckBBox[0]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2]]
				elif index1ofMAX3 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 2 1 0 0')									
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX3 == 1 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 2 1 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX3 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==0:  
					print('imhere 2 1 1 0')									
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[2],backupTotalCheckBBox[0],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]
				elif index1ofMAX3 == 1 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 2 1 1 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[2],backupTotalCheckBBox[1]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5]]  
				elif index1ofMAX3 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==0:
					print('imhere 2 2 0 0')									
					ID1[checkframenumber] = (TotalCheckBBox[0])
					ID2[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]  
				elif index1ofMAX3 == 2 and MFirstAssign == 0 and index1ofMMAX1 ==1:
					print('imhere 2 2 0 1')									
					ID1[checkframenumber] = (TotalCheckBBox[1])
					ID2[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]         
				elif index1ofMAX3 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==0:
					print('imhere 2 2 1 0')									
					ID2[checkframenumber] = (TotalCheckBBox[0])
					ID1[checkframenumber] = (TotalCheckBBox[1])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[1],backupTotalCheckBBox[0],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]]         
				elif index1ofMAX3 == 2 and MFirstAssign == 1 and index1ofMMAX2 ==1:
					print('imhere 2 2 1 1')									
					ID2[checkframenumber] = (TotalCheckBBox[1])
					ID1[checkframenumber] = (TotalCheckBBox[0])
					print('backupTotalCheckBBox',backupTotalCheckBBox)
					F1 = [backupTotalCheckBBox[0],backupTotalCheckBBox[1],backupTotalCheckBBox[2]]
					F1hist = [backupTotalCheck[0],backupTotalCheck[1],backupTotalCheck[2],backupTotalCheck[3],backupTotalCheck[4],backupTotalCheck[5],backupTotalCheck[6],backupTotalCheck[7],backupTotalCheck[8]] 
			
			NEWF1 = F1
			NEWF1hist = F1hist 
			print('NEWF1',NEWF1)
		controllingfactor = 1# important for backtrack
		ReSim = []
		CS1 = 0
		path1 = os.path.join(Sourcepath,'Black.npy')
		path2 = os.path.join(Sourcepath,'White.npy')
		path3 = os.path.join(Sourcepath,'Cyan.npy')
		path4 = os.path.join(Sourcepath,'framenumber.npy')
		np.save(path1, ID1)
		np.save(path2, ID2)
		np.save(path3, ID3)
		np.save(path4,actualframenumber)
		Totalassumed = []   
		#insidebacktrack=0
		#actualframenumber.append(checkframenumber)
		if checkmodule[2] == 3:
		    TotalassumeBBox = []
		elif checkmodule[2] <= 2:
			TotalassumeBBox = TotalassumeBBox
	else:
		controllingfactor = 0
		#insidebacktrack=0

# np.save('ID1', ID1)
# np.save('ID2', ID2)
# np.save('ID3', ID3)
# np.save('framenumber',actualframenumber)

def remove_dup(a):
   i = 0
   while i < len(a):
      j = i + 1
      while j < len(a):
         if a[i] == a[j]:
            del a[j]
         else:
            j += 1
      i += 1
print('beforee',len(actualframenumber))
remove_dup(actualframenumber)
print('afterrr',len(actualframenumber))
#print(actualframenumber)


save_path_new = str(pathlib.Path(__file__).parent.resolve())

fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
plt.plot(xaxisframe, ConScoID1, marker = 'o')
plt.title('Video')
plt.xlabel('Framenumber')
plt.ylabel('Confidence Score')
#plt.show()
fig.savefig(os.path.join(save_path_new,'1.png'))   # save the figure to file


CSFREAME = ConScoID1
print('CSFREAME', len(CSFREAME))
lengthforplot = min(len(ID1),len(ID2),len(ID3),len(CSFREAME))
print('lengthforplot',lengthforplot)
#video1=cv2.VideoCapture('/home/monsley/Downloads/yolov7-object-tracking-main/20sec/2.mp4')
video1=cv2.VideoCapture(videofilename)
print('Videofilename',video1)
total_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

#total_frames = round(video1.get(cv2.CAP_PROP_FRAME_COUNT))
print('total_frames',total_frames)

width  = video1.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = video1.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

fourcc3 = cv2.VideoWriter_fourcc(*'mp4v')
frame_size=(width, height)


#output3 = cv2.VideoWriter('/home/monsley/Downloads/yolov7-object-tracking-main/processed_op/100723/2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (984,984), True)
outputvideo = os.path.join(Sourcepath,'output.avi')
output3 = cv2.VideoWriter(outputvideo, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (984,984), True)

#save_path_new = str(pathlib.Path(__file__).parent.resolve())
path1 = os.path.join(Sourcepath,'Black.npy')
path2 = os.path.join(Sourcepath,'White.npy')
path3 = os.path.join(Sourcepath,'Cyan.npy')

print('path1',path1)

ID1 = np.load(path1, allow_pickle=True).reshape(1)
ID2 = np.load(path2, allow_pickle=True).reshape(1)
ID3 = np.load(path3, allow_pickle=True).reshape(1)


colo = []
colo1 = []
colo2 = []
IFN = 2
incrementcheck = 0
incrementcheck1 = 0
for framesnumber in range(2,total_frames-1):#12490
	ret, frame = video1.read()
	#print('ret',framesnumber, ret)
	for insideframesnumber in (ID1[0].keys()):
		#print('insideframesnumber',insideframesnumber)
		if framesnumber ==ID1[0][insideframesnumber][0][0]:
			anboxdeta = ID1[0][insideframesnumber][0]
			ax = round(int(anboxdeta[1]))
			ay = round(int(anboxdeta[2]))
			aw = round(int(anboxdeta[3]))
			ah = round(int(anboxdeta[4]))
			cv2.rectangle(frame, (ax, ay), (aw, ah), (0, 0, 0),3)#black
		elif framesnumber !=ID1[0][insideframesnumber][0][0]:
			#print('framesnumber1',framesnumber)
			continue
		if framesnumber ==ID2[0][insideframesnumber][0][0]:
			anboxdeta2 = ID2[0][insideframesnumber][0]
			bx = round(int(anboxdeta2[1]))
			by = round(int(anboxdeta2[2]))
			bw = round(int(anboxdeta2[3]))
			bh = round(int(anboxdeta2[4]))		
			cv2.rectangle(frame, (bx, by), (bw, bh), (255, 255, 255),3)#white
		elif framesnumber !=ID2[0][insideframesnumber][0][0]:
			#print('framesnumber2',framesnumber)
			continue
		if framesnumber ==ID3[0][insideframesnumber][0][0]:
			anboxdeta3 = ID3[0][insideframesnumber][0]
			cx = round(int(anboxdeta3[1]))
			cy = round(int(anboxdeta3[2]))
			cw = round(int(anboxdeta3[3]))
			ch = round(int(anboxdeta3[4]))		
			cv2.rectangle(frame, (cx, cy), (cw, ch), (255, 255, 0),3)#cyan color
		elif framesnumber !=ID3[0][insideframesnumber][0][0]:
			#print('framesnumber3',framesnumber)
			continue
		cv2.putText(frame,  str(insideframesnumber),  org = (200, 200),  fontFace = cv2.FONT_HERSHEY_DUPLEX,  fontScale = 3.0,  color = (125, 246, 55),  thickness = 3)		
	output3.write(frame)

print('COMPLETED MAN!')
