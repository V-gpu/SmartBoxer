
import os
import cv2
import pathlib
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

global video1


save_path_new = str(pathlib.Path(__file__).parent.resolve()) #give the current working directory
CWD = str(pathlib.Path(__file__).parent.parent) #give the main working directory

fullpath = '0SegmentedVideos'
fullpathprocess = 'Toprocess'
fullpathID = '2Tracking'

Count = 0

save_path_new1 = CWD+'/'+fullpathID # the folder with the IDS


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filelist")
args = parser.parse_args()
#config = vars(args)
#print(f'{args.fullpathpickle}')

videofilename = args.filelist
nameofvideo = os.path.split(videofilename)[-1]
nameoffolder,ext = os.path.splitext(nameofvideo)

#Load the video
#filelist = glob.glob(CWD+'/'+fullpath+'/'+fullpathprocess+'/'+"*.avi")  # Get all files in the current folder
# for file in filelist:
#     videofilename = file
#     nameofvideo = os.path.split(file)[-1]
#     nameoffolder,ext = os.path.splitext(nameofvideo)

video1=cv2.VideoCapture(videofilename)

total_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
width  = video1.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = video1.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fourcc3 = cv2.VideoWriter_fourcc(*'mp4v')
frame_size=(width, height)

#folder creation
CWD = str(pathlib.Path(__file__).parent.parent) #give the main working directory


metricfolder = '4Metrics'

nameofvideopath = CWD+'/'+ metricfolder+'/' + nameoffolder
isExist = os.path.exists(nameofvideopath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(nameofvideopath)
   print("The video directory is created!")

Directionalhistogrampath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "Directionalhistogram"
Zonemanagmentpath = CWD+'/'+ metricfolder+'/' + nameoffolder  +'/'+ "Zonemanagment"
Hotspotpath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "Hotspot"
EngageDisengagepath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "EngageDisengage"
# Check whether the specified path exists or not

isExist = os.path.exists(Directionalhistogrampath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(Directionalhistogrampath)
   print("The Directionalhistogram directory is created!")
isExist = os.path.exists(Zonemanagmentpath)   
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(Zonemanagmentpath)
   print("The Zonemanagment directory is created!")
isExist = os.path.exists(Hotspotpath)   
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(Hotspotpath)
   print("The Hotspot directory is created!")
isExist = os.path.exists(EngageDisengagepath)     
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(EngageDisengagepath)
   print("The EngageDisengagep directory is created!")


#outputvideo = os.path.join(save_path_new,nameofvideo)
outputvideo = os.path.join(nameofvideopath,nameofvideo)
print('outputvideo',outputvideo)
output3 = cv2.VideoWriter(outputvideo, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (984,984), True)

orderingfile = 0
Sourcepath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "Sourcefolder"

# for file in os.listdir(Sourcepath):
# 	# Sourcepath = file
# 	if orderingfile == 0:
# 		path1 = os.path.join(Sourcepath, file)
# 		orderingfile +=1
# 	if orderingfile == 1:
# 		path2 = os.path.join(Sourcepath, file)
# 	# path3 = os.path.join(directory_path, Sourcepath,'Sourcefolder','Cyan.npy')

# 	# print('path1',path1)
# 	# print('path2',path2)

# 	# print('path3',path3)
# 	ID1 = np.load(path1, allow_pickle=True).reshape(1)#Black
# 	ID2 = np.load(path2, allow_pickle=True).reshape(1)#White

path1 = os.path.join(Sourcepath,'Black.npy')
path2 = os.path.join(Sourcepath,'White.npy')
path3 = os.path.join(Sourcepath,'Cyan.npy')

# #print('path1',path1)

ID1 = np.load(path1, allow_pickle=True).reshape(1)#Black
ID2 = np.load(path2, allow_pickle=True).reshape(1)#White
ID3 = np.load(path3, allow_pickle=True).reshape(1)#Cyan

CD1 = []
CD2 = []
CD3 = []
colo1 = []
colo2 = []
IFN = 2
incrementcheck = 0
incrementcheck1 = 0

D1F1 = None
D2F1 = None
D3F1 = None

#inputtin which ID to chose

# D1 = input('Enter D1:')
# D2 = input('Enter D2:')
# D3 = input('Enter D3:')
D1 =''
D2= 'T'
D3 = 'T'



# print('D1, D2',D1, D2)


if D1 and D2:
	print('inside D1 D2')
	first_key_D1 = list(ID1[0].keys())[0]
	D1F1 =ID1[0][first_key_D1][0]
	D1F1Y = round(int(D1F1[2]))
	first_key_D2= list(ID2[0].keys())[0]	
	D2F1 =ID2[0][first_key_D2][0]
	D2F1Y = round(int(D2F1[2]))
	MD1 = ID1
	MD2 = ID2
	x1 = list(ID1[0].keys())
	y1 = list(ID2[0].keys())
	XY=list(set(x1).intersection(y1))	
	if D1F1Y < D2F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		ColorD1 = (255,0,0)# Blue
		ColorD2 = (0,0,255)# Red
		COLOR1 = (1,0,0)
		COLOR2 = (0,0,1)
		C1 = 'blue'
		C2 = 'red'
	elif D1F1Y > D2F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		ColorD2 = (255,0,0)# Blue
		ColorD1 = (0,0,255)# Red
		COLOR2 = (1,0,0)
		COLOR1 = (0,0,1)
		C2 = 'blue'
		C1 = 'red'				
elif D1 and D3:
	print('inside D1 D3')
	first_key_D1 = list(ID1[0].keys())[0]	
	D1F1 =ID1[0][first_key_D1][0]
	D1F1Y = round(int(D1F1[2]))
	first_key_D3 = list(ID3[0].keys())[0]	
	D3F1 =ID3[0][first_key_D3][0]
	D3F1Y = round(int(D3F1[2]))
	MD1 = ID1
	MD2 = ID3
	x1 = list(ID1[0].keys())
	y1 = list(ID3[0].keys())
	XY=list(set(x1).intersection(y1))		
	if D1F1Y < D3F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		ColorD1 = (255,0,0)# Blue
		ColorD3 = (0,0,255)# Red
		COLOR1 = (1,0,0)
		COLOR2 = (0,0,1)
		C1 = 'blue'
		C2 = 'red'				
	elif D1F1Y > D3F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		ColorD3 = (255,0,0)# Blue
		ColorD1 = (0,0,255)# Red
		COLOR2 = (1,0,0)
		COLOR1 = (0,0,1)
		C2 = 'blue'
		C1 = 'red'				 
elif D2 and D3:
	print('inside D2 D3')	
	first_key_D2= list(ID2[0].keys())[0]	
	D2F1 =ID2[0][first_key_D2][0]
	D2F1Y = round(int(D2F1[2]))
	first_key_D3 = list(ID3[0].keys())[0]	
	D3F1 =ID3[0][first_key_D3][0]
	D3F1Y = round(int(D3F1[2]))
	MD1 = ID2
	MD2 = ID3
	x1 = list(ID2[0].keys())
	y1 = list(ID3[0].keys())
	XY=list(set(x1).intersection(y1))			
	if D2F1Y < D3F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		print('inside 1')
		ColorD2 = (255,0,0)# Blue
		ColorD3 = (0,0,255)# Red
		COLOR1 = (1,0,0)
		COLOR2 = (0,0,1)
		C1 = 'blue'
		C2 = 'red'				
	elif D2F1Y > D3F1Y: # D1 is closer to up blue , D2 is closer to down  red 
		ColorD3 = (255,0,0)# Blue
		ColorD2 = (0,0,255)# Red 
		COLOR2 = (1,0,0)
		COLOR1 = (0,0,1)
		C2 = 'blue'
		C1 = 'red'			
# color needs to be chosen based on the position of boxers 


#Manual intervention is required to spot the boxers and the referee 



x1 = list(range(2,total_frames-1))
y1 = XY
finalXY=list(set(x1).intersection(y1))



for framesnumber in range(2,total_frames-1):#12490
	ret, frame = video1.read()
	print('ret',framesnumber, ret)
	for insideframesnumber in (ID1[0].keys()):
		print('insideframesnumber',insideframesnumber)
		if framesnumber ==ID1[0][insideframesnumber][0][0] and D1:
			print('framesnumbers',framesnumber)
			anboxdeta = ID1[0][insideframesnumber][0]
			ax = round(int(anboxdeta[1]))
			ay = round(int(anboxdeta[2]))
			aw = round(int(anboxdeta[3]))
			ah = round(int(anboxdeta[4]))
			CNT = (int((ax+aw)/2), int((ay+ah)/2))
			#print('ColorD1',ColorD1)
			#cv2.circle(frame, (CNT[0],CNT[1]), 2, (255,0,0), -1)
			cv2.rectangle(frame, (ax, ay), (aw, ah), ColorD1,3)
			#cv2.circle(frame, (CNT[0],CNT[1]), 2, ColorD1, -1)
			CD1.append(CNT)
			colo1 = CD1
			#cv2.line(frame,(colo[insideframesnumber-1]), (colo[insideframesnumber]) ,(0,0,255),2)	
			
			#cv2.rectangle(frame, (ax, ay), (aw, ah), (255, 0, 255),3)
			#cv2.rectangle(frame, (ax, ay), (aw, ah), (255, 0, 0),3)
			#cv2.rectangle(frame, (ax, ay), (aw, ah), (255, 0, 0),3)
			#cv2.putText(frame,str(actualframenumber[insideframesnumber]),org = (200, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			cv2.putText(frame,str([insideframesnumber]),org = (200, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			#output1.write(frame)
		elif framesnumber !=ID1[0][insideframesnumber][0][0]:
			#print('framesnumber1',framesnumber)
			continue
		if framesnumber ==ID2[0][insideframesnumber][0][0] and D2:
			anboxdeta2 = ID2[0][insideframesnumber][0]
			bx = round(int(anboxdeta2[1]))
			by = round(int(anboxdeta2[2]))
			bw = round(int(anboxdeta2[3]))
			bh = round(int(anboxdeta2[4]))		
			cv2.rectangle(frame, (bx, by), (bw, bh), ColorD2,3)
			CNT1 = (int((bx+bw)/2), int((by+bh)/2))
			#print('0',CNT1[0])
			#print('1',CNT1[1])			
			#cv2.circle(frame, (CNT1[0],CNT1[1]), 4, ColorD2, -1)
			CD2.append(CNT1)
			colo2 = CD2
			#print('ColorD2',ColorD2)
			# if incrementcheck>2:
			# 	cv2.line(frame,(colo1[IFN-1]), (colo1[IFN]) ,(0, 0, 255),2)			
			#cv2.putText(frame,str(actualframenumber[insideframesnumber]),org = (200, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			cv2.putText(frame,str([insideframesnumber]),org = (200, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			#print('2')
			#output2.write(frame)
			# incrementcheck +=1
			#print('colo1',colo1)
		elif framesnumber !=ID2[0][insideframesnumber][0][0]:
			#print('framesnumber2',framesnumber)
			continue
		if framesnumber ==ID3[0][insideframesnumber][0][0] and D3:
			anboxdeta3 = ID3[0][insideframesnumber][0]
			cx = round(int(anboxdeta3[1]))
			cy = round(int(anboxdeta3[2]))
			cw = round(int(anboxdeta3[3]))
			ch = round(int(anboxdeta3[4]))		
			#cv2.rectangle(frame, (cx, cy), (cw, ch), (0, 255, 255),3)
			cv2.rectangle(frame, (cx, cy), (cw, ch), ColorD3,3)
			CNT2 = (int((cx+cw)/2), int((cy+ch)/2))
			#print('00',CNT2[0])
			#print('11',CNT2[1])			
			#cv2.circle(frame, (CNT2[0],CNT2[1]),4, ColorD3, -1)
			CD3.append(CNT2)
			colo3 = CD3
			#print('ColorD3',ColorD3)
			# if incrementcheck1>2:
            #                   cv2.line(frame,(colo2[IFN-1]), (colo2[IFN]) ,(255, 0, 0),2)
            #                   IFN +=1			
			#cv2.putText(frame,str(CSFREAME[insideframesnumber]),org = (700, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			cv2.putText(frame,str([insideframesnumber]),org = (200, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.0, color = (255, 255, 255),thickness = 2)
			#output3.write(frame)
			# incrementcheck1+=1
			#IFN +=1
			#print('colo1',colo2)
		elif framesnumber !=ID3[0][insideframesnumber][0][0]:
			#print('framesnumber3',framesnumber)
			continue
	#print('IFN',colo1, colo2,IFN)
	#print('LENGTH', len(CD1), len(CD2))
	if D1 and D2 :
		colo1 = CD1
		colo2 = CD2
		#print('D1 D2')
		for i in range(2,min([len(colo1), len(colo2)])):
			cv2.line(frame,(colo1[i-1]), (colo1[i]) ,ColorD1,6)
			cv2.line(frame,(colo2[i-1]), (colo2[i]) ,ColorD2,6)
	elif D1 and D3:	
		colo1 = CD1
		colo2 = CD3	
		#print('D1 D3')
		for i in range(2,min([len(colo1), len(colo2)])):
			cv2.line(frame,(colo1[i-1]), (colo1[i]) ,ColorD1,6)
			cv2.line(frame,(colo2[i-1]), (colo2[i]) ,ColorD3,6)				
	elif D2 and D3 :	
		colo1 = CD2
		colo2 = CD3	
		#print('D2 D3')
		for i in range(2,min([len(colo1), len(colo2)])):
			cv2.line(frame,(colo1[i-1]), (colo1[i]) ,ColorD2,6)
			cv2.line(frame,(colo2[i-1]), (colo2[i]) ,ColorD3,6)			
	# for i in range(2,min([len(colo1), len(colo2)])):
    #             cv2.line(frame,(colo1[i-1]), (colo1[i]) ,(0, 0, 255),6)
    #             cv2.line(frame,(colo2[i-1]), (colo2[i]) ,(255, 0, 0),6)
	output3.write(frame)

print('COMPLETED MAN!')



# ----------------------------------------Metric generation ------------------------------------------------#



# #save_path_new1 = CWD+'/'+fullpathID # the folder with the IDS
# Sourcepath = CWD+'/'+ metricfolder+'/' + nameoffolder +'/'+ "Sourcefolder"
# # save_path_new1 = Sourcepath
# # print('save_path_new1',save_path_new1)

# path1 = os.path.join(Sourcepath,'Black.npy')
# path2 = os.path.join(Sourcepath,'White.npy')
# path3 = os.path.join(Sourcepath,'Cyan.npy')


# D1 = np.load(path1, allow_pickle=True).reshape(1)#black
# D2 = np.load(path2, allow_pickle=True).reshape(1)#white
# D3 = np.load(path3, allow_pickle=True).reshape(1)#cyan

# ID1 = D1
# ID2 = D2


print('MD1', len(MD1))
print('MD2', len(MD2))

# P = []
# PP = []
# # PP = []
# PF1 = []
# PF2 = []

# for insideframesnumber in (ID1[0].keys()):
# 	anboxdeta = ID1[0][insideframesnumber][0]
# 	if len(anboxdeta):
# 		P.append(anboxdeta)
# 		filename = 1
# 		PF1.append(anboxdeta[0])
# 	else:
# 		#P.append([])
# 		continue

# for insideframesnumber in (ID2[0].keys()):
# 	anboxdeta1 = ID2[0][insideframesnumber][0]
# 	if len(anboxdeta1):
# 		PP.append(anboxdeta1)
# 		filename = 2
# 		PF2.append(anboxdeta1[0])
# 	else:
# 		#P.append([])
# 		continue

# print('F1111',max(PF1))

# print('F1111',max(PF1))

# FPS = 60
# TimeLimit = 20
# DataLimit = FPS*TimeLimit
# print('DataLimit',DataLimit)
# Twentysecdata = round((max(PF1))/(1200))
# print('Twentysecdata',Twentysecdata)

# ar1 = np.array(PF1)
# ar2 = np.array(PF2)

# def find_nearest(array, value):
# 	array = np.asarray(array)
# 	idx = (np.abs(array - value)).argmin()
# 	print('idx',idx)
# 	return idx, array[idx]

# P1idxlst = []
# P2idxlst = []
# for value1 in range(1,Twentysecdata):
# 	idx1, I1 = (find_nearest(ar1, value=DataLimit* value1))
# 	P1idxlst.append(idx1)
# 	idx2, I2 = (find_nearest(ar2, value=DataLimit* value1))
# 	P2idxlst.append(idx2)

# print('P1idxlst',P1idxlst)
# print('P2idxlst',P2idxlst)
# lastvalue = (min(len(P1idxlst), len(P2idxlst)))
# print('lastvalue',lastvalue)

# for k in range(0, lastvalue+1):
# 	print('k',k)
# 	if k==0:
# 		print(k,0,P1idxlst[(k)])
# 		person1 = P[0:P1idxlst[(k)]]
# 		person2 = PP[0:P2idxlst[(k)]]		
# 	elif k>=1 and k<lastvalue:
# 		print('kk',k, P1idxlst[(k-1)],P1idxlst[(k)])
# 		person1 = P[P1idxlst[(k-1)]:P1idxlst[(k)]]
# 		person2 = PP[P2idxlst[(k-1)]:P2idxlst[(k)]]			
# 	elif k == lastvalue:
# 		print('lastvalue',P1idxlst[lastvalue-1])
# 		person1 = P[:P1idxlst[lastvalue-1]]
# 		person2 = PP[:P2idxlst[lastvalue-1]]			




def directionalhistogram (MD1, MD2):
	ID1 = MD1
	ID2 = MD2
	#def Directional_Graph(contents):
	Ncount = 0
	NEcount = 0
	Ecount = 0
	SEcount = 0
	Scount = 0
	SWcount = 0
	Wcount = 0
	NWcount = 0
	Ncount1 = 0
	NEcount1 = 0
	Ecount1 = 0
	SEcount1 = 0
	Scount1 = 0
	SWcount1 = 0
	Wcount1 = 0
	NWcount1 = 0
	P = []
	PP=[]
	ind = np.arange(4)
	width = 0.3
	Extensions = ind+width
	FPS = 60
	TimeLimit = 20
	print('Extensions',Extensions)


	def addlabels(x,y):
		for i in range(len(x)):
			plt.text(i,y[i],y[i])

	P = []
	PP = []
	# PP = []
	PF1 = []
	PF2 = []

	for insideframesnumber in (ID1[0].keys()):
		anboxdeta = ID1[0][insideframesnumber][0]
		if len(anboxdeta):
			P.append(anboxdeta)
			filename = 1
			PF1.append(anboxdeta[0])
		else:
			#P.append([])
			continue

	for insideframesnumber in (ID2[0].keys()):
		anboxdeta1 = ID2[0][insideframesnumber][0]
		if len(anboxdeta1):
			PP.append(anboxdeta1)
			filename = 2
			PF2.append(anboxdeta1[0])
		else:
			#P.append([])
			continue

	print('F1111',max(PF1))

	print('F1111',max(PF1))

	FPS = 60
	TimeLimit = 20
	DataLimit = FPS*TimeLimit
	print('DataLimit',DataLimit)
	Twentysecdata = round((max(PF1))/(1200))
	print('Twentysecdata',Twentysecdata)

	ar1 = np.array(PF1)
	ar2 = np.array(PF2)

	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		print('idx',idx)
		return idx, array[idx]

	P1idxlst = []
	P2idxlst = []
	for value1 in range(1,Twentysecdata):
		idx1, I1 = (find_nearest(ar1, value=DataLimit* value1))
		P1idxlst.append(idx1)
		idx2, I2 = (find_nearest(ar2, value=DataLimit* value1))
		P2idxlst.append(idx2)

	print('P1idxlst',P1idxlst)
	print('P2idxlst',P2idxlst)
	lastvalue = (min(len(P1idxlst), len(P2idxlst)))
	print('lastvalue',lastvalue)

	for k in range(0, lastvalue+1):
		print('k',k)
		if k==0:
			print(k,0,P1idxlst[(k)])
			person1 = P[0:P1idxlst[(k)]]
			person2 = PP[0:P2idxlst[(k)]]
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k>=1 and k<lastvalue:
			print('kk',k, P1idxlst[(k-1)],P1idxlst[(k)])
			person1 = P[P1idxlst[(k-1)]:P1idxlst[(k)]]
			person2 = PP[P2idxlst[(k-1)]:P2idxlst[(k)]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k == lastvalue:
			print('lastvalue',P1idxlst[lastvalue-1])
			person1 = P[:P1idxlst[lastvalue-1]]
			person2 = PP[:P2idxlst[lastvalue-1]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))
		
		Ncount = 0
		Ecount = 0
		Scount = 0
		Wcount = 0
		Ncount1 = 0
		Ecount1 = 0
		Scount1 = 0
		Wcount1 = 0


		for kl in range(len(person1)-1):
		
			POL=person1[kl]
			POL1=person1[kl+1]
			# print('Ncount, Ecount, Scount, Wcount',Ncount, Ecount, Scount, Wcount)
			#print('1',POL,POL1)
			if len(POL) and len(POL1) == 0:
				print('Detection miss', kl)
			elif len(POL) and len(POL1) != 0:
				#Act=POL
				#Actx=round(Act[0])
				#Acty=round(Act[1])
				#print('NOTEMPTY', kl)
				X1Y1=person1[kl]
				X2Y2=person1[kl+1]						
				bx = round(int(X1Y1[1]))
				by = round(int(X1Y1[2]))
				bw = round(int(X1Y1[3]))
				bh = round(int(X1Y1[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)				
				#print(X1Y1)
				bx1 = round(int(X2Y2[1]))
				by1 = round(int(X2Y2[2]))
				bw1 = round(int(X2Y2[3]))
				bh1 = round(int(X2Y2[4]))	
				Actx1=round((bx1+bw1)/2)
				Acty1=round((by1+bh1)/2)				
				X1=Actx
				Y1=Acty
				X2=Actx1
				Y2=Acty1
				deltaX = X2 - X1
				deltaY = Y2 - Y1
				#print('deltaX, deltaY',deltaX, deltaY)
				theta = (math.atan2(deltaY,deltaX)/math.pi)*180
				degrees_temp = theta
				
				if degrees_temp < 0:
					degrees_final = degrees_temp+360
					compass_brackets = ["E", "S", "W","N", "E"]
				else:
					degrees_final = degrees_temp
					compass_brackets =  ["E", "S", "W", "N", "E"] 
				
				#print('theta', round(degrees_final,2))
				compass_lookup = abs(round(degrees_final/90))
				D = compass_brackets[compass_lookup]
				#print('D',D)
				Th = round(degrees_final,2)
				#print('Th',Th)
				if D == "N":
					Ncount += 1
				elif D == "E":
					Ecount += 1		
				elif D == "S":
					Scount += 1
				elif D == "W":
					Wcount += 1
							
			s = ['F', 'L', 'B', 'R']
			sop = ['Q'] 
			# listcontaining = [Ncount, Ecount, Scount, Wcount]
			listcontaining = [100, 100, 100, 100]
			# print('Ncount, Ecount, Scount, Wcount',Ncount, Ecount, Scount, Wcount)
		# plt.bar(ind,listcontaining,width, color=COLOR1)#blue
		# plt.xticks([])
		# print('listcontaining',listcontaining)
		# Ncount = 0
		# Ecount = 0
		# Scount = 0
		# Wcount = 0

		# plt.savefig(Directionalhistogrampath + '/'+'Directional Histogram123' + str(k) +'.tiff', dpi = 300)
		# print('Ncount1, Ecount1, Scount1, Wcount1',Ncount1, Ecount1, Scount1, Wcount1)

		for kl in range(len(person2)-1):
	
			POL2=person2[kl]
			POL21=person2[kl+1]
			#print('2',POL2,POL21)		
			if len(POL2) and len(POL21) == 0:
				print('Detection miss', kl)
			elif len(POL2) and len(POL21) != 0:
				#Act=POL
				#Actx=round(Act[0])
				#Acty=round(Act[1])
				#print('NOTEMPTY',kl)	
				#print(kl)
				X1Y1=person2[kl]
				X2Y2=person2[kl+1]
				bx = round(int(X1Y1[1]))
				by = round(int(X1Y1[2]))
				bw = round(int(X1Y1[3]))
				bh = round(int(X1Y1[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)				
				#print(X1Y1)
				bx1 = round(int(X2Y2[1]))
				by1 = round(int(X2Y2[2]))
				bw1 = round(int(X2Y2[3]))
				bh1 = round(int(X2Y2[4]))	
				Actx1=round((bx1+bw1)/2)
				Acty1=round((by1+bh1)/2)				
				X1=Actx
				Y1=Acty
				X2=Actx1
				Y2=Acty1
				deltaX = X2 - X1
				deltaY = Y2 - Y1
				#print('deltaX, deltaY',deltaX, deltaY)
				theta = (math.atan2(deltaY,deltaX)/math.pi)*180
				degrees_temp = theta
				#print('degrees_temp',degrees_temp)
				if degrees_temp < 0:
					degrees_final = degrees_temp+360
					compass_brackets = ["E", "S", "W", "N", "E"]
				else:
					degrees_final = degrees_temp
					compass_brackets =  ["E", "S", "W", "N", "E"]
				compass_lookup = abs(round(degrees_final/90))
				#print('compass_lookup',compass_lookup)
				D1 = compass_brackets[compass_lookup]
				#print('D1',D1)
				Th = round(degrees_final,2)
				#print('Th',Th)
				if D1 == "N":
					Ncount1 += 1
				elif D1 == "E":
					Ecount1 += 1		
				elif D1 == "S":
					Scount1 += 1
				elif D1 == "W":
					Wcount1 += 1
			s1 = ['F', 'L', 'B', 'R']
			# print('Ncount1, Ecount1, Scount1, Wcount1',Ncount1, Ecount1, Scount1, Wcount1)				
			listcontaining1 = [Ncount1, Ecount1, Scount1, Wcount1]
			bars = ('Front', 'Left', 'Back', 'Right')
			y_pos = np.arange(4)
			#print('listcontaining', listcontaining1)
		maxlist = max(max(listcontaining), max(listcontaining1))
		print('maxlist',maxlist)
		print('[Ncount/maxlist, Ecount/maxlist, Scount/maxlist, Wcount/maxlist]',[Ncount/maxlist, Ecount/maxlist, Scount/maxlist, Wcount/maxlist])		
		plt.bar(ind,[Ncount/maxlist, Ecount/maxlist, Scount/maxlist, Wcount/maxlist],width, color=COLOR1)#blue
		plt.xticks([])
		# print('listcontaining',listcontaining)


		plt.bar(ind+width,[Ncount1/maxlist, Ecount1/maxlist, Scount1/maxlist, Wcount1/maxlist],width,color=COLOR2)
		plt.xticks([])
		#plt.annotate(str(s1),xy =(Extensions,listcontaining1),  ha='center', va='bottom')
		plt.xticks(y_pos, bars, color='black', rotation=0, fontweight='bold', fontsize='20', horizontalalignment='center')
		plt.yticks(fontsize=20, fontweight='bold')
		# print('listcontaining1',listcontaining1)
		print('[Ncount/maxlist, Ecount/maxlist, Scount/maxlist, Wcount/maxlist]',[Ncount1/maxlist, Ecount1/maxlist, Scount1/maxlist, Wcount1/maxlist])

		plt.savefig(Directionalhistogrampath + '/'+'Directional Histogram' +'_'+ str(k) +'.tiff', dpi = 300)
		listcontaining1 = []
		listcontaining = []
		Ncount = 0
		Ecount = 0
		Scount = 0
		Wcount = 0
		Ncount1 = 0
		Ecount1 = 0
		Scount1 = 0
		Wcount1 = 0		

	return 1


def ENGDIS (MD1, MD2):
	ID1 = MD1
	ID2 = MD2
	TimeLimit = 20

	P = []
	PP = []
	end_point = 20
	start_point = 0
	FPS = 60

	# for insideframesnumber in (ID1[0].keys()):
	# 	anboxdeta = ID1[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		P.append(anboxdeta)
	# 		filename = 1
	# 	else:
	# 		#P.append([])
	# 		continue
	# for insideframesnumber in (ID2[0].keys()):
	# 	anboxdeta = ID2[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		PP.append(anboxdeta)
	# 		filename = 2
	# 	else:
	# 		#P.append([])
	# 		continue


	# DataLimit = FPS*TimeLimit	
	# Twentysecdata = round(min(len(PP),len(P))/(1200))
	# print('Twentysecdata',Twentysecdata)


	# for k in range(Twentysecdata+1):# individual 20 sec and 1 3 minute
	# 	print('------------------------k----------------------', k)
	# 	if k <= Twentysecdata-1:
	# 		person1 = P[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		person2 = PP[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		print('tilllast',len(person1),len(person2),(DataLimit*k)+1,DataLimit*(k+1))
	# 		#print(k)
	# 		print('--------------------------------', len(person1), len(person2))
	# 		start_point = (20 * k) + 1
	# 		end_point =  (20 * k) + 20  
	# 	elif k == Twentysecdata:
	# 		person1 = P[:(DataLimit*(k+1))+1]
	# 		person2 = PP[:(DataLimit*(k+1))+1]
	# 		print('lastbefore',len(person1),len(person2),(DataLimit*(k+1))+1)
	# 		print('--------------------------------', len(person1), len(person2))
	# 		start_point = (20 * k) + 1
	# 		end_point =  (20 * k) + 20 		
	# 	elif k == Twentysecdata+1:
	# 		person1 = P
	# 		person2 = PP
	# 		print('--------------------------------', len(person1), len(person2))
	# 		start_point = 1
	# 		end_point =  (20 * k) 
	P = []
	PP = []
	# PP = []
	PF1 = []
	PF2 = []

	for insideframesnumber in (ID1[0].keys()):
		anboxdeta = ID1[0][insideframesnumber][0]
		if len(anboxdeta):
			P.append(anboxdeta)
			filename = 1
			PF1.append(anboxdeta[0])
		else:
			#P.append([])
			continue

	for insideframesnumber in (ID2[0].keys()):
		anboxdeta1 = ID2[0][insideframesnumber][0]
		if len(anboxdeta1):
			PP.append(anboxdeta1)
			filename = 2
			PF2.append(anboxdeta1[0])
		else:
			#P.append([])
			continue

	print('F1111',max(PF1))

	print('F1111',max(PF1))

	FPS = 60
	TimeLimit = 20
	DataLimit = FPS*TimeLimit
	print('DataLimit',DataLimit)
	Twentysecdata = round((max(PF1))/(1200))
	print('Twentysecdata',Twentysecdata)

	ar1 = np.array(PF1)
	ar2 = np.array(PF2)

	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		print('idx',idx)
		return idx, array[idx]

	P1idxlst = []
	P2idxlst = []
	for value1 in range(1,Twentysecdata):
		idx1, I1 = (find_nearest(ar1, value=DataLimit* value1))
		P1idxlst.append(idx1)
		idx2, I2 = (find_nearest(ar2, value=DataLimit* value1))
		P2idxlst.append(idx2)

	print('P1idxlst',P1idxlst)
	print('P2idxlst',P2idxlst)
	lastvalue = (min(len(P1idxlst), len(P2idxlst)))
	print('lastvalue',lastvalue)

	for k in range(0, lastvalue+1):
		print('k',k)
		if k==0:
			print(k,0,P1idxlst[(k)])
			person1 = P[0:P1idxlst[(k)]]
			person2 = PP[0:P2idxlst[(k)]]
			print('-------------------------------1-------------------------',len(person1), len(person2))
			start_point = (20 * k) + 1
			end_point =  (20 * k) + 20  					
		elif k>=1 and k<lastvalue:
			print('kk',k, P1idxlst[(k-1)],P1idxlst[(k)])
			person1 = P[P1idxlst[(k-1)]:P1idxlst[(k)]]
			person2 = PP[P2idxlst[(k-1)]:P2idxlst[(k)]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))
			start_point = (20 * k) + 1
			end_point =  (20 * k) + 20 						
		elif k == lastvalue:
			print('lastvalue',P1idxlst[lastvalue-1])
			person1 = P[:P1idxlst[lastvalue-1]]
			person2 = PP[:P2idxlst[lastvalue-1]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))
			start_point = 1
			end_point =  (20 * k) 


		# person1 = P
		# person2 = PP
		Id_1 = person1
		Id_2 = person2			
				

		ID1_Engage=[]
		ID2_Engage=[]
		ID3_Engage=[]
		AED_B1 = []
		AED_B2 = []
		LENGTH = min(len(Id_1),len(Id_2))
		frm_count = LENGTH
		print('LENGTH', len(Id_1),len(Id_2))


		for frame in range(2, LENGTH):
			if frame < LENGTH-1:
				Past_11 = Id_1[frame]
				Past_21 = Id_2[frame]
				Present_11 = Id_1[frame+1]
				Present_21 = Id_2[frame+1]
				
				bx = round(int(Past_11[1]))
				by = round(int(Past_11[2]))
				bw = round(int(Past_11[3]))
				bh = round(int(Past_11[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)				
				#print(X1Y1)
				Past_1 = [Actx, Acty]
				bx1 = round(int(Past_21[1]))
				by1 = round(int(Past_21[2]))
				bw1 = round(int(Past_21[3]))
				bh1 = round(int(Past_21[4]))	
				Actx1=round((bx1+bw1)/2)
				Acty1=round((by1+bh1)/2)
				Past_2 = [Actx1, Acty1]

				bx2 = round(int(Present_11[1]))
				by2 = round(int(Present_11[2]))
				bw2 = round(int(Present_11[3]))
				bh2 = round(int(Present_11[4]))	
				Actx2=round((bx2+bw2)/2)
				Acty2=round((by2+bh2)/2)
				Present_1 = [Actx2, Acty2]

				bx3 = round(int(Present_21[1]))
				by3 = round(int(Present_21[2]))
				bw3 = round(int(Present_21[3]))
				bh3 = round(int(Present_21[4]))	
				Actx3=round((bx3+bw3)/2)
				Acty3=round((by3+bh3)/2)
				Present_2 = [Actx3, Acty3]


							
				#print('Past_1,Past_2',Past_1,Past_2)
				if len(Past_1) and len(Past_2) == 0 and len(Present_1) and len(Present_2) == 0:
					print('Detection miss')

				elif len(Past_1) and len(Past_2) != 0 and len(Present_1) and len(Present_2) != 0:	# ID1 to Other IDs    
					X_1_1 = abs(Past_1[0]-Present_1[0])#X
					Y_1_1 = abs(Past_1[1]-Present_1[1])#Y
					X_1_2 = abs(Past_1[0]-Present_2[0])#X
					Y_1_2 = abs(Past_1[1]-Present_2[1])#Y

					#print( X_1_1, Y_1_1, X_1_2, Y_1_2, X_1_3, Y_1_3)

					# ID2 to Other IDs
					X_2_1 = abs(Past_2[0]-Present_1[0])#X
					Y_2_1 = abs(Past_2[1]-Present_1[1])#Y
					X_2_2 = abs(Past_2[0]-Present_2[0])#X
					Y_2_2 = abs(Past_2[1]-Present_2[1])#Y
					
					
					EN1X = abs (Present_1[0] - Present_2[0])
					EN1Y = abs (Present_1[1] - Present_2[1])


					#print( X_2_1, Y_2_1, X_2_2, Y_2_2, X_2_3, Y_2_3)
					#print('X_1_1 + Y_1_1',X_1_1 + Y_1_1, 'X_1_2 + Y_1_2', X_1_2 + Y_1_2 )
					#print('X_2_2 + Y_2_2',X_2_2 + Y_2_2, 'X_2_1 + Y_2_1', X_2_1 + Y_2_1 )


					# #first ID
					if (X_1_1 + Y_1_1) <= 30 and (X_1_2 + Y_1_2) > (EN1X+EN1Y):		
						ID1_Engage.append(1)
					else:
						ID1_Engage.append(0)

					if  (X_2_2 + Y_2_2) <= 30 and (X_2_1 + Y_2_1) > (EN1X+EN1Y) :
						ID2_Engage.append(1)
					else:
						ID2_Engage.append(0)
						
					if  ((X_1_1 + Y_1_1) <= 30 and (X_1_2 + Y_1_2) > (EN1X+EN1Y)) and ((X_2_2 + Y_2_2) <= 10 and (X_2_1 + Y_2_1) > (EN1X+EN1Y)) :
						ID3_Engage.append(1)
					else:
						ID3_Engage.append(0)			
					
							
		FrSeMi = int(60)# Per second overlap  
		NumberofEngagement = round(int(LENGTH)/FrSeMi)
		#print('ID1_Engage',len(ID1_Engage), len(ID2_Engage))

		Backup1=ID1_Engage
		Backup2=ID2_Engage
		Backup3=ID3_Engage
		Totalenge = []
		Combined = []
		l = 0
		m = 0
		o = 0

		# we are trying for a per frame data, taking culmination of 70 frame data. Its an overlapping window  
		for kk in range(NumberofEngagement-1):
			ED_Boxer_1 = ID1_Engage[(kk*FrSeMi)+1:(kk+1)*FrSeMi]
			Act_ED_B1 = sum(ED_Boxer_1)
			#print('Act_ED_B1',Act_ED_B1)	
			if Act_ED_B1 > 30:
				Backup1[(l*FrSeMi):(l+1)*FrSeMi] = [1]*FrSeMi
			else:
				Backup1[(l*FrSeMi):(l+1)*FrSeMi] = [0]*FrSeMi
			l += 1
		#print('Backup1',len(Backup1))

		for q in range(NumberofEngagement-1):

			ED_Boxer_2 = ID2_Engage[(q*FrSeMi)+1:(q+1)*FrSeMi]
			Act_ED_B2 = sum(ED_Boxer_2)
			#print('Act_ED_B2',Act_ED_B2)	
			if Act_ED_B2 > 30:
				Backup2[(m*FrSeMi):(m+1)*FrSeMi] = [1]*FrSeMi
			else:	
				Backup2[(m*FrSeMi):(m+1)*FrSeMi] = [0]*FrSeMi
			m += 1
		#print('Backup2',len(Backup2))

		#Totalenge = sum([Backup1], [Backup2])


		for i in range(len(Backup1)):
			Totalenge.append(Backup1[i] + Backup2[i])
			
		for i in range(len(Totalenge)):
			if Totalenge[i] >= 1:
				Combined.append(1)
			else:
				Combined.append(0)		
				
		#print('Totalenge',len(Totalenge), len(Combined))
		#for o in range(NumberofEngagement-1):
		#	ED_Boxer_3 = ID3_Engage[(o*FrSeMi)+1:(o+1)*FrSeMi]
		#	Act_ED_B3 = sum(ED_Boxer_3)	
		#	print('Act_ED_B3',Act_ED_B3)
		#	if Act_ED_B3 > 35:
		#		Backup3[(o*FrSeMi):(o+1)*FrSeMi] = [1]*FrSeMi
		#	else:	
		#		Backup3[(o*FrSeMi):(o+1)*FrSeMi] = [0]*FrSeMi
		#	o += 1
		#print('Backup3',len(Backup3))
		#start_point = (20 * k) + 1
		#end_point =  (20 * k) + 20  


		#print(len(ID1_Engage),len(ID2_Engage) )       
		time_frame = np.linspace(0,frm_count/FrSeMi,frm_count)
		time_diff = (end_point - start_point)*FrSeMi
		#print('time_frame,time_diff ', time_frame,time_diff )
		if time_diff > frm_count:
			time_diff = frm_count
		#print(len(time_frame), len(ID1_Engage))	
		time_frame1 = np.linspace(start_point,end_point,len(ID1_Engage)) 
		time_frame2 = np.linspace(start_point,end_point,len(ID2_Engage))
		time_frame3 = np.linspace(start_point,end_point,len(ID2_Engage))
		#time_frame1 = np.linspace(0,len(ID1_Engage)/FrSeMi,len(ID1_Engage)) 
		#time_frame2 = np.linspace(0,len(ID2_Engage)/FrSeMi,len(ID2_Engage))
		#time_frame3 = np.linspace(0,len(ID2_Engage)/FrSeMi,len(ID2_Engage))
		#print('time_frame1',len(Backup1), len(time_frame1))
		print('--------------------+++++++++++++++++++++',k, start_point, end_point)

		#engage_ratio = round(engage_count[start_point*70:end_point*70].count(1)/(time_diff), 2)
		#plt.plot(time_frame1,ID1_Engage, color = 'blue')
		#plt.plot(time_frame2,ID2_Engage, color = 'red')
		fig=plt.figure(figsize=(20, 20))
		ax1=fig.add_subplot(311)
		# ax1.xaxis.get_label().set_fontsize(20)
		# ax1.tick_params(axis='x', labelsize=35)
		#fig.subplots_adjust(left=0,right=1,bottom=0,top=1) #R1 T1
		#plt.plot(time_frame1,Backup1, color = 'blue', linewidth = 5)
		plt.bar(time_frame1,Backup1,color=COLOR1)
		plt.xlim(start_point,end_point)
		plt.xticks([])
		plt.yticks([])
		#plt.xlabel('Time in secs', fontsize=30)
		# plt.ylabel('Engage/Disengage',fontsize=30)
		# plt.yticks(fontweight='bold')


		ax2=fig.add_subplot(312)
		#plt.plot(time_frame2,Backup2, color = 'red', linewidth = 5)
		plt.bar(time_frame2,Backup2,color=COLOR2)
		# ax2.xaxis.get_label().set_fontsize(20)
		# ax2.tick_params(axis='x', labelsize=35)
		#plt.plot(time_frame1,AED_B1, color = 'blue')
		#plt.plot(time_frame2,AED_B2, color = 'red')
		#plt.plot(time_frame,dist_engage, color = 'green')
		#plt.legend([P1_id,P2_id,'Nearby'])
		#plt.legend(['Player1','Player2'])
		plt.xlim(start_point,end_point)
		#plt.xlabel('Time in secs', fontsize=30)
		plt.yticks([])			
		plt.ylabel('Engage/Disengage', fontsize=100)
		plt.yticks(fontweight='bold')
		plt.xticks([])
		#plt.text(0.9*end_point , 0.8, 'enagement_ratio = ' + str(engage_ratio), horizontalalignment='center',verticalalignment='center', fontsize=10)
			
			
		ax3=fig.add_subplot(313)
		#plt.plot(time_frame2,Backup2, color = 'red', linewidth = 5)
		plt.bar(time_frame3,Combined,color='blueviolet')
		# ax3.xaxis.get_label().set_fontsize(20)
		ax3.tick_params(axis='x', labelsize=50)
		ax3.set_xlabel('Time in Seconds', fontweight='bold', fontsize=80)
		#plt.plot(time_frame1,AED_B1, color = 'blue')
		#plt.plot(time_frame2,AED_B2, color = 'red')
		#plt.plot(time_frame,dist_engage, color = 'green')
		#plt.legend([P1_id,P2_id,'Nearby'])
		#plt.legend(['Player1','Player2'])
		plt.xlim(start_point,end_point)
		plt.xlabel('Time in secs', fontsize=100)
		plt.xticks(fontweight='bold')
		plt.yticks([])		
		# plt.ylabel('Engage/Disengage', fontsize=30)
		# plt.yticks(fontweight='bold')
		#plt.text(0.9*end_point , 0.8, 'enagement_ratio = ' + str(engage_ratio), horizontalalignment='center',verticalalignment='center', fontsize=10)
		#plt.savefig('ED.tiff', dpi = 300)
		plt.savefig(EngageDisengagepath + '/'+ 'Engage-Disengage' + str(k) +'.tiff', dpi = 300)    
	
	return 1


def hotspot(MD1, MD2):
	ID1 = MD1
	ID2 = MD2	
	G = 50 
	# P = []
	# PP = []
	# TimeLimit = 20    

	# for insideframesnumber in (ID1[0].keys()):
	# 	anboxdeta = ID1[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		P.append(anboxdeta)
	# 		filename = 1
	# 	else:
	# 		P.append([])
	# 		continue
	# for insideframesnumber in (ID2[0].keys()):
	# 	anboxdeta = ID2[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		PP.append(anboxdeta)
	# 		filename = 2
	# 	else:
	# 		PP.append([])
	# 		continue

	# FPS = 60
	# T = int((len(ID1[0].keys())))
	
	
	# #dividing the data into 20 sec 
	# print('-------------------------------------------------------------------',len(PP),len(P))
	
	# DataLimit = FPS*TimeLimit	
	# Twentysecdata = round(min(len(PP),len(P))/(1200))
	# print('Twentysecdata',Twentysecdata)
	# for k in range(1):#(Twentysecdata+1):# individual 20 sec and 1 3 minute
	# 	if k <= Twentysecdata-1:
	# 		person1 = P[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		person2 = PP[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		print(len(person1),len(person2),(DataLimit*k)+1,DataLimit*(k+1))
	# 		print('114',k)
	# 	elif k == Twentysecdata:
	# 		person1 = P[:(DataLimit*(k+1))+1]
	# 		person2 = PP[:(DataLimit*(k+1))+1]
	# 		print(len(person1),len(person2),(DataLimit*(k+1))+1)
	# 		print('224',k)
	# 	elif k == Twentysecdata+1:
	# 		person1 = P
	# 		person2 = PP

	P = []
	PP = []
	# PP = []
	PF1 = []
	PF2 = []

	for insideframesnumber in (ID1[0].keys()):
		anboxdeta = ID1[0][insideframesnumber][0]
		if len(anboxdeta):
			P.append(anboxdeta)
			filename = 1
			PF1.append(anboxdeta[0])
		else:
			#P.append([])
			continue

	for insideframesnumber in (ID2[0].keys()):
		anboxdeta1 = ID2[0][insideframesnumber][0]
		if len(anboxdeta1):
			PP.append(anboxdeta1)
			filename = 2
			PF2.append(anboxdeta1[0])
		else:
			#P.append([])
			continue

	print('F1111',max(PF1))

	print('F1111',max(PF1))

	FPS = 60
	TimeLimit = 20
	DataLimit = FPS*TimeLimit
	print('DataLimit',DataLimit)
	Twentysecdata = round((max(PF1))/(1200))
	print('Twentysecdata',Twentysecdata)

	ar1 = np.array(PF1)
	ar2 = np.array(PF2)

	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		print('idx',idx)
		return idx, array[idx]

	P1idxlst = []
	P2idxlst = []
	for value1 in range(1,Twentysecdata):
		idx1, I1 = (find_nearest(ar1, value=DataLimit* value1))
		P1idxlst.append(idx1)
		idx2, I2 = (find_nearest(ar2, value=DataLimit* value1))
		P2idxlst.append(idx2)

	print('P1idxlst',P1idxlst)
	print('P2idxlst',P2idxlst)
	lastvalue = (min(len(P1idxlst), len(P2idxlst)))
	print('lastvalue',lastvalue)

	for k in range(0, lastvalue+1):
		print('k',k)
		if k==0:
			print(k,0,P1idxlst[(k)])
			person1 = P[0:P1idxlst[(k)]]
			person2 = PP[0:P2idxlst[(k)]]
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k>=1 and k<lastvalue:
			print('kk',k, P1idxlst[(k-1)],P1idxlst[(k)])
			person1 = P[P1idxlst[(k-1)]:P1idxlst[(k)]]
			person2 = PP[P2idxlst[(k-1)]:P2idxlst[(k)]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k == lastvalue:
			print('lastvalue',P1idxlst[lastvalue-1])
			person1 = P[:P1idxlst[lastvalue-1]]
			person2 = PP[:P2idxlst[lastvalue-1]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))
		#print('336',k)		
		#image = Image.open('/home/monsley/Desktop/presentation/Metrics/frame1.jpg')
		#image = plt.imread('/home/monsley/Desktop/presentation/Metrics/frame1.jpg')
		image = plt.imread(save_path_new +'/'+'frame1.jpg')
		my_dpi=300
		#person1 = P[???]	
		# Set up figure
		fig=plt.figure(figsize=(20, 10))
		ax1=fig.add_subplot(121)
		ax1.set_title('Boxer 1', fontsize=20)
		#ax1.set_xlim([0,984])
		#ax1.set_ylim([984,0])
		ax1.set_xlim([0,984])
		ax1.set_ylim([984,0])
		#plt.xlim(0, min(x))
		#plt.ylim([984, 0])	
		#ax1.set_xticks([],labelsize=10)
		# ax1.set_yticks([])
		# ax1.set_xticks([])
		#ax1.imshow(np.flipud(image))
		ax1.imshow((image))
		#image1=np.flipud(image)# to show the image
		#ax1.imshow(image1, origin='upper')
		for kl in range(len(person1)):
			POL=person1[kl]
			if len(POL) == 0:
				print('Detection miss')
			else:
				Act=POL
				bx = round(int(Act[1]))
				by = round(int(Act[2]))
				bw = round(int(Act[3]))
				bh = round(int(Act[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)				
				print('NOTEMPTY',Actx,  Acty)
				#circle = plt.Circle((Actx,Acty),5, fc='blue',ec="blue", fill=True)
				circle = plt.Circle((Actx,Acty),5, fc=C1,ec=C1, fill=True)
				plt.gca().add_patch(circle)
				#plt.axis('scaled')			
		plt.xticks([])
		plt.yticks([])
		plt.tight_layout()	
		#fig.savefig('Boxer1_hotspot.tiff',dpi=my_dpi)
		print('COMPLETED_1')
		
		#image = Image.open('/home/monsley/Desktop/presentation/Metrics/frame1.jpg')
		#image = plt.imread('/home/monsley/Desktop/presentation/Metrics/frame1.jpg')
		image = plt.imread(save_path_new +'/'+'frame1.jpg')
		myInterval=G
		my_dpi=300
		#person2 = PP
		# Set up figure
		ax2=fig.add_subplot(122)
		ax2.set_title('Boxer 2', fontsize=20)
		ax2.set_xlim([0,984])
		ax2.set_ylim([984,0])
		# ax2.set_xticks([],labelsize=10)
		# ax2.set_yticks([])
		#ax2.imshow(np.flipud(image))
		ax2.imshow((image))
		#ax2.imshow(np.flipud(image), origin='upper')	
		for kl in range(len(person2)):
			POL=person2[kl]
			#print(POL)
			#print(Actx,Acty)
			if len(POL) == 0:
				print('Detection miss')
			else:
				Act=POL
				bx = round(int(Act[1]))
				by = round(int(Act[2]))
				bw = round(int(Act[3]))
				bh = round(int(Act[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)
				print('NOTEMPTY',Actx,  Acty)					
				print('NOTEMPTY')
				circle = plt.Circle((Actx,Acty),5, fc=C2,ec=C2,fill=True)
				plt.gca().add_patch(circle)
				#plt.axis('scaled')
		g = 0              					
		plt.tight_layout()
		plt.xticks([])
		plt.yticks([])
		fig.savefig(Hotspotpath + '/'+'Complete_Hotspot' + str(k) +'.tiff',dpi=my_dpi)
		print('COMPLETED_2')
		print(k)
		
	return g


def zonemanagement(MD1, MD2):
	ID1 = MD1
	ID2 = MD2	
	print('insidezonemanagement',zonemanagement)
	image = cv2.imread(save_path_new +'/'+'frame1.jpg')
	overlay = image.copy()
	fig=plt.figure(figsize=(20, 10))

	#Zone = input('NUMBER OF ZONES')

	Zone = 8

	TNZ = int(Zone)
	Zone = TNZ
	alpha = 0.5
	Len = 890
	ReofSep= round(890/(TNZ*2))
	Start = 0
	End = 890
	print('ReofSep',ReofSep, Start, End)

	for k in range (1, TNZ+1):
		print('k',k)
		if k ==1:
			Z8_S=0+ReofSep
			Z8_E=980-ReofSep
			x, y, w, h = Z8_S, Z8_S, Z8_E, Z8_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (0, 200, 0), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
		elif k==2:
			Z7_S=Z8_S+ReofSep
			Z7_E=Z8_E-ReofSep
			x, y, w, h = Z7_S, Z7_S, Z7_E, Z7_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (200, 200, 0), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==3:
			Z6_S=Z7_S+ReofSep
			Z6_E=Z7_E-ReofSep
			x, y, w, h = Z6_S, Z6_S, Z6_E, Z6_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (0, 200, 200), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==4:
			Z5_S=Z6_S+ReofSep
			Z5_E=Z6_E-ReofSep
			x, y, w, h = Z5_S, Z5_S, Z5_E, Z5_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (200, 0, 200), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==5:
			Z4_S=Z5_S+ReofSep
			Z4_E=Z5_E-ReofSep
			x, y, w, h = Z4_S, Z4_S, Z4_E, Z4_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (100, 100, 100), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==6:
			Z3_S=Z4_S+ReofSep
			Z3_E=Z4_E-ReofSep
			x, y, w, h = Z3_S, Z3_S, Z3_E, Z3_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (100, 0, 100), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==7:
			Z2_S=Z3_S+ReofSep
			Z2_E=Z3_E-ReofSep
			x, y, w, h = Z2_S, Z2_S, Z2_E, Z2_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (100, 100, 0), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)		
		elif k==8:
			Z1_S=Z2_S+ReofSep
			Z1_E=Z2_E-ReofSep
			x, y, w, h = Z1_S, Z1_S, Z1_E, Z1_E  # Rectangle parameters
			cv2.rectangle(overlay, (x, y), (w, h), (0, 100, 100), -1)  # A filled rectangle
			image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)	
	
	print('Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E',Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E)


	TimeLimit = 20
	zone1 = 0
	zone2 = 0
	zone3 = 0
	zone4 = 0
	zone5 = 0
	zone6 = 0
	zone7 = 0
	zone8 = 0	
	P = []
	Totalnumberofzones = int(Zone)
	Szone1 = 0
	Szone2 = 0
	Szone3 = 0
	Szone4 = 0
	Szone5 = 0
	Szone6 = 0
	Szone7 = 0
	Szone8 = 0	
	PP = []
	Totalnumberofzones = int(Zone)	
	
	# for insideframesnumber in (ID1[0].keys()):
	# 	anboxdeta = ID1[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		P.append(anboxdeta)
	# 		filename = 1
	# 	else:
	# 		#P.append([])
	# 		continue
	# for insideframesnumber in (ID2[0].keys()):
	# 	anboxdeta = ID2[0][insideframesnumber][0]
	# 	if len(anboxdeta):
	# 		PP.append(anboxdeta)
	# 		filename = 2
	# 	else:
	# 		#PP.append([])
	# 		continue
	# FPS = 60
	
	
	# #dividing the data into 20 sec 
	
	# DataLimit = FPS*TimeLimit	
	# Twentysecdata = round(min(len(PP),len(P))/(1200))
	# print('Twentysecdata',Twentysecdata)
	# for k in range(Twentysecdata+1):# individual 20 sec and 1 3 minute
	# 	if k <= Twentysecdata-1:
	# 		person1 = P[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		person2 = PP[(DataLimit*k)+1:DataLimit*(k+1)]
	# 		#print(len(person1),len(person2),(DataLimit*k)+1,DataLimit*(k+1))
	# 		#print(k)
	# 	elif k == Twentysecdata:
	# 		person1 = P[:(DataLimit*(k+1))+1]
	# 		person2 = PP[:(DataLimit*(k+1))+1]
	# 		#print(len(person1),len(person2),(DataLimit*(k+1))+1)
	# 	elif k == Twentysecdata+1:
	# 		person1 = P
	# 		person2 = PP

	P = []
	PP = []
	# PP = []
	PF1 = []
	PF2 = []

	for insideframesnumber in (ID1[0].keys()):
		anboxdeta = ID1[0][insideframesnumber][0]
		if len(anboxdeta):
			P.append(anboxdeta)
			filename = 1
			PF1.append(anboxdeta[0])
		else:
			#P.append([])
			continue

	for insideframesnumber in (ID2[0].keys()):
		anboxdeta1 = ID2[0][insideframesnumber][0]
		if len(anboxdeta1):
			PP.append(anboxdeta1)
			filename = 2
			PF2.append(anboxdeta1[0])
		else:
			#P.append([])
			continue

	print('F1111',max(PF1))

	print('F1111',max(PF1))

	FPS = 60
	TimeLimit = 20
	DataLimit = FPS*TimeLimit
	print('DataLimit',DataLimit)
	Twentysecdata = round((max(PF1))/(1200))
	print('Twentysecdata',Twentysecdata)

	ar1 = np.array(PF1)
	ar2 = np.array(PF2)

	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		print('idx',idx)
		return idx, array[idx]

	P1idxlst = []
	P2idxlst = []
	for value1 in range(1,Twentysecdata):
		idx1, I1 = (find_nearest(ar1, value=DataLimit* value1))
		P1idxlst.append(idx1)
		idx2, I2 = (find_nearest(ar2, value=DataLimit* value1))
		P2idxlst.append(idx2)

	print('P1idxlst',P1idxlst)
	print('P2idxlst',P2idxlst)
	lastvalue = (min(len(P1idxlst), len(P2idxlst)))
	print('lastvalue',lastvalue)

	for k in range(0, lastvalue+1):
		print('k',k)
		if k==0:
			print(k,0,P1idxlst[(k)])
			person1 = P[0:P1idxlst[(k)]]
			person2 = PP[0:P2idxlst[(k)]]
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k>=1 and k<lastvalue:
			print('kk',k, P1idxlst[(k-1)],P1idxlst[(k)])
			person1 = P[P1idxlst[(k-1)]:P1idxlst[(k)]]
			person2 = PP[P2idxlst[(k-1)]:P2idxlst[(k)]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))		
		elif k == lastvalue:
			print('lastvalue',P1idxlst[lastvalue-1])
			person1 = P[:P1idxlst[lastvalue-1]]
			person2 = PP[:P2idxlst[lastvalue-1]]	
			print('-------------------------------1-------------------------',len(person1), len(person2))


		#person1 = P
		ind = np.arange(8)
		width = 0.3
		#person2 = PP
		# ind = np.arange(8)
		width = 0.3
		print(len(person1),len(person2))		
		for kl in range(len(person1)):
			POL=person1[kl]
			if len(POL) == 0:
				print('Detection miss')
			else:
				Act=POL
				bx = round(int(Act[1]))
				by = round(int(Act[2]))
				bw = round(int(Act[3]))
				bh = round(int(Act[4]))	
				Actx=round((bx+bw)/2)
				Acty=round((by+bh)/2)				
				
				#Actx=round(Act[0])
				#Acty=round(Act[1])
				print('NOTEMPTY')		
				if Totalnumberofzones == 4:
					ind = np.arange(4)
					if (90<=Actx<=Z8_S and 90<=Acty<=890) or (90<=Actx<=890 and 90<=Acty<=Z8_S) or (Z8_E<=Actx<=890 and 90<Acty<=890) or (90<Actx<=890 and Z8_E<=Acty<=890):#outer region
						zone8 +=1
					elif (Z8_S+1<=Actx<=Z7_S and Z8_S+1<=Acty<=Z8_E-1) or (Z8_S+1<=Actx<=Z8_E-1 and Z8_S+1<=Acty<=Z7_S) or (Z7_E<=Actx<=Z8_E-1 and Z8_S+1<Acty<=Z8_E-1) or (Z8_S+1<Actx<=Z8_E-1 and Z7_E<=Acty<=Z8_E-1):
						zone7 +=1
					elif (Z7_S+1<=Actx<=Z6_S and Z7_S+1<=Acty<=Z7_E-1) or (Z7_S+1<=Actx<=Z7_E-1 and Z7_S+1<=Acty<=Z6_S) or (Z6_E<=Actx<=Z7_E-1 and Z7_S+1<Acty<=Z7_E-1) or (Z7_S+1<Actx<=Z7_E-1 and Z6_E<=Acty<=Z7_E-1):
						zone6 +=1
					elif (Z6_S+1<=Actx<=Z5_S and Z6_S+1<=Acty<=Z6_E-1) or (Z6_S+1<=Actx<=Z6_E-1 and Z6_S+1<=Acty<=Z5_S) or (Z5_E<=Actx<=Z6_E-1 and Z6_S+1<Acty<=Z6_E-1) or (Z6_S+1<Actx<=Z6_E-1 and Z5_E<=Acty<=Z6_E-1):
						zone5 +=1
				elif Totalnumberofzones == 8:
					ind = np.arange(8)
					if (90<=Actx<=Z8_S and 90<=Acty<=890) or (90<=Actx<=890 and 90<=Acty<=Z8_S) or (Z8_E<=Actx<=890 and 90<Acty<=890) or (90<Actx<=890 and Z8_E<=Acty<=890):#outer region
						zone8 +=1
					elif (Z8_S+1<=Actx<=Z7_S and Z8_S+1<=Acty<=Z8_E-1) or (Z8_S+1<=Actx<=Z8_E-1 and Z8_S+1<=Acty<=Z7_S) or (Z7_E<=Actx<=Z8_E-1 and Z8_S+1<Acty<=Z8_E-1) or (Z8_S+1<Actx<=Z8_E-1 and Z7_E<=Acty<=Z8_E-1):
						zone7 +=1
					elif (Z7_S+1<=Actx<=Z6_S and Z7_S+1<=Acty<=Z7_E-1) or (Z7_S+1<=Actx<=Z7_E-1 and Z7_S+1<=Acty<=Z6_S) or (Z6_E<=Actx<=Z7_E-1 and Z7_S+1<Acty<=Z7_E-1) or (Z7_S+1<Actx<=Z7_E-1 and Z6_E<=Acty<=Z7_E-1):
						zone6 +=1
					elif (Z6_S+1<=Actx<=Z5_S and Z6_S+1<=Acty<=Z6_E-1) or (Z6_S+1<=Actx<=Z6_E-1 and Z6_S+1<=Acty<=Z5_S) or (Z5_E<=Actx<=Z6_E-1 and Z6_S+1<Acty<=Z6_E-1) or (Z6_S+1<Actx<=Z6_E-1 and Z5_E<=Acty<=Z6_E-1):
						zone5 +=1
					elif (Z5_S+1<=Actx<=Z4_S and Z5_S+1<=Acty<=Z4_E-1) or (Z5_S+1<=Actx<=Z5_E-1 and Z5_S+1<=Acty<=Z4_S) or (Z4_E<=Actx<=Z5_E-1 and Z5_S+1<Acty<=Z5_E-1) or (Z5_S+1<Actx<=Z5_E-1 and Z4_E<=Acty<=Z5_E-1):
						zone4 +=1
					elif (Z4_S+1<=Actx<=Z3_S and Z4_S+1<=Acty<=Z3_E-1) or (Z4_S+1<=Actx<=Z4_E-1 and Z4_S+1<=Acty<=Z3_S) or (Z3_E<=Actx<=Z4_E-1 and Z4_S+1<Acty<=Z4_E-1) or (Z4_S+1<Actx<=Z4_E-1 and Z3_E<=Acty<=Z4_E-1):
						zone3 +=1
					elif (Z3_S+1<=Actx<=Z2_S and Z3_S+1<=Acty<=Z2_E-1) or (Z3_S+1<=Actx<=Z3_E-1 and Z3_S+1<=Acty<=Z2_S) or (Z2_E<=Actx<=Z3_E-1 and Z3_S+1<Acty<=Z3_E-1) or (Z3_S+1<Actx<=Z3_E-1 and Z2_E<=Acty<=Z3_E-1):
						zone2 +=1															
					elif (Z2_S+1<=Actx<Z1_S and Z2_S+1<=Acty<Z1_E):
						zone1 +=1
		# if Totalnumberofzones == 4:
		# 	X1 = [zone5, zone6, zone7, zone8]
		# 	X=['zone1', 'zone2', 'zone3','zone4']
		# 	print(X,X1)
		# elif Totalnumberofzones == 8:
		# 	X1 = [zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8]
		# 	X=['zone1', 'zone2', 'zone3','zone4','zone5', 'zone6', 'zone7','zone8']
		# 	print(X,X1)
		# #plt.bar(ind,X1,width, color=(0,0,1))
		# plt.bar(ind,X1,width, color=COLOR1)
		# zone1 = 0
		# zone2 = 0
		# zone3 = 0
		# zone4 = 0
		# zone5 = 0
		# zone6 = 0
		# zone7 = 0
		# zone8 = 0	


		#plt.show()
		#plt.savefig('Boxer1_Zonemanagment.tiff',dpi = 300)	
		for kl in range(len(person2)):
			POL1=person2[kl]
			if len(POL1) == 0:
				print('Detection miss')
			else:
				Act=POL1
				bx1 = round(int(Act[1]))
				by1 = round(int(Act[2]))
				bw1 = round(int(Act[3]))
				bh1 = round(int(Act[4]))	
				Actx=round((bx1+bw1)/2)
				Acty=round((by1+bh1)/2)	
				#Actx=round(Act[0])
				#Acty=round(Act[1])
				print('NOTEMPTY')
				if Totalnumberofzones == 4:
					ind = np.arange(4)
					if (90<=Actx<=Z8_S and 90<=Acty<=890) or (90<=Actx<=890 and 90<=Acty<=Z8_S) or (Z8_E<=Actx<=890 and 90<Acty<=890) or (90<Actx<=890 and Z8_E<=Acty<=890):#outer region
						Szone8 +=1
					elif (Z8_S+1<=Actx<=Z7_S and Z8_S+1<=Acty<=Z8_E-1) or (Z8_S+1<=Actx<=Z8_E-1 and Z8_S+1<=Acty<=Z7_S) or (Z7_E<=Actx<=Z8_E-1 and Z8_S+1<Acty<=Z8_E-1) or (Z8_S+1<Actx<=Z8_E-1 and Z7_E<=Acty<=Z8_E-1):
						Szone7 +=1
					elif (Z7_S+1<=Actx<=Z6_S and Z7_S+1<=Acty<=Z7_E-1) or (Z7_S+1<=Actx<=Z7_E-1 and Z7_S+1<=Acty<=Z6_S) or (Z6_E<=Actx<=Z7_E-1 and Z7_S+1<Acty<=Z7_E-1) or (Z7_S+1<Actx<=Z7_E-1 and Z6_E<=Acty<=Z7_E-1):
						Szone6 +=1
					elif (Z6_S+1<=Actx<=Z5_S and Z6_S+1<=Acty<=Z6_E-1) or (Z6_S+1<=Actx<=Z6_E-1 and Z6_S+1<=Acty<=Z5_S) or (Z5_E<=Actx<=Z6_E-1 and Z6_S+1<Acty<=Z6_E-1) or (Z6_S+1<Actx<=Z6_E-1 and Z5_E<=Acty<=Z6_E-1):
						Szone5 +=1
				elif Totalnumberofzones == 8:
					ind = np.arange(8)
					if (90<=Actx<=Z8_S and 90<=Acty<=890) or (90<=Actx<=890 and 90<=Acty<=Z8_S) or (Z8_E<=Actx<=890 and 90<Acty<=890) or (90<Actx<=890 and Z8_E<=Acty<=890):#outer region
						Szone8 +=1
					elif (Z8_S+1<=Actx<=Z7_S and Z8_S+1<=Acty<=Z8_E-1) or (Z8_S+1<=Actx<=Z8_E-1 and Z8_S+1<=Acty<=Z7_S) or (Z7_E<=Actx<=Z8_E-1 and Z8_S+1<Acty<=Z8_E-1) or (Z8_S+1<Actx<=Z8_E-1 and Z7_E<=Acty<=Z8_E-1):
						Szone7 +=1
					elif (Z7_S+1<=Actx<=Z6_S and Z7_S+1<=Acty<=Z7_E-1) or (Z7_S+1<=Actx<=Z7_E-1 and Z7_S+1<=Acty<=Z6_S) or (Z6_E<=Actx<=Z7_E-1 and Z7_S+1<Acty<=Z7_E-1) or (Z7_S+1<Actx<=Z7_E-1 and Z6_E<=Acty<=Z7_E-1):
						Szone6 +=1
					elif (Z6_S+1<=Actx<=Z5_S and Z6_S+1<=Acty<=Z6_E-1) or (Z6_S+1<=Actx<=Z6_E-1 and Z6_S+1<=Acty<=Z5_S) or (Z5_E<=Actx<=Z6_E-1 and Z6_S+1<Acty<=Z6_E-1) or (Z6_S+1<Actx<=Z6_E-1 and Z5_E<=Acty<=Z6_E-1):
						Szone5 +=1
					elif (Z5_S+1<=Actx<=Z4_S and Z5_S+1<=Acty<=Z4_E-1) or (Z5_S+1<=Actx<=Z5_E-1 and Z5_S+1<=Acty<=Z4_S) or (Z4_E<=Actx<=Z5_E-1 and Z5_S+1<Acty<=Z5_E-1) or (Z5_S+1<Actx<=Z5_E-1 and Z4_E<=Acty<=Z5_E-1):
						Szone4 +=1
					elif (Z4_S+1<=Actx<=Z3_S and Z4_S+1<=Acty<=Z3_E-1) or (Z4_S+1<=Actx<=Z4_E-1 and Z4_S+1<=Acty<=Z3_S) or (Z3_E<=Actx<=Z4_E-1 and Z4_S+1<Acty<=Z4_E-1) or (Z4_S+1<Actx<=Z4_E-1 and Z3_E<=Acty<=Z4_E-1):
						Szone3 +=1
					elif (Z3_S+1<=Actx<=Z2_S and Z3_S+1<=Acty<=Z2_E-1) or (Z3_S+1<=Actx<=Z3_E-1 and Z3_S+1<=Acty<=Z2_S) or (Z2_E<=Actx<=Z3_E-1 and Z3_S+1<Acty<=Z3_E-1) or (Z3_S+1<Actx<=Z3_E-1 and Z2_E<=Acty<=Z3_E-1):
						Szone2 +=1															
					elif (Z2_S+1<=Actx<Z1_S and Z2_S+1<=Acty<Z1_E):
						Szone1 +=1	

		Maxzone12 = max(zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, Szone1, Szone2, Szone3, Szone4, Szone5, Szone6, Szone7, Szone8)
		print('Maxzone1',zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8)
		if Totalnumberofzones == 4:
			X1 = [zone5, zone6, zone7, zone8]
			X=['zone1', 'zone2', 'zone3','zone4']
			print(X,X1)
		elif Totalnumberofzones == 8:
			X1 = [zone1/Maxzone12, zone2/Maxzone12, zone3/Maxzone12, zone4/Maxzone12, zone5/Maxzone12, zone6/Maxzone12, zone7/Maxzone12, zone8/Maxzone12]
			X=['zone1', 'zone2', 'zone3','zone4','zone5', 'zone6', 'zone7','zone8']
			print(X,X1)
		#plt.bar(ind,X1,width, color=(0,0,1))
		plt.bar(ind,X1,width, color=COLOR1)
		zone1 = 0
		zone2 = 0
		zone3 = 0
		zone4 = 0
		zone5 = 0
		zone6 = 0
		zone7 = 0
		zone8 = 0	

		#X1 = [zone1/len(person1), zone2/len(person1), zone3/len(person1),zone4/len(person1)]
		Maxzone2 = max(Szone1, Szone2, Szone3, Szone4, Szone5, Szone6, Szone7, Szone8)
		if Totalnumberofzones == 4:
			X2 = [Szone5, Szone6, Szone7, Szone8]
			X=['zone1', 'zone2', 'zone3','zone4']
			print(X,X2)
		elif Totalnumberofzones == 8:
			X2 = [Szone1/Maxzone12, Szone2/Maxzone12, Szone3/Maxzone12, Szone4/Maxzone12, Szone5/Maxzone12, Szone6/Maxzone12, Szone7/Maxzone12, Szone8/Maxzone12]
			X=['zone1', 'zone2', 'zone3','zone4','zone5', 'zone6', 'zone7','zone8']
			print(X,X2)

		plt.bar(ind+width,X2,width,color=COLOR2)
		plt.xticks(ind + width / 2, ('Z1', 'Z2', 'Z3','Z4','Z5', 'Z6', 'Z7','Z8'),fontweight='bold', fontsize='40', horizontalalignment='center')
		plt.yticks(fontsize=40, fontweight='bold')
		plt.savefig(Zonemanagmentpath +'/'+'Zonemanagment' + str(k) +'.tiff', dpi = 300)
		# fig.tight_layout()
		# plt.subplots_adjust(0,0,1,1,0,0)		

		Szone1 = 0
		Szone2 = 0
		Szone3 = 0
		Szone4 = 0
		Szone5 = 0
		Szone6 = 0
		Szone7 = 0
		Szone8 = 0
		# fignumber +=1
		# print('FIGURELENGHT', fignumber)
		# # lengoffile += 1 		
		# #print(zone1, zone2, zone3, zone4, Pzone1, Pzone2, Pzone3,Pzone4)

		# #X1 = [zone1/len(person1), zone2/len(person1), zone3/len(person1),zone4/len(person1)]
		# if Totalnumberofzones == 4:
		# 	X2 = [Szone5, Szone6, Szone7, Szone8]
		# 	X=['zone1', 'zone2', 'zone3','zone4']
		# 	print(X,X2)
		# elif Totalnumberofzones == 8:
		# 	X2 = [Szone1, Szone2, Szone3, Szone4, Szone5, Szone6, Szone7, Szone8]
		# 	X=['zone1', 'zone2', 'zone3','zone4','zone5', 'zone6', 'zone7','zone8']
		# 	print(X,X2)

		# #plt.bar(ind+width,X2,width,color=(1,0,0))
		# plt.bar(ind+width,X2,width,color=COLOR2)
		# #plt.show()
		
		# # if Totalnumberofzones == 4:
		# # 	plt.xticks(ind + width / 2, ('zone1', 'zone2', 'zone3','zone4',))
		# # 	plt.savefig('4_Zonemanagment.tiff', dpi = 300)
		# # if Totalnumberofzones == 8:
		# plt.xticks(ind + width / 2, ('zone1', 'zone2', 'zone3','zone4','zone5', 'zone6', 'zone7','zone8'),fontweight='bold', fontsize='40', horizontalalignment='center')
		# plt.yticks(fontsize=20, fontweight='bold')
		# plt.savefig(Zonemanagmentpath +'/'+'Zonemanagment' + str(k) +'.tiff', dpi = 300)
		# Szone1 = 0
		# Szone2 = 0
		# Szone3 = 0
		# Szone4 = 0
		# Szone5 = 0
		# Szone6 = 0
		# Szone7 = 0
		# Szone8 = 0		
		#print(zone1, zone2, zone3, zone4, Pzone1, Pzone2, Pzone3,Pzone4)
	print('Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E',Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E)
				
	# return Totalnumberofzones
	print('ReofSep',ReofSep, Start, End)
	print('Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E',Z1_S,Z2_S,Z3_S,Z4_S,Z5_S,Z5_S,Z6_S,Z7_S,Z8_S, Z1_E,Z2_E,Z3_E,Z4_E,Z5_E,Z5_E,Z6_E,Z7_E,Z8_E)
	
	return Totalnumberofzones
	

directionalhistogram (MD1, MD2)
ENGDIS (MD1, MD2)
hotspot(MD1, MD2)
zonemanagement(MD1, MD2)


