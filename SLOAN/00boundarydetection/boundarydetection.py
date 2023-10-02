import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import wasserstein_distance
from datetime import datetime
import pathlib, glob


model = torch.hub.load("ultralytics/yolov5", "yolov5l")

def rectangular_mask(img):
    img[0:60,:]=0
    img[-60:-1,:]=0
    img[:,0:60]=0
    img[:,-60:-1]=0
    return img


def circular_mask(img, radius, coordinates):
    #coordinates will be list of tuples
    color = (0, 0, 0)
    thickness = -1
    for i in range(len(coordinates)):
        img = cv2.circle(img, coordinates[i], radius, color, thickness)
    return img

def preprocessing(img):
    img = rectangular_mask(img)
    coordinates = [(90, 90),(90, 894),(894, 90), (894, 894)]
    radius = 50
    img = circular_mask(img,radius, coordinates)
    return img

def histogram_segmented_objects(img, masks):
    hist_all_objects = []
    for i in range(len(masks)):
        seg_pixels = img[np.where(np.squeeze(masks[i]==True))]
        hist_three_planes = []
        for ch in range(3):
            hist, bin_edges = np.histogram(seg_pixels[:,ch],256)
            hist_three_planes.append(hist)
        hist_all_objects.append(hist_three_planes) #no of object x no of planes x 256
        
    return hist_all_objects

def relative_distance(input_box):
    centroid = []
    for i in range(len(input_box)):
        centre = ((input_box[i][0]+input_box[i][2])/2,(input_box[i][1]+input_box[i][3])/2)
        centroid.append(centre)
    centroid = np.array(centroid)
    d1 = np.linalg.norm(centroid[0] - centroid[1])
    d2 = np.linalg.norm(centroid[0] - centroid[2])
    d3 = np.linalg.norm(centroid[1] - centroid[2])
    dist = (d1+d2+d3)/3
    return dist

def bounding_boxes_person_class(coordinate_list):
    input_box_all = []
    for i in range(len(coordinate_list)):
        input_box = np.array(coordinate_list[i])
        input_box_all.append(input_box)
    return input_box_all

def timestamp(frame_count, fps):
    timestamp_seconds = frame_count / fps
    hours = int(timestamp_seconds // 3600)
    minutes = int((timestamp_seconds % 3600) // 60)
    seconds = int(timestamp_seconds % 60)
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return timestamp


from openpyxl import Workbook
workbook = Workbook()
sheet = workbook.active


save_path_new = str(pathlib.Path(__file__).parent.resolve())
#print('save_path_new',save_path_new)

inputpath = save_path_new+'/'+'input' # the folder with the IDS
#print('inputpath',inputpath)

nameofvideopath = inputpath
isExist = os.path.exists(nameofvideopath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(nameofvideopath)
   print("The video directory is created!")

CWD = str(pathlib.Path(__file__).parent.parent)
#print('CWD',CWD)

outputpath = CWD + '/' +  '0SegmentedVideos' + '/' + 'segmentedbouts' + '/'
#print('outputpath',outputpath)

nameofoutputpath = outputpath
isExist = os.path.exists(nameofoutputpath)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(nameofoutputpath)
   print("The output directory is created!")


#Load the video
filelist = glob.glob(inputpath+'/'+"*.avi")  # Get all files in the current folder
print(filelist)
for file in filelist:
    videofilename = file
    nameofvideo = os.path.split(file)[-1]
    nameoffolder,ext = os.path.splitext(nameofvideo)

#print('videofilename',videofilename,nameofvideo,nameoffolder)

input_file = videofilename
output_dir = outputpath

print('input_file',input_file)
print('output_dir',output_dir)



import re
words = re.split('/|\.', input_file)

output_prefix = words[-2]
# min_clip_duration = 60  #in seconds
mid_clip_duration = 150
max_clip_duration = 240
max_wait_dist_duration = 1.5 # 1 second
max_wait_loc_duration = 1

video = cv2.VideoCapture(input_file)  
video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)  
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
 
# duration = min_clip_duration * fps  
mid_duration = mid_clip_duration * fps
max_duration = max_clip_duration * fps

output_template = output_dir + output_prefix + "_output_clip" + "_{}.avi"  

frame_count_time = 0

frame_counter = 0  

clip_counter = 1
max_wait_0 = 7*fps
max_wait_1 = 3*fps #350
max_wait_2 = 4*fps #210
max_wait_4 = 3*fps #210
max_wait_5 = 3*fps #490
wait_for_next_clip = 0

max_wait_3 = 2*fps #175 #Waiting no of frames to detect start of a bout

wait_0 = 0
wait_1 = 0
wait_2 = 0
wait_4 = 0
wait_5 = 0

wait_loc = 0
wait_dist = 0

w_3 = 0

datetime_format = "%H:%M:%S"

if video.isOpened() == False:
    print("Error opening video file")


for fr_num in range(video_length):
    
    success, frame = video.read()
    # frm = frame.copy()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = preprocessing(img)
    if success:
        frm = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocessing(img)
        results = model(img)
        res =results.pandas().xyxy[0]
        res_list= res.values.tolist()
        
        cls = res['class']
        confd = res['confidence']
        
        person_count = 0
        
        cor_list = []
        for i in range(len(cls)):
            if cls[i]==0 and confd[i]>0.4:
                if np.sqrt((res_list[i][2]-res_list[i][0])**2 + (res_list[i][3]-res_list[i][1])**2) > 115:
                    cor =res_list[i][0:4]
                    cor_list.append(cor)
                    person_count += 1
            elif cls[i]==33 and confd[i]>0.7:
                if np.sqrt((res_list[i][2]-res_list[i][0])**2 + (res_list[i][3]-res_list[i][1])**2) > 115:
                    cor =res_list[i][0:4]
                    cor_list.append(cor)
                    person_count += 1
        # print('Person:', person_count)          

    

        if frame_counter < max_wait_3 and person_count == 3:
            # conf_frame_counter += 1
            input_box = bounding_boxes_person_class(cor_list)
            dist = relative_distance(input_box)
            if dist > 300:
                frame_counter -= 1
#                 print(frame_counter, dist)
            else:
                # print("frame count:",frame_counter)
                wait_for_next_clip +=1
                # wait_1 = 0
                # wait_2 = 0
                # wait_4 = 0
                # wait_other = 0
                # print( wait_for_next_clip, frame_counter)

            if wait_for_next_clip == (max_wait_3 * 0.8): # 80%
                # conf_score = wait_for_next_clip / conf_frame_counter
                
                # conf_frame_counter = 0
                wait_for_next_clip = 0

                
                start_time = timestamp(frame_count_time, fps)
                print(f"Start of a Bout number {clip_counter} Detected, started at {start_time}")
                cell_address1 = f"B{clip_counter}"
                sheet[cell_address1] = start_time
                datetime1 = datetime.strptime(start_time, datetime_format)

                output_file = output_template.format(clip_counter)
                output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))

                output_file_excel_template = output_prefix + "output_clip" + "_{}.avi"  
                output_file_excel = output_file_excel_template.format(clip_counter)
                cell_address2 = f"A{clip_counter}"
                sheet[cell_address2] = output_file_excel
                # output.write(frm)
                frame_counter = max_wait_3
                # clip_counter += 1
        
        elif frame_counter == max_wait_3 and wait_for_next_clip < (max_wait_3 * 0.8):
            frame_counter = 0  
            # print("Reset frame counter")      

        if frame_counter >= max_wait_3: 
            if person_count == 3:
                output.write(frm)
                w_3 += 1
                if w_3 == (fps // 2):
                    wait_0 = 0
                    wait_1 = 0
                    wait_2 = 0
                    wait_4 = 0
                    wait_5 = 0
                    w_3 = 0
                
            # print("Wait frame:",wait)

            elif person_count == 2:
                input_box_all = bounding_boxes_person_class(cor_list)
                b21 = np.sqrt((input_box_all[0][2]-input_box_all[0][0])**2 + (input_box_all[0][3]-input_box_all[0][1])**2) 
                b22 = np.sqrt((input_box_all[1][2]-input_box_all[1][0])**2 + (input_box_all[1][3]-input_box_all[1][1])**2)
                if b21 <= 150 and b22 <= 150:
                    wait_2 += 1
                output.write(frm)
                
#                 if wait_2 > 200:
#                     print("Wait frame 2:",wait_2)
            elif person_count == 4:
                input_box_all = bounding_boxes_person_class(cor_list)
                b41 = np.sqrt((input_box_all[0][2]-input_box_all[0][0])**2 + (input_box_all[0][3]-input_box_all[0][1])**2)
                b42 = np.sqrt((input_box_all[1][2]-input_box_all[1][0])**2 + (input_box_all[1][3]-input_box_all[1][1])**2)
                b43 = np.sqrt((input_box_all[2][2]-input_box_all[2][0])**2 + (input_box_all[2][3]-input_box_all[2][1])**2)
                b44 = np.sqrt((input_box_all[3][2]-input_box_all[3][0])**2 + (input_box_all[3][3]-input_box_all[3][1])**2)
                if b41 <= 150 and b42 <= 150 and b43 <= 150 and b44 <= 150:
                    wait_4 += 1
                output.write(frm)
#                 if wait_4 > 200:
#                     print("Wait frame 4:",wait_4)
            elif person_count == 1:
                input_box_all = bounding_boxes_person_class(cor_list)
                if np.sqrt((input_box_all[0][2]-input_box_all[0][0])**2 + (input_box_all[0][3]-input_box_all[0][1])**2) < 215:
                    wait_1 += 1
                output.write(frm)
#                 if wait_1 > 200:
#                     print("Wait frame 1:",wait_1)
            elif person_count == 5:
                input_box_all = bounding_boxes_person_class(cor_list)
                b51 = np.sqrt((input_box_all[0][2]-input_box_all[0][0])**2 + (input_box_all[0][3]-input_box_all[0][1])**2)
                b52 = np.sqrt((input_box_all[1][2]-input_box_all[1][0])**2 + (input_box_all[1][3]-input_box_all[1][1])**2)
                b53 = np.sqrt((input_box_all[2][2]-input_box_all[2][0])**2 + (input_box_all[2][3]-input_box_all[2][1])**2)
                b54 = np.sqrt((input_box_all[3][2]-input_box_all[3][0])**2 + (input_box_all[3][3]-input_box_all[3][1])**2)
                b55 = np.sqrt((input_box_all[4][2]-input_box_all[4][0])**2 + (input_box_all[4][3]-input_box_all[4][1])**2)
                if b51 <= 150 and b52 <= 150 and b53 <= 150 and b54 <= 150 and b55 <= 150:
                    wait_5 += 1
                output.write(frm)
#                 if wait_other > 200:
#                     print("Wait frame other:",wait_other)
            elif person_count == 0:
                wait_0 += 1
                output.write(frm)

        
        if frame_counter > max_wait_3 and frame_counter < max_duration:
                    
            if wait_2 >= max_wait_2 or wait_4 >= max_wait_4 or wait_1 >= max_wait_1 or wait_5 >= max_wait_5 or wait_0 >= max_wait_0:
                # print(f"End of a Bout Detected with waiting time:{clip_counter}, wait0 = {wait_0}, wait1 = {wait_1}, wait2 = {wait_2}, wait4 = {wait_4}, wait5 = {wait_5}")
                
                end_time = timestamp(frame_count_time, fps)
                print(f"End of a Bout number {clip_counter} Detected with waiting time, ended at {end_time}")
                # print(f"Stop Timestamp: {end_time}")

                cell_address3 = f"C{clip_counter}"
                sheet[cell_address3] = end_time
                datetime2 = datetime.strptime(end_time, datetime_format)

                time_diff = datetime2 - datetime1
                cell_address4 = f"D{clip_counter}"
                sheet[cell_address4] = time_diff

                total_seconds = int(time_diff.total_seconds())
                if total_seconds <= 50:
                    cell_address5 = f"E{clip_counter}"
                    sheet[cell_address5] = "Might be an outlier!"

                cell_address6 = f"F{clip_counter}"
                sheet[cell_address6] = "End of a bout detected with waiting time!"
        
                frame_counter = 0  
                output.release()  
                clip_counter += 1
                wait_0 = 0
                wait_1 = 0
                wait_2 = 0
                wait_4 = 0
                wait_5 = 0
                wait_dist = 0
                wait_loc =0


        
            elif person_count == 3:
                input_box = bounding_boxes_person_class(cor_list)
                dist = relative_distance(input_box)
                if dist > 550:#earlier 550
                    wait_dist += 1
                    # print(f"wait distance: {wait_dist}")

                    if wait_dist == (max_wait_dist_duration*fps):
                        # print("End of a Bout Detected with distance:",clip_counter)

                        end_time = timestamp(frame_count_time, fps)
                        print(f"End of a Bout number {clip_counter} Detected with distance, ended at {end_time}")
                        # print(f"Stop Timestamp: {end_time}")

                        cell_address3 = f"C{clip_counter}"
                        sheet[cell_address3] = end_time
                        datetime2 = datetime.strptime(end_time, datetime_format)

                        time_diff = datetime2 - datetime1
                        cell_address4 = f"D{clip_counter}"
                        sheet[cell_address4] = time_diff

                        total_seconds = int(time_diff.total_seconds())
                        if total_seconds <= 50:
                            cell_address5 = f"E{clip_counter}"
                            sheet[cell_address5] = "Might be an outlier!"
                        
                        cell_address6 = f"F{clip_counter}"
                        sheet[cell_address6] = "End of a bout detected with Distance!"

                        frame_counter = 0  
                        output.release()
                        clip_counter += 1
                        wait_0 = 0
                        wait_1 = 0
                        wait_2 = 0
                        wait_4 = 0
                        wait_5 = 0  
                        wait_dist = 0
                        wait_loc = 0
                else:
                    cnt = 0
                    for i in range(len(input_box)):
                        if np.sqrt((input_box[i][2]-input_box[i][0])**2 + (input_box[i][3]-input_box[i][1])**2) < 215:
                            if (input_box[i][2] > 900 or input_box[i][0] < 80 or input_box[i][3] > 900 or input_box[i][1] < 80):
                                cnt += 1
                    if cnt == 3:
                        wait_loc += 1
                        # print(f"wait location: {wait_loc}")
                        
                        if wait_loc == (max_wait_loc_duration * fps):
                            # print("End of a Bout Detected with players location:",clip_counter)
                            
                            end_time = timestamp(frame_count_time, fps)
                            print(f"End of a Bout number {clip_counter} Detected with players location, ended at {end_time}")
                            # print(f"Stop Timestamp: {end_time}")

                            cell_address3 = f"C{clip_counter}"
                            sheet[cell_address3] = end_time
                            datetime2 = datetime.strptime(end_time, datetime_format)

                            time_diff = datetime2 - datetime1
                            cell_address4 = f"D{clip_counter}"
                            sheet[cell_address4] = time_diff

                            total_seconds = int(time_diff.total_seconds())
                            if total_seconds <= 50:
                                cell_address5 = f"E{clip_counter}"
                                sheet[cell_address5] = "Might be an outlier!"

                            cell_address6 = f"F{clip_counter}"
                            sheet[cell_address6] = "End of a bout detected with players location!"
                            

                            frame_counter = 0   
                            output.release() 
                            clip_counter += 1
                            wait_0 = 0
                            wait_1 = 0
                            wait_2 = 0
                            wait_4 = 0
                            wait_5 = 0 
                            wait_dist = 0
                            wait_loc =0 
        elif frame_counter == max_duration:
            # print("End of a Bout Detected with maximum duration reached:",clip_counter)
            
            end_time = timestamp(frame_count_time, fps)
            print(f"End of a Bout number {clip_counter} Detected with maximum duration reached, ended at {end_time}")
            print(f"-----------Human Intervention is needed for bout no. {clip_counter}-----------")
            # print(f"Stop Timestamp: {end_time}")

            cell_address3 = f"C{clip_counter}"
            sheet[cell_address3] = end_time
            datetime2 = datetime.strptime(end_time, datetime_format)

            time_diff = datetime2 - datetime1
            cell_address4 = f"D{clip_counter}"
            sheet[cell_address4] = time_diff

            cell_address5 = f"E{clip_counter}"
            sheet[cell_address5] = "Human intervention may be required!"

            cell_address6 = f"F{clip_counter}"
            sheet[cell_address6] = "Maximum duration reached!"
            
            frame_counter = 0 
            output.release() 
            clip_counter += 1
            wait_0 = 0
            wait_1 = 0
            wait_2 = 0
            wait_4 = 0
            wait_5 = 0 
            wait_dist = 0
            wait_loc =0 

        frame_counter += 1
        frame_count_time += 1


workbook.save(os.path.join(output_dir, "TimeStamp.xlsx"))
workbook.close()
video.release()  
cv2.destroyAllWindows()