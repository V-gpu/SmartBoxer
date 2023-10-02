import pandas as pd

import pathlib
import glob
import os
from natsort import natsorted
import shutil

filnumber = 1
save_path_new = str(pathlib.Path(__file__).parent.resolve())
#print('save_path_new',save_path_new)

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

outputpath_New = CWD + '/' +  '0SegmentedVideos' + '/' + 'renamedsegmentedbouts' + '/'


# # removing the older 
# shutil.rmtree(outputpath_New)

isExist = os.path.exists(outputpath_New)
if isExist:
    # removing the older 
    shutil.rmtree(outputpath_New)

elif not isExist:
   # Create a new directory because it does not exist
   os.makedirs(outputpath_New)
   print("The output directory is created!")
   # PGM to make a duplicate filelist
   movingfiles = os.listdir(outputpath)
   # Iterate through the files and copy them to the destination folder
   for file in movingfiles:
        source_file = os.path.join(outputpath, file)
        destination_file = os.path.join(outputpath_New, file)
        
        # Check if the item is a file (not a subfolder)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)  # Use copy2 to preserve metadata



# # once moved delete the segmentedbouts folder
# #shutil.rmtree(outputpath)


# #For renaming
filelist = glob.glob(outputpath_New+'/'+"*.mp4")  # Get all files in the current folder
filess = natsorted(filelist)
print('listoffiless',filess)

# Define the path to the Excel file and sheet name
excel_file = 'Boxing Sparing Data.xlsx'  # Replace with your Excel file path
sheet_name = '2023'  # Replace with your sheet name

# Load the Excel sheet into a DataFrame
df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
#numberofvideos = tota + 1
indxx = 289

for k in range(289, 309+1): #37 to 52 na 36 to 51+1; 291 to 311 na 289 to 309+1
    # Define the row and column indices (0-based)
    print('k',k)
    row_index = k  # Replace with the row index you want to access (e.g., row 3)

    column_name0 = 'Player 1(Red Corner)'  # Replace with the column name you want to access
    column_name1 = 'Player 2(Blue Corner)'  # Replace with the column name you want to access

    # Access the specific cell value based on row and column
    try:
        cell_value0 = df.at[row_index, column_name0]
        cell_value1 = df.at[row_index, column_name1]
        print(f'Value at row {row_index + 1}, {column_name0}: {cell_value0}')
        print(f'Value at row {row_index + 1}, {column_name1}: {cell_value1}')
        new_folder_name = str(cell_value0) + '_'+ str(cell_value1) +'_'+ str(k-indxx) +'.avi'
        # print('new_folder_name',new_folder_name)
    except KeyError:
        print(f'Invalid row or column name: Row {row_index + 1}, Column {column_name0}')
    except IndexError:
        print(f'Invalid row index: {row_index + 1}')
        

    outputpath_new_folder_name = outputpath_New + new_folder_name  # renamed
    print('outputpath_new_folder_name',outputpath_new_folder_name)

    #print('filess',filess)
    print('checking whether it works-----------------------------------------',filess[k-indxx])
    videofilename = filess[k-indxx]
    filnumber +=1
    #nameofvideo = os.path.split(file)[-1]
    #nameoffolder,ext = os.path.splitext(nameofvideo)
    current_folder_name = videofilename
    print('current_folder_name',current_folder_name)
    new_folder_name = outputpath_new_folder_name

    # print('current_folder_name',current_folder_name)
    print('new_folder_name',new_folder_name)

    if os.path.exists(current_folder_name):
        # Rename the folder
        os.rename(current_folder_name, new_folder_name)
        print(f'Renamed: {current_folder_name} -> {new_folder_name}')
    else:
        print(f'Folder does not exist: {current_folder_name}')



