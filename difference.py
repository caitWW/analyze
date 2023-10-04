import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import math
from natsort import natsorted, ns

# Paths to your folders
folder1 = "/Users/cw/Desktop/run2_test2/Original/"
folder2 = "/Users/cw/Desktop/run2_test2/Recon/"

# Get a list of filenames in each folder, sorted alphabetically
files1 = natsorted(os.listdir(folder1), alg=ns.PATH)
files2 = natsorted(os.listdir(folder2), alg=ns.PATH)

def calculate_distance_from_center(height, width, center_x, center_y):
    y_indices, x_indices = np.indices((height, width))
    return np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)

with open('/Users/cw/Desktop/run2_retina.json', 'r') as f:
    fixation_data = json.load(f)

with open('/Users/cw/Desktop/ids', 'r') as f:
    ids = json.load(f)

i = 0
# Initialize empty lists to hold all distance and error values
distance = []
errors = []
fixation = [[143, 69], [276, 106], [82, 142], [266, 154], [161, 113], [120, 119], 
            [267, 126], [147, 62], [57, 123], [100, 96], [200, 111], [207, 153], 
            [154, 86], [175, 133], [217, 173], [68, 74], [169, 64], [197, 80], [141, 175],
            [173, 132], [272, 93], [181, 110], [131, 171], [70, 87], [83, 70], [151, 74],
            [202, 61], [51, 68], [266, 166], [169, 171], [134, 170], [163, 117]]

# Iterate through both lists simultaneously
for file1, file2 in zip(files1, files2):

    if not (file1.endswith(".png") and file2.endswith(".png")):
        continue

    image_id = ids[i]
    

    # fixation = fixation_data[image_id]
    x_fixation = fixation[i][0]
    y_fixation = fixation[i][1]
    i+=1

    # Complete paths to the images
    img_path1 = os.path.join(folder1, file1)
    print(x_fixation)
    print(img_path1)
    img_path2 = os.path.join(folder2, file2)

    # Load the two images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Convert the images from BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Calculate squared difference
    # diff = np.square(img1.astype(np.float64) - img2.astype(np.float64))

    # Compute the absolute difference on a per-pixel basis
    diff = cv2.absdiff(img1, img2)

    # Sum differences along color channels to get total difference per pixel
    diff = np.sum(diff, axis=-1)

    x = min(x_fixation, 320-x_fixation)
    y = min(y_fixation, 240-y_fixation)
    min_dist = math.sqrt(x**2+y**2)

    # Calculate distance from center for each pixel
    distances = calculate_distance_from_center(img1.shape[0], img1.shape[1], x_fixation, y_fixation)

    #mask = distances <= min_dist
    #print(min_dist)

    # Flatten the distance and error arrays and append them to the lists
    distance.append(distances.flatten())
    errors.append(diff.flatten())

# Flatten lists of arrays into single arrays
distance = np.concatenate(distance)
errors = np.concatenate(errors)

# Create a DataFrame
df = pd.DataFrame({'Distance': distance, 'Error': errors})

# Group by 'Distance' and calculate the mean of 'Error'
df_grouped = df.groupby('Distance')['Error'].mean().reset_index()

# Plot averaged differences against distances
plt.scatter(df_grouped['Distance'], df_grouped['Error'], s=1)
plt.xlabel('Distance from Fixation Point (320x240 images)')
plt.ylabel('Difference (0-765 per pixel)')
plt.show()
