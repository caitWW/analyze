import os
import cv2
import json

def draw_cross(im, fixations, color=(0, 255, 0), thickness=2, size=3):
    """
    Draw a cross at the location of each fixation.

    Parameters:
    - im: The image to draw on.
    - fixations: The list of fixation points.
    - color: The color of the cross (default is green).
    - thickness: The thickness of the lines (default is 2).
    - size: The size of the cross (default is 20).
    """
    for (x, y) in fixations:
        # Draw a line from top to bottom
        cv2.line(im, (x, y-size), (x, y+size), color, thickness)
        # Draw a line from left to right
        cv2.line(im, (x-size, y), (x+size, y), color, thickness)
    return im

# Directory containing images
folder_path = '/Users/cw/Desktop/run2_test2_shifted/Recon/'

with open('/Users/cw/Desktop/run2_retina.json', 'r') as f:
    fixation_data = json.load(f)

with open('/Users/cw/Desktop/ids', 'r') as f:
    ids = json.load(f)

'''
filename = 'test_run_64_test_1_9.png'

img_path = os.path.join(folder_path, filename)
img = cv2.imread(img_path)

# Define fixation points, you should replace this with actual data
fixations = [(100, 96)]
img_with_cross = draw_cross(img, fixations)

# If you want to save the image with the cross, uncomment the line below.
cv2.imwrite('/Users/cw/Desktop/hi3/'+filename, img_with_cross)
'''


i = 0
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".png"): 
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Check the dimensions (width and height) of the image
        height, width, channels = img.shape

        print(f"Image width: {width} pixels")
        print(f"Image height: {height} pixels")
        print(f"Number of channels: {channels}")

        image_id = ids[i]
        i += 1

        fixation = fixation_data[image_id]
        print(fixation)
        x_fixation = fixation['xc']
        y_fixation = fixation['yc'] 

        # y_fixation = img.shape[0] - y_fixation  # assuming y_fixation starts from the bottom

        # Define fixation points, you should replace this with actual data
        fixations = [(160, 120)]

        img_with_cross = draw_cross(img, fixations)

        # If you want to save the image with the cross, uncomment the line below.
        cv2.imwrite('/Users/cw/Desktop/hi5/'+filename, img_with_cross)
