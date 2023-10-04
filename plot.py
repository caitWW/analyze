import cv2

# Callback function for mouse events
def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Event is left button down
        print(f'Pixel coordinates: x = {x}, y = {y}')

# Read an image
img_path = 'test2_result/CLEVR_new_000000.png'  # Use your image path here
img = cv2.imread(img_path)

# Create a window and set mouse callback
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', callback)

# Show the image and wait for a key press
cv2.imshow('image', img)
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()