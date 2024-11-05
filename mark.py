import cv2

# Load the image
image_path = 'D:/Documents/GitHub/weedy_images/results_captured_image_20241022_152954.jpg'  # Change to your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Define the marker position and properties
marker_position = (800, 737)  # (x, y) coordinates, change to your target position
marker_color = (0, 0, 255)    # Red color in BGR
marker_size = 10               # Size of the marker
marker_thickness = 2           # Thickness of the marker lines

# Place the marker
cv2.drawMarker(image, marker_position, marker_color, markerType=cv2.MARKER_CROSS, 
               markerSize=marker_size, thickness=marker_thickness)

# Define the marker position and properties
marker_position = (421, 691)  # (x, y) coordinates, change to your target position
marker_color = (0, 255, 0)    # Red color in BGR
marker_size = 10               # Size of the marker
marker_thickness = 2           # Thickness of the marker lines

# Place the marker
cv2.drawMarker(image, marker_position, marker_color, markerType=cv2.MARKER_CROSS, 
               markerSize=marker_size, thickness=marker_thickness)

# Save the resulting image
output_path = 'D:/Documents/GitHub/weedy_images/results_captured_image_20241022_152954_mark.jpg'  # Change to your desired output path
cv2.imwrite(output_path, image)

