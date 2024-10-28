#Doğa Yüksel

import cv2
import numpy as np
import os

# Define folders for images and boxes
image_folder_path = r"C:\Users\doa\Desktop\y\denemelabel"
box_folder_path = r"C:\Users\doa\Desktop\y\denemelabel"
output_folder_path = r"C:\Users\doa\Desktop\y\denemelabel"

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Get a list of all files in the folders
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpeg')]
box_files = [f for f in os.listdir(box_folder_path) if f.endswith('.txt')]

# Sort the lists to ensure matching order
image_files.sort()
box_files.sort()

# Function to process each image and its corresponding box file
def process_image_and_boxes(image_path, boxes_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Load the text file
    with open(boxes_path, 'r') as file:
        boxes = file.readlines()

    # Get image dimensions
    img_height, img_width, _ = image.shape

    # Initialize variables for global score calculation
    glbRed = 0
    glbBlue = 0
    glbGreen = 0
    results = []

    # Initialize a list to store the text results
    text_results = []

    for i, box in enumerate(boxes):
        # Extract the values
        cls, cx, cy, w, h = map(float, box.split())

        # Convert from normalized to actual coordinates
        cx = int(cx * img_width)
        cy = int(cy * img_height)
        w = int(w * img_width * 1)
        h = int(h * img_height * 1)

        # Get the coordinates of the box
        x1 = max(0, cx - w // 2)
        x2 = min(img_width, cx + w // 2)
        y1 = max(0, cy - h // 2)
        y2 = min(img_height, cy + h // 2)

        # Extract the box from the image
        box_pixels = image[y1:y2, x1:x2]

        # Calculate the total sum of RGB values
        sum_r = np.sum(box_pixels[:, :, 2])
        sum_g = np.sum(box_pixels[:, :, 1])
        sum_b = np.sum(box_pixels[:, :, 0])

        # Calculate the number of pixels
        num_pixels = box_pixels.shape[0] * box_pixels.shape[1]

        # Calculate the average RGB values
        avg_r = sum_r / num_pixels
        avg_g = sum_g / num_pixels
        avg_b = sum_b / num_pixels

        # Update global RGB sums
        glbRed += avg_r
        glbBlue += avg_b
        glbGreen += avg_g

        # Save the results
        results.append((int(cls), sum_r, sum_g, sum_b, avg_r, avg_g, avg_b, x1, x2, y1, h))

    # Calculate global score
    if len(results) > 0:
        glbScore = (glbRed - (glbBlue + glbGreen) // 2) / len(results)
    else:
        glbScore = 0

    # Prepare text for the right side with only SCORE
    for (cls, sum_r, sum_g, sum_b, avg_r, avg_g, avg_b, x1, x2, y1, h) in results:
        score = avg_r - (avg_b + avg_g) // 2 - glbScore
        text = f'Digit {cls}: SCORE: {score:.2f}'
        if(avg_b>avg_r*1.3):
          text = f'Digit {cls}: SCORE: {score:.2f}'
          
        
        text_results.append(text)
        

    # Calculate the height needed for the text
    line_height = 20  # Height for each line of text
    text_height = len(text_results) * line_height + 30  # Add some margin at the top

    # Set the height of the new image to accommodate the text
    total_img_height = max(img_height, text_height)
    text_area_width = 200  # Adjust width for the text area
    new_img_width = img_width + text_area_width

    # Create a new image with white background
    new_image = np.ones((total_img_height, new_img_width, 3), dtype=np.uint8) * 255

    # Place the original image on the left side
    new_image[:img_height, :img_width] = image

    # Draw the text on the right side of the new image
    text_y = 30  # Starting y position for text
    for line in text_results:
        cv2.putText(new_image, line, (img_width + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_y += line_height  # Increment y position for the next line of text

    # Save the final image
    output_image_path = os.path.join(output_folder_path, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, new_image)

# Process each pair of image and box files
for image_file, box_file in zip(image_files, box_files):
    image_path = os.path.join(image_folder_path, image_file)
    boxes_path = os.path.join(box_folder_path, box_file)
    process_image_and_boxes(image_path, boxes_path)