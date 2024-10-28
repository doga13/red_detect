#Doğa Yüksel

import cv2
import numpy as np
import os

# Define folders for images and boxes
image_folder_path = r"C:\Users\doa\Desktop\y\images"
box_folder_path = r"C:\Users\doa\Desktop\x\labels"
output_folder_path = r"C:\Users\doa\Desktop\y\maskelenmisscore"

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

    # Copy the original image for masking purposes
    masked_image = image.copy()

    for i, box in enumerate(boxes):
        # Extract the values
        cls, cx, cy, w, h = map(float, box.split())

        # Convert from normalized to actual coordinates
        cx = int(cx * img_width)
        cy = int(cy * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

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

        # Determine if the box is Blue Dominant
        blue_dominant = avg_b > avg_r * 1.3

        # Save the results
        results.append((int(cls), avg_r, avg_g, avg_b, blue_dominant, x1, x2, y1, y2))

    # Calculate global score
    if len(results) > 0:
        glbScore = (glbRed - ((glbBlue + glbGreen) / 2)) / len(results)
    else:
        glbScore = 0

    # Initialize a flag to track if masking should be applied
    apply_masking = False

    # Determine if the first three scores are negative
    if len(results) >= 3 and all((r[1] - ((r[2] + r[3]) / 2) - glbScore) < 0 for r in results[:3]):
        # Check for special case with 5 digits
        if len(results) == 5 and (results[-1][1] - ((results[-1][2] + results[-1][3]) / 2) - glbScore) > 0:
            apply_masking = True
            mask_indices = [4]  # Only mask the last digit
        else:
            apply_masking = True
            mask_indices = [i for i, (_, r, g, b, bd, _, _, _, _) in enumerate(results) 
                            if (r - ((b + g) / 2) - glbScore) > 0 and not bd]

    # Prepare text for the right side and apply mask if applicable
    for i, (cls, avg_r, avg_g, avg_b, blue_dominant, x1, x2, y1, y2) in enumerate(results):
        score = avg_r - ((avg_b + avg_g) / 2) - glbScore
        if blue_dominant:
            text = f'Digit {cls}: Blue Dominant: SCORE: {score:.2f}'
        else:
            text = f'Digit {cls}: SCORE: {score:.2f}'
        
        text_results.append(text)

        # Apply the mask if required
        if apply_masking and i in mask_indices:
            masked_image[y1:y2, x1:x2] = [0, 255, 0]  # Green color

    # Calculate the height needed for the text
    line_height = 20  # Height for each line of text
    text_height = len(text_results) * line_height + 30  # Add some margin at the top

    # Set the height of the new image to accommodate the text
    total_img_height = max(img_height, text_height)
    text_area_width = 200  # Adjust width for the text area
    mask_area_width = img_width  # Width for the masked image area
    new_img_width = img_width + text_area_width + mask_area_width

    # Create a new image with white background
    new_image = np.ones((total_img_height, new_img_width, 3), dtype=np.uint8) * 255

    # Place the original image on the left side
    new_image[:img_height, :img_width] = image

    # Draw the text on the middle of the new image
    text_y = 30  # Starting y position for text
    for line in text_results:
        cv2.putText(new_image, line, (img_width + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_y += line_height  # Increment y position for the next line of text

    # Place the masked image on the right side
    new_image[:img_height, img_width + text_area_width:img_width + text_area_width + mask_area_width] = masked_image

    # Save the final image
    output_image_path = os.path.join(output_folder_path, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, new_image)

# Process each pair of image and box files
for image_file, box_file in zip(image_files, box_files):
    image_path = os.path.join(image_folder_path, image_file)
    boxes_path = os.path.join(box_folder_path, box_file)
    process_image_and_boxes(image_path, boxes_path)
