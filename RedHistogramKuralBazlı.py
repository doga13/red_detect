#Doğa Yüksel

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def mask_red_areas_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only image files (assuming jpeg images)
        if filename.endswith('.jpeg'):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {filename}. Skipping.")
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract RGB channels
            red_channel = image_rgb[:, :, 0]
            green_channel = image_rgb[:, :, 1]
            blue_channel = image_rgb[:, :, 2]

            # Determine red intensity
            red_intensity = red_channel - (green_channel / 2) - (blue_channel / 2)
            red_intensity[red_intensity < 0] = 0  # Zero out negative values

            # Calculate red intensity across the width
            red_intensity_per_column = np.sum(red_intensity, axis=0)

            # Calculate half of the width
            half_width = len(red_intensity_per_column) // 2

            # Calculate total red intensity for left and right halves
            left_red_intensity = np.sum(red_intensity_per_column[:half_width])
            right_red_intensity = np.sum(red_intensity_per_column[half_width:])

            # Check if left side has more red intensity than the right side
            if left_red_intensity > right_red_intensity:
                print(f"Left side red intensity is greater than the right side for {filename}. Saving original image.")
                plt.figure(figsize=(18, 6))

                # Original image
                plt.subplot(1, 3, 1)
                plt.title("Orijinal Görsel")
                plt.imshow(image_rgb)
                plt.axis('off')

                # Histogram
                plt.subplot(1, 3, 2)
                plt.title("Kırmızı Yoğunluk Histogramı")
                plt.xlabel("Width (Sütunlar)")
                plt.ylabel("Toplam Kırmızı Yoğunluğu")
                plt.plot(red_intensity_per_column, color='red')

                # Masked image (same as original in this case)
                plt.subplot(1, 3, 3)
                plt.title("Maskelenmiş Görsel")
                plt.imshow(image_rgb)
                plt.axis('off')

                plt.savefig(output_path)
                plt.close()
                continue

            # Check if total red intensity is zero
            if np.sum(red_intensity_per_column) == 0:
                print(f"Total red intensity is zero for {filename}. Saving original image.")
                plt.figure(figsize=(18, 6))

                # Original image
                plt.subplot(1, 3, 1)
                plt.title("Orijinal Görsel")
                plt.imshow(image_rgb)
                plt.axis('off')

                # Histogram
                plt.subplot(1, 3, 2)
                plt.title("Kırmızı Yoğunluk Histogramı")
                plt.xlabel("Width (Sütunlar)")
                plt.ylabel("Toplam Kırmızı Yoğunluğu")
                plt.plot(red_intensity_per_column, color='red')

                # Masked image (same as original in this case)
                plt.subplot(1, 3, 3)
                plt.title("Maskelenmiş Görsel")
                plt.imshow(image_rgb)
                plt.axis('off')

                plt.savefig(output_path)
                plt.close()
                continue

            # Calculate intensity differences
            intensity_diff = np.diff(red_intensity_per_column)

            # Detect columns with a significant increase
            threshold = np.mean(intensity_diff) + 1.65 * np.std(intensity_diff)
            large_increases = np.where(intensity_diff[half_width:] > threshold)[0] + half_width

            # Dynamically adjust threshold for flatness detection
            flatness_threshold = np.mean(red_intensity_per_column[:50]) + np.std(red_intensity_per_column[:50])

            # Check for flatness in the first 50 columns
            flat_region = np.all((red_intensity_per_column[:50] >= 0) & 
                                 (red_intensity_per_column[:50] <= flatness_threshold))

            if flat_region:
                # Find where the flat region ends
                flatness_start_index = 50 + np.argmax(red_intensity_per_column[50:] > flatness_threshold)
            else:
                # Use the first detected significant increase
                if large_increases.size > 0:
                    flatness_start_index = large_increases[0] + 1
                else:
                    flatness_start_index = np.argmax(red_intensity_per_column[half_width:] > 0) + half_width

            # Create a mask from the point where flatness ends
            mask = np.zeros_like(red_channel)
            mask[:, flatness_start_index:] = 255

            # Apply the mask on the original image
            masked_image = image_rgb.copy()
            masked_image[mask == 255] = [255, 0, 0]  # Use red color for masking

            # Save the masked image to the output folder
            plt.figure(figsize=(18, 6))

            # Original image
            plt.subplot(1, 3, 1)
            plt.title("Orijinal Görsel")
            plt.imshow(image_rgb)
            plt.axis('off')

            # Histogram
            plt.subplot(1, 3, 2)
            plt.title("Kırmızı Yoğunluk Histogramı")
            plt.xlabel("Width (Sütunlar)")
            plt.ylabel("Toplam Kırmızı Yoğunluğu")
            plt.plot(red_intensity_per_column, color='red')
            plt.axvline(x=flatness_start_index, color='blue', linestyle='--')

            # Masked image
            plt.subplot(1, 3, 3)
            plt.title("Maskelenmiş Görsel")
            plt.imshow(masked_image)
            plt.axis('off')

            plt.savefig(output_path)
            plt.close()

# Example usage
input_folder = r'C:\Users\doa\Desktop\y\images'
output_folder = r'C:\Users\doa\Desktop\y\degisimsolsagkarsilastir'
mask_red_areas_in_folder(input_folder, output_folder)
