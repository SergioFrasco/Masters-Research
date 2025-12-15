import cv2
import os
from datetime import datetime
import numpy as np
from glob import glob
import re

def create_video_from_images(image_folder, output_video_path, fps=5, sort_numerically=True):
    """
    Create a video from a folder of images.
    
    Parameters:
    - image_folder: Path to the folder containing PNG images
    - output_video_path: Path where the output video will be saved
    - fps: Frames per second for the output video
    - sort_numerically: If True, sorts filenames numerically (e.g., img1.png, img2.png, img10.png)
                        If False, sorts alphabetically (e.g., img1.png, img10.png, img2.png)
    """
    # Get all PNG images in the folder
    image_files = glob(os.path.join(image_folder, "*.png"))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return
        
    # Sort the images
    if sort_numerically:
        # Extract numbers from filenames and sort based on those numbers
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or '0'))
    else:
        # Regular alphabetical sort
        image_files.sort()
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI format
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Counter for progress tracking
    total_images = len(image_files)
    print(f"Creating video from {total_images} images...")
    
    # Add each image to the video
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        video.write(img)
        
        # Print progress
        if (i + 1) % 10 == 0 or i + 1 == total_images:
            print(f"Processed {i + 1}/{total_images} images")
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_video_path}")

# Example usage
if __name__ == "__main__":
    # Replace these with your own paths
    image_folder = "../results/current/24_04_2025_1"
    output_video = "../results/new_model_video.mp4"
    
    create_video_from_images(image_folder, output_video, fps=30)

def get_latest_run_dir(parent_dir="results/current"):
    """
    Returns the run directory with the highest number under today's date,
    e.g., results/current/2025-05-27/3
    """
    today = datetime.today().strftime("%m_%d_%Y")
    date_dir = os.path.join(parent_dir, today)

    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"No directory found for today's date: {date_dir}")

    run_dirs = []
    for d in os.listdir(date_dir):
        d_path = os.path.join(date_dir, d)
        if os.path.isdir(d_path) and d.isdigit():
            run_dirs.append((int(d), d_path))

    if not run_dirs:
        raise FileNotFoundError(f"No numeric run directories found in '{date_dir}'")

    # ✅ FIX: Use numeric comparison, then return the path
    _, latest_run_path = max(run_dirs, key=lambda x: x[0])
    return latest_run_path