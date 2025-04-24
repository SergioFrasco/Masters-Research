import cv2
import os

image_folder = '../results/current/24_04_2025_2'
output_video = '../results/current/old_vision_model.mp4'  # Save as MP4 now

# Get image files
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
images.sort()

if not images:
    raise ValueError("No images found.")

# Read first image for dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
if frame is None:
    raise ValueError(f"Couldn't read first image: {first_image_path}")

height, width, layers = frame.shape

# MP4 with 'mp4v' codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames
written_frames = 0
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Warning: Couldn't read {image_path}, skipping.")
        continue
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    video.write(frame)
    written_frames += 1

video.release()

if written_frames == 0:
    print("❌ No valid frames were written. Check image paths and formats.")
else:
    print(f"✅ Video saved successfully with {written_frames} frames: {output_video}")
