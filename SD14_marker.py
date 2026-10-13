import os
import pandas as pd
from PIL import Image

# --- Configuration ---
csv_file = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_POINT_CSV\SD14_4000.csv"      # Path to your CSV file
input_dir = r'C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD14'        # Folder containing the original images
output_dir = r'C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_CLEAN'     # Folder where cropped images will be saved
crop_size = 64                    # Size of the cropped image (64x64)





# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Calculate half-size for the math (32 pixels on each side of the center)
half_size = crop_size // 2

# Iterate through each row in the CSV
for index, row in df.iterrows():
    filename = row['filename']
    x = row['x']
    y = row['y']
    # ---- Uncomment if coord match the image
    x = 2*x
    y = 2*y
    
    img_path = os.path.join(input_dir, filename)
    
    # Check if the image actually exists in the folder
    if not os.path.exists(img_path):
        print(f"Warning: {filename} not found in {input_dir}. Skipping.")
        continue
        
    try:
        # Open the image
        with Image.open(img_path) as img:
            # Calculate the bounding box for the crop
            # The box is defined as a 4-tuple: (left, upper, right, lower)
            left = x - half_size
            upper = y - half_size
            right = x + half_size
            lower = y + half_size
            
            # Crop and save the image
            cropped_img = img.crop((left, upper, right, lower))
            
            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path)
            
            print(f"Successfully cropped and saved: {filename}")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Done processing all images!")