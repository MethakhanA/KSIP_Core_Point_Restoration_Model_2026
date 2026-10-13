import os
from glob import glob
from tqdm import tqdm
import shutil
from PIL import Image
import wsq  # Registers the WSQ plugin with Pillow

def wsq_to_png(input_file, output_file):
    try:
        # Open the WSQ image
        img = Image.open(input_file)
        
        # Save it as a PNG file
        img.save(output_file, "PNG")
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# wsq_to_png("fingerprint.wsq", "fingerprint.png")


###---- Extract Format
# if __name__ == "__main__":
#     path = r"C:\work\image_processing\dataset\NIST SD14"
#     out_path = r"C:\work\image_processing\latent_fingerprint\model\data\SD14"
#     for file in glob(os.path.join(path, "*.wsq")):
#         filename = os.path.basename(file).split('.')[0]
#         wsq_to_png(file, os.path.join(out_path, filename+".png"))
## ---- ---- ----

###---- 
import cv2 as cv
if __name__ =="__main__":
    out_path = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_4000"
    # path = r"C:\work\image_processing\latent_fingerprint\model\data\SD14"
    path_file_list = glob(os.path.join(out_path, '*'))
    for i in tqdm(range(4000)):
        # shutil.copy(path_file_list[i], os.path.join(out_path, os.path.basename(path_file_list[i])))
        img = cv.imread(path_file_list[i], 0)
        resized_img = cv.resize(
            img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA
        )

        # Save or display the result
        cv.imwrite(os.path.join(path_file_list[i]), resized_img)