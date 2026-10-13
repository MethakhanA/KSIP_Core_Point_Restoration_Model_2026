import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from tqdm import tqdm
# ============================


from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class UNet64(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 64x64 -> 32x32
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 32x32 -> 16x16

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 16x16 -> 32x32
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)   # 32x32 -> 64x64
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Matches [0, 1] normalized output

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.sigmoid(self.final(d1))

def test_model(image_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize model and load saved weights
    model = UNet64(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # 2. Load and preprocess the test image
    img = Image.open(image_path).convert('L')
    
    # Convert to tensor [1, 64, 64] and add batch dimension [1, 1, 64, 64]
    input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # 3. Predict
    with torch.no_grad():
        prediction = model(input_tensor) # Shape: [1, 1, 64, 64]

    # 4. Save the output
    # Remove batch dimension -> [1, 64, 64] and save
    save_image(prediction.squeeze(0), output_path)
    print(f"Saved prediction to: {output_path}")

if __name__ == "__main__":
    test_model(
        image_path=r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_TEST\F0000001_intensity.png",
        model_path=r"C:\work\image_processing\latent_fingerprint\model\unet64_weights.pth", 
        output_path=r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_OUT\F0000001_intensity.png"
    )