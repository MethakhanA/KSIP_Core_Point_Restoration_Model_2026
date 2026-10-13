import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

''' ### Currently Under Review by Methakhan 12/10/26'''
# ==========================================
# 1. Custom Dataset Class
# ==========================================
class TwoImageDataset(Dataset):
    def __init__(self, dir_in1, dir_in2, dir_target):
        self.dir_in1 = dir_in1 # input image 1
        self.dir_in2 = dir_in2 # input image 2
        self.dir_target = dir_target # target image
        
        # Sort filenames so sample_001.png aligns across all 3 directories
        self.filenames = sorted(os.listdir(dir_in1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load images as grayscale ('L' mode)
        img1 = Image.open(os.path.join(self.dir_in1, filename)).convert('L')
        img2 = Image.open(os.path.join(self.dir_in2, filename)).convert('L')
        target = Image.open(os.path.join(self.dir_target, filename)).convert('L')
        
        # Convert PIL images to Tensors and normalize pixel intensities to [0.0, 1.0]
        # to_tensor converts range [0, 255] -> [0.0, 1.0] and adds channel dimension: (1, 64, 64)
        t_img1 = TF.to_tensor(img1)
        t_img2 = TF.to_tensor(img2)
        t_target = TF.to_tensor(target)
        
        # Concatenate inputs along channel dimension: Shape becomes (2, 64, 64)
        inputs = torch.cat([t_img1, t_img2], dim=0)
        
        return inputs, t_target

# ==========================================
# 2. U-Net Architecture
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # Convolution 3x3
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True), # Activation func
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # Convolution 3x3
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True) # Activation func
        )

    def forward(self, x):
        return self.conv(x)

class UNet64(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        # In 2 channels, out 1 channel
        super().__init__() # Inherite from nn.Module
        self.enc1 = DoubleConv(in_channels, 32) # Convolution 32 out channels
        self.pool1 = nn.MaxPool2d(2)  # 64x64 -> 32x32 # Max Pooling
        
        self.enc2 = DoubleConv(32, 64) # Encoding 32->64
        self.pool2 = nn.MaxPool2d(2)  # 32x32 -> 16x16

        self.bottleneck = DoubleConv(64, 128) # Encoding 64->128

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

# ==========================================
# 3. Setup DataLoader & Training Loop
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths to directories
    dataset = TwoImageDataset(
        dir_in1="dataset/input1",
        dir_in2="dataset/input2",
        dir_target="dataset/target"
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Initialize Model, Loss Function, and Optimizer
    model = UNet64(in_channels=2, out_channels=1).to(device)
    criterion = nn.L1Loss()  # L1 loss works well for image translation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Example training loop
    epochs = 10
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")