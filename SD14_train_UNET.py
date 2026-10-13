import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.utils import save_image  # <-- Added for saving images

# ==========================================
# 1. Dataset & Model (Unchanged)
# ==========================================
class MultiInputImageDataset(Dataset):
    def __init__(self, dir_in, dir_target, separator="_"):
        self.dir_in = dir_in
        self.dir_target = dir_target
        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        input_files = sorted(os.listdir(dir_in))

        for filename in input_files:
            if not filename.lower().endswith(valid_exts):
                continue
            if separator in filename:
                base_name, _ = filename.rsplit(separator, 1)
                ext = os.path.splitext(filename)[1]
                target_filename = f"{base_name}{ext}"
                target_path = os.path.join(self.dir_target, target_filename)

                if os.path.exists(target_path):
                    input_path = os.path.join(self.dir_in, filename)
                    self.samples.append((input_path, target_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]
        img = Image.open(input_path).convert('L')
        target = Image.open(target_path).convert('L')
        t_img = TF.to_tensor(img)
        t_target = TF.to_tensor(target)
        return t_img, t_target

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
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

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
# 2. Helper Functions (Unchanged)
# ==========================================
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = -10.0 * math.log10(mse.item())
    return psnr

def plot_metrics(train_losses, val_losses, val_psnrs):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (L1)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_psnrs, 'g--', label='Val PSNR')
    ax2.set_ylabel('PSNR (dB)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    plt.title('Training Metrics over Epochs')
    plt.grid(True)
    plt.savefig('training_metrics.png')
    plt.close() # Close plot so it doesn't block execution if running in terminal

# ==========================================
# 3. Main Script
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurations
    dir_in = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_NOISE"
    dir_target = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_CORE_CLEAN"
    out_dir = r"C:\work\image_processing\latent_fingerprint\model\data\SD14_Training\SD_OUT"  # <-- Directory to save output images
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3
    save_path = "best_unet.pth"

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Load Full Dataset
    full_dataset = MultiInputImageDataset(dir_in=dir_in, dir_target=dir_target)
    total_size = len(full_dataset)
    
    # Ratio Control (70% Train, 15% Val, 15% Test)
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 1.0 - train_ratio - val_ratio
    
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: {train_size} Train | {val_size} Val | {test_size} Test")

    model = UNet64(in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
    best_val_loss = float('inf')

    # --- TRAINING & VALIDATION LOOP ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                batch_psnr = calculate_psnr(outputs, targets)
                val_psnr_total += batch_psnr * inputs.size(0)
                
        val_loss /= len(val_dataset)
        val_psnr_avg = val_psnr_total / len(val_dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr_avg)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val PSNR: {val_psnr_avg:.2f} dB")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    plot_metrics(history['train_loss'], history['val_loss'], history['val_psnr'])

    # --- TESTING PHASE ---
    print(f"\n--- Starting Test Phase (Saving to '{out_dir}/') ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_loss = 0.0
    test_psnr = 0.0
    saved_img_count = 0  # Counter for saved image filenames
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            test_loss += criterion(outputs, targets).item() * inputs.size(0)
            test_psnr += calculate_psnr(outputs, targets) * inputs.size(0)
            
            # --- SAVE TEST IMAGES ---
            # Loop through the current batch
            for i in range(inputs.size(0)):
                pred_img = outputs[i]
                target_img = targets[i]
                input_img = inputs[i]
                
                # Save predicted output, ground truth, and input for comparison
                save_image(pred_img, os.path.join(out_dir, f"test_{saved_img_count:03d}_pred.png"))
                save_image(target_img, os.path.join(out_dir, f"test_{saved_img_count:03d}_target.png"))
                save_image(input_img, os.path.join(out_dir, f"test_{saved_img_count:03d}_input.png"))
                
                saved_img_count += 1
            
    test_loss /= len(test_dataset)
    test_psnr /= len(test_dataset)
    
    print(f"Test Results -> Loss (L1): {test_loss:.4f} | PSNR: {test_psnr:.2f} dB")
    print(f"[*] Saved {saved_img_count} test image sets to '{out_dir}/'")