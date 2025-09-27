import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from lapsrn import Net
from My_function import input_matrix_wpn, adapt_checkpoint
from torch.serialization import add_safe_globals
# === Config ===
checkpoint_path = "/content/drive/MyDrive/THESE/CODE/Projet_Demosaic/checkpoint/toto/mcan_model.pth"
image_path = "/content/drive/MyDrive/THESE/CODE/Projet_Demosaic/CAVE_dataset/new_val/clay_ms_IMECMine_DE.tif"  # remplace si besoin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Autoriser explicitement Net pour deserialization
add_safe_globals([Net])
# === Charger modèle ===
model = Net().to(device)
model.eval() 
#checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
adapted = adapt_checkpoint(checkpoint)

model.load_state_dict(adapted['model_state_dict'], strict=False)
model.eval()

# === Charger et préparer image ===
def load_tiff(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        if img.shape[0] == 16:
            img = img[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
        elif img.shape[2] == 16:
            img = np.moveaxis(img, 2, 0)[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
    raw_tensor = torch.from_numpy(img.copy()).float().unsqueeze(0) / 65535.0
    input_tensor = raw_tensor[:, [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
    return raw_tensor, input_tensor

raw_tensor, input_tensor = load_tiff(image_path)
H, W = raw_tensor.shape[2], raw_tensor.shape[3]
scale_map = input_matrix_wpn(H, W).to(device)

# === Inférence ===
with torch.no_grad():
    output = model([input_tensor.to(device), raw_tensor.to(device)], scale_map)
output = output.squeeze().cpu().numpy()
output = np.clip(output, 0, 1)

# === Afficher résultat ===
band_to_show = 8  # Choisis une bande de 0 à 15
plt.imshow(output[band_to_show], cmap='gray')
plt.title(f"Bande {band_to_show} reconstruite")
plt.axis('off')
plt.show()
