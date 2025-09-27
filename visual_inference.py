import os
import tifffile
import numpy as np
import torch
from tqdm import tqdm
from lapsrn import Net
from My_function import input_matrix_wpn
# Configuration
checkpoint_path = "/content/drive/MyDrive/THESE/CODE/Projet_Demosaic/checkpoint/De_happy_model_epoch_250.pth"
input_dir = "CAVE_dataset/new_val"
output_dir = "visual_results"
os.makedirs(output_dir, exist_ok=True)

# Initialisation du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction pour adapter les poids du checkpoint
def adapt_checkpoint(checkpoint):
    state_dict = checkpoint['model'].state_dict() if 'model' in checkpoint else checkpoint
    
    # Conversion des clés FC vers MHSA
    key_mapping = {
        'convt_br1_front.0.se.fc.0.weight': 'convt_br1_front.0.se.proj.weight',
        'convt_br1_front.0.se.fc.2.weight': 'convt_br1_front.0.se.mhsa.out_proj.weight',
        'convt_F1.0.ma.fc.0.weight': 'convt_F1.0.ma.mhsa.in_proj_weight',
        'convt_F1.0.ma.fc.2.weight': 'convt_F1.0.ma.mhsa.out_proj.weight',
        'convt_F2.0.ma.fc.0.weight': 'convt_F2.0.ma.mhsa.in_proj_weight',
        'convt_F2.0.ma.fc.2.weight': 'convt_F2.0.ma.mhsa.out_proj.weight'
    }
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # Transposition des matrices de poids si nécessaire
        if key.endswith('fc.0.weight'):
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value.t()  # Transposition importante
        else:
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value
    
    return {'model_state_dict': new_state_dict}

# Chargement adaptatif du modèle
model = Net().to(device)
'''
try:
    # Essai de chargement standard
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Vérification de compatibilité
    try:
        model.load_state_dict(checkpoint['model'].state_dict())
        print("Checkpoint chargé directement avec succès")
    except:
        print("Adaptation des poids du checkpoint...")
        adapted_checkpoint = adapt_checkpoint(checkpoint)
        model.load_state_dict(adapted_checkpoint['model_state_dict'], strict=False)
        
except Exception as e:
    print(f"Erreur de chargement : {str(e)}")
    raise

model.eval()
'''
try:
    # Essai en mode sécurisé d'abord
    torch.serialization.add_safe_globals([Net])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print("Checkpoint chargé en mode sécurisé")
    
except Exception as secure_error:
    print(f"Échec du chargement sécurisé : {secure_error}")
    
    if input("Voulez-vous essayer le mode non sécurisé ? (y/n) ").lower() == 'y':
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print("Checkpoint chargé en mode non sécurisé")
        except Exception as insecure_error:
            print(f"Échec du chargement : {insecure_error}")
            raise
    else:
        raise secure_error

# Initialisation du modèle
model = Net().to(device)
model.load_state_dict(checkpoint["model"].state_dict())
model.eval()

def load_tiff(path):
    """Charge et prépare l'image TIFF"""
    try:
        img = tifffile.imread(path)
        print(f"Shape original : {img.shape}")
        
        # Conversion des dimensions
        if img.ndim == 3:
            if img.shape[0] == 16:  # Format CHW
                img = img[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
            elif img.shape[2] == 16:  # Format HWC
                img = np.moveaxis(img, 2, 0)[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
        
        raw_tensor = torch.from_numpy(img.copy()).float().unsqueeze(0) / 65535.0
        input_tensor = raw_tensor[:, [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15], :, :]
        
        return raw_tensor, input_tensor
    
    except Exception as e:
        print(f"Erreur chargement {os.path.basename(path)}: {str(e)}")
        return None, None

def process_image(img_path):
    """Effectue le dématriçage complet"""
    raw_tensor, input_tensor = load_tiff(img_path)
    if raw_tensor is None:
        return None
    
    _, _, H, W = input_tensor.shape
    scale_coord_map = input_matrix_wpn(H, W).to(device)
    
    with torch.no_grad():
        output = model([input_tensor.to(device), raw_tensor.to(device)], scale_coord_map)
    
    return output.cpu()

def save_results(output, path):
    """Sauvegarde les résultats en TIFF 16 bits"""
    output_np = output.squeeze().numpy()
    output_np = (np.clip(output_np, 0, 1) * 65535).astype(np.uint16)
    output_np = output_np[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], :, :]
    tifffile.imwrite(path, output_np)
    print(f"Résultat sauvegardé : {path}")

# Traitement des images
tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
print(f"\nDébut du dématriçage de {len(tiff_files)} images...")

for filename in tqdm(tiff_files):
    try:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        output = process_image(input_path)
        if output is not None:
            save_results(output, output_path)
            
    except Exception as e:
        print(f"\nÉchec du traitement {filename}: {str(e)}")

print("\nDématriçage terminé. Résultats disponibles dans:", os.path.abspath(output_dir))