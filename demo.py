import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import matplotlib.pyplot as plt

from libtiff import TIFFfile, TIFFimage
from os.path import join
from sklearn.metrics import mean_squared_error
from My_function import reorder_imec
#from sewar.full_ref import ergas_matlab
from sewar.full_ref import ergas
from lapsrn import Net

def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    
    tif = TIFFfile(filepath)  # Charger le fichier TIFF à partir du chemin donné
    picture, _ = tif.get_samples()  # Extraire les données d'image (échantillons) à partir du fichier TIFF
    img = picture[0].transpose(2, 1, 0)  # Transposer les dimensions de l'image (changer l'ordre des axes)
    
    # img_test = Image.fromarray(img[:,:,1])  # Cette ligne est commentée : elle convertit un canal en image PIL
    return img  # Retourner l'image transposée


def mask_input(GT_image):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)  # Initialiser un masque de taille (hauteur, largeur, 16)
    # Définir les sous-échantillons pour chaque canal
    mask[0::4, 0::4, 0] = 1
    mask[0::4, 1::4, 1] = 1
    mask[0::4, 2::4, 2] = 1
    mask[0::4, 3::4, 3] = 1
    mask[1::4, 0::4, 4] = 1
    mask[1::4, 1::4, 5] = 1
    mask[1::4, 2::4, 6] = 1
    mask[1::4, 3::4, 7] = 1
    mask[2::4, 0::4, 8] = 1
    mask[2::4, 1::4, 9] = 1
    mask[2::4, 2::4, 10] = 1
    mask[2::4, 3::4, 11] = 1
    mask[3::4, 0::4, 12] = 1
    mask[3::4, 1::4, 13] = 1
    mask[3::4, 2::4, 14] = 1
    mask[3::4, 3::4, 15] = 1
    
    # Créer l'image masquée
    input_image = mask * GT_image
    return input_image


def psnr(x_true, x_pred):
    n_bands = x_true.shape[2]  # Nombre de bandes spectrales dans l'image
    PSNR = np.zeros(n_bands)  # Tableau pour stocker les PSNR par bande
    MSE = np.zeros(n_bands)   # Tableau pour stocker les MSE par bande
    mask = np.ones(n_bands)   # Masque pour filtrer les bandes où la valeur maximale est zéro

    x_true = x_true[:, :, :]  # Image de vérité terrain
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])  # Aplatissement de la bande k de l'image réelle
        x_pred_k = x_pred[:, :, k].reshape([-1])  # Aplatissement de la bande k de l'image prédite

        MSE[k] = mean_squared_error(x_true_k, x_pred_k)  # Calcul de la MSE pour la bande k

        MAX_k = np.max(x_true_k)  # Valeur maximale dans la bande k de l'image réelle
        if MAX_k != 0:
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])  # Calcul du PSNR si la valeur max n'est pas nulle
        else:
            mask[k] = 0  # Si la valeur max est zéro, on ignore cette bande en mettant le masque à 0

    psnr = PSNR.sum() / mask.sum()  # Calcul du PSNR moyen en tenant compte des bandes non nulles
    mse = MSE.mean()  # Calcul de la MSE moyenne
    return psnr, mse  # Retourne le PSNR et la MSE moyennés


def ssim(x_true, x_pre):
    num = x_true.shape[2]  # Nombre de bandes spectrales dans l'image
    ssimm = np.zeros(num)  # Tableau pour stocker les valeurs SSIM par bande
    c1 = 0.0001  # Constante pour stabiliser la division par de petites valeurs
    c2 = 0.0009  # Constante pour stabiliser la division par de petites valeurs
    n = 0  # Compteur pour les bandes
    
    for x in range(x_true.shape[2]):  # Pour chaque bande spectrale
        z = np.reshape(x_pre[:, :, x], [-1])  # Aplatir la bande x de l'image prédite
        sa = np.reshape(x_true[:, :, x], [-1])  # Aplatir la bande x de l'image vraie
        
        # Calcul de la covariance entre l'image réelle et l'image prédite pour cette bande
        y = [z, sa]
        cov = np.cov(y)  # Calcul de la matrice de covariance
        oz = cov[0, 0]   # Variance de z (image prédite)
        osa = cov[1, 1]  # Variance de sa (image vraie)
        ozsa = cov[0, 1]  # Covariance entre z et sa
        
        # Moyennes des bandes
        ez = np.mean(z)
        esa = np.mean(sa)
        
        # Calcul du SSIM pour la bande x
        ssimm[n] = ((2 * ez * esa + c1) * (2 * ozsa + c2)) / ((ez * ez + esa * esa + c1) * (oz + osa + c2))
        n += 1  # Incrémenter le compteur

    # Calcul de la moyenne du SSIM sur toutes les bandes
    SSIM = np.mean(ssimm)
    return SSIM  # Retourne le SSIM moyen


def sam(x_true, x_pre):
    # Calcule le nombre total de pixels (largeur * hauteur)
    num = (x_true.shape[0]) * (x_true.shape[1])
    
    # Initialise un tableau pour stocker les valeurs SAM de chaque pixel
    samm = np.zeros(num)
    
    # Compteur pour suivre l'index du pixel dans le tableau samm
    n = 0
    
    # Parcourt chaque pixel dans les dimensions 2D de l'image (hauteur et largeur)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            # Extrait les valeurs spectrales pour le pixel courant dans les deux images et les aplatie
            z = np.reshape(x_pre[x, y, :], [-1])
            sa = np.reshape(x_true[x, y, :], [-1])
            
            # Calcule le produit scalaire des vecteurs spectraux
            tem1 = np.dot(z, sa) + 2.2204e-16  # Ajoute une petite constante pour éviter la division par zéro
            
            # Calcule le produit des normes (magnitudes) des vecteurs spectraux
            tem2 = (np.linalg.norm(z) + 2.2204e-16) * (np.linalg.norm(sa) + 2.2204e-16)
            
            # Calcule le cosinus de l'angle entre les vecteurs spectraux
            buff1 = tem1 / tem2
            
            # Vérifie si la valeur du cosinus est valide pour éviter les erreurs de domaine dans arccos
            if buff1 > 1:
                samm[n] = 0  # Définit à 0 si hors du domaine valide pour arccos
            else:
                samm[n] = np.arccos(buff1)  # Calcule SAM (angle en radians) si valide
            
            # Passe au pixel suivant dans le tableau samm
            n = n + 1
    
    # Calcule la moyenne de toutes les valeurs SAM pour tous les pixels
    buff = np.mean(samm)
    
    # Convertit SAM des radians en degrés
    SAM = buff * 180 / np.pi
    
    return SAM


def sam1(x_true, x_pre):
    # Calcule le produit élément par élément entre les images vraies et prédites
    buff1 = x_true * x_pre
    
    # Somme des produits à travers la troisième dimension (canaux spectraux)
    buff2 = np.sum(buff1, 2)
    
    # Remplace les valeurs nulles par une petite constante pour éviter la division par zéro
    buff2[buff2 == 0] = 2.2204e-16
    
    # Calcule la norme (magnitude) de l'image vraie pour chaque pixel
    buff4 = np.sqrt(np.sum(x_true * x_true, 2))
    buff4[buff4 == 0] = 2.2204e-16  # Évite la division par zéro
    
    # Calcule la norme (magnitude) de l'image prédite pour chaque pixel
    buff5 = np.sqrt(np.sum(x_pre * x_pre, 2))
    buff5[buff5 == 0] = 2.2204e-16  # Évite la division par zéro
    
    # Calcule le rapport entre le produit scalaire et la norme de l'image vraie
    buff6 = buff2 / buff4
    
    # Calcule le rapport final en divisant par la norme de l'image prédite
    buff8 = buff6 / buff5
    
    # Limite les valeurs à 1 pour éviter les erreurs de domaine dans arccos
    buff8[buff8 > 1] = 1
    
    # Calcule la moyenne de l'angle (SAM en radians) pour tous les pixels
    buff9 = np.mean(np.arccos(buff8))
    
    # Convertit l'angle SAM en degrés
    SAM = buff9 * 180 / np.pi
    
    return SAM


def PSNR(pred, gt, shave_border=0):
    # Récupère la hauteur et la largeur de l'image prédite
    height, width = pred.shape[:2]
    
    # Si spécifié, coupe les bords des images prédite et vraie
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    
    # Calcule la différence entre les images prédite et vraie
    imdff = pred - gt
    
    # Calcule la racine de l'erreur quadratique moyenne (RMSE)
    rmse = math.sqrt(np.mean(imdff ** 2))
    
    # Si l'erreur est nulle, retourne un PSNR très élevé (100) indiquant une correspondance parfaite
    if rmse == 0:
        return 100
    
    # Calcule le PSNR en dB en fonction du RMSE
    return 20 * math.log10(255.0 / rmse)


def input_matrix_wpn(inH, inW, add_id_channel=False):
    '''
    inH, inW : la hauteur et la largeur des cartes de caractéristiques (feature maps)
    add_id_channel : option pour ajouter un canal d'identité (non utilisé ici)
    '''

    # Définit les dimensions de sortie en fonction des dimensions d'entrée
    outH, outW = inH, inW
    
    # Crée des matrices pour les coordonnées de décalage en hauteur et en largeur, initialisées à zéro
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    
    # Définit les décalages pour chaque groupe de lignes en blocs de 4 dans la matrice de hauteur
    h_offset_coord[0::4, :, 0] = 0.25  # Déplacement de 0.25 pour la première ligne de chaque bloc
    h_offset_coord[1::4, :, 0] = 0.5   # Déplacement de 0.5 pour la deuxième ligne de chaque bloc
    h_offset_coord[2::4, :, 0] = 0.75  # Déplacement de 0.75 pour la troisième ligne de chaque bloc
    h_offset_coord[3::4, :, 0] = 1.0   # Déplacement de 1.0 pour la quatrième ligne de chaque bloc
    
    # Définit les décalages pour chaque groupe de colonnes en blocs de 4 dans la matrice de largeur
    w_offset_coord[:, 0::4, 0] = 0.25  # Déplacement de 0.25 pour la première colonne de chaque bloc
    w_offset_coord[:, 1::4, 0] = 0.5   # Déplacement de 0.5 pour la deuxième colonne de chaque bloc
    w_offset_coord[:, 2::4, 0] = 0.75  # Déplacement de 0.75 pour la troisième colonne de chaque bloc
    w_offset_coord[:, 3::4, 0] = 1.0   # Déplacement de 1.0 pour la quatrième colonne de chaque bloc
    
    # Concatène les matrices de décalage en hauteur et en largeur pour former une matrice de position
    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    
    # Applique un changement de forme (flattening) pour obtenir un tenseur de dimensions (1, -1, 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)

    return pos_mat


parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/mcan_model.pth", type=str, help="model path")
parser.add_argument("--val_dir", default="/content/drive/MyDrive/THESE/CODE/Projet_Demosaic/CAVE_dataset/new_val", type=str, help="model path")
parser.add_argument("--image", default="beads_ms", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

opt = parser.parse_args()
cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# Initialisation du modèle et chargement des poids sauvegardés
model = Net()
m_state_dict = torch.load(opt.model)  # Chargement des poids du modèle à partir du chemin spécifié dans opt.model
model.load_state_dict(m_state_dict, strict=False)
# Chargement de l'image de référence et normalisation de la plage de valeurs à 0-255
im_gt_y = load_img(opt.val_dir + "\\" + opt.image + "_" + "IMECMine_D65" + ".tif")  # Image 3D (512, 512, 16)
max_new = np.max(im_gt_y)
im_gt_y = im_gt_y / max_new * 255  # Normalisation à 255

# Transpose l'image pour correspondre au format souhaité (largeur, hauteur, canaux)
im_gt_y = im_gt_y.transpose(1, 0, 2)

# Masque et réorganise l'image en entrée (hyperspectrale)
im_l_y = mask_input(im_gt_y)  # Applique un masque à l'image
im_l_y = reorder_imec(im_l_y)  # Réorganise les canaux spectrals
im_gt_y = reorder_imec(im_gt_y)  # Réorganise l'image de référence de la même manière

# Conversion en float pour effectuer des calculs précis
im_gt_y = im_gt_y.astype(float)
im_l_y = im_l_y.astype(float)

# Mise à l'échelle de l'entrée de l'image en entrée dans une plage de 0-1
im_input = im_l_y / 255.

# Transpose les images pour obtenir le format (canaux, hauteur, largeur)
im_gt_y = im_gt_y.transpose(2, 0, 1)
im_l_y = im_l_y.transpose(2, 0, 1)
im_input = im_input.transpose(2, 0, 1)

# Crée une carte de coordonnées d'échelle avec la fonction `input_matrix_wpn`
raw = im_input.sum(axis=0)
scale_coord_map = input_matrix_wpn(raw.shape[0], raw.shape[1])

# Convertit l'entrée en un tenseur PyTorch et ajoute la dimension du lot
im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])
raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])

# Déplace les données sur le GPU si disponible
if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
    raw = raw.cuda()
    scale_coord_map = scale_coord_map.cuda()
else:
    model = model.cpu()

# Mesure du temps d'exécution du modèle
start_time = time.time()
HR_4x = model([im_input, raw], scale_coord_map)  # Passage dans le modèle
elapsed_time = time.time() - start_time

# Transformation du résultat en format image et mise à l'échelle des valeurs de pixel de 0 à 255
HR_4x = HR_4x.cpu()
im_h_y = HR_4x.data[0].numpy().astype(np.float32) * 255.
im_h_y = np.clip(np.rint(im_h_y), 0, 255).astype(np.uint8).astype(np.float)

# Prépare `raw` et `im_input` pour affichage ou sauvegarde
raw = raw.cpu().data[0].numpy().astype(np.float32) * 255.
raw = np.clip(raw, 0, 255)

im_input = im_input.cpu().data[0].numpy().astype(np.float32) * 255.
im_input = np.clip(im_input, 0, 255)

# Calcul et affichage des métriques d'évaluation
[psnr_predicted, mse] = psnr(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("PSNR_multi=", psnr_predicted)
ssim_predicted = ssim(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("ssim_multi=", ssim_predicted)
sam_predicted = sam1(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("sam_multi=", sam_predicted)
ergas_predicted = ergas_matlab(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("ergas_predicted=", ergas_predicted)
print("It takes {}s for processing".format(elapsed_time))

# Affichage des images de référence, d'entrée, de sortie et des canaux
nband = 0
fig = plt.figure()
ax = plt.subplot("221")
ax.imshow(im_gt_y[nband, :, :], cmap='gray')
ax.set_title("GT")

ax = plt.subplot("222")
ax.imshow(im_input[nband, :, :], cmap='gray')
ax.set_title("Input(band1)")

ax = plt.subplot("223")
ax.imshow(raw[0, :, :], cmap='gray')
ax.set_title("Input(raw)")

ax = plt.subplot("224")
ax.imshow(im_h_y[nband, :, :], cmap='gray')
ax.set_title("Output(MCAN)")
plt.show()
