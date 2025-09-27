import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
#from libtiff import TIFFfile, TIFFimage
import tifffile as tiff
from tifffile import TiffFile
from os import listdir
from os.path import join
from PIL import Image

def msfaTOcube(raw, msfa_size):
    mask = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)  # Crée un masque vide
    cube = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)  # Crée un cube vide
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1  # Remplit le masque selon le motif MSFA
    for i in range(msfa_size**2):
        cube[:, :, i] = raw * (mask[:, :, i])  # Applique le masque pour créer les couches du cube
    return cube  # Retourne le cube d'images multibandes

def mask_input(GT_image, msfa_size):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)  # Crée un masque vide
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1  # Remplit le masque selon le motif MSFA
    GT_image = GT_image[:, :, :16]  # ou [:, :, :nC] selon ton cas
    input_image = mask * GT_image  # Applique le masque à l'image de référence
    return input_image  # Retourne l'image masquée

def reorder_imec(old):
    ### reorder the multiband cube, making the center wavelength from small to large
    _, _, C = old.shape  # Extraire les dimensions du cube : hauteur, largeur, et nombre de canaux (C)
    new = np.zeros_like(old)  # Créer un nouveau tableau vide de la même forme que 'old'
    
    if C == 16:  # Vérifier si le cube a 16 canaux
        new[:, :, 0] = old[:, :, 2]  # Réorganiser les canaux selon l'ordre spécifié
        new[:, :, 1] = old[:, :, 0]
        new[:, :, 2] = old[:, :, 9]
        new[:, :, 3] = old[:, :, 1]
        new[:, :, 4] = old[:, :, 15]
        new[:, :, 5] = old[:, :, 14]
        new[:, :, 6] = old[:, :, 12]
        new[:, :, 7] = old[:, :, 13]
        new[:, :, 8] = old[:, :, 7]
        new[:, :, 9] = old[:, :, 6]
        new[:, :, 10] = old[:, :, 4]
        new[:, :, 11] = old[:, :, 5]
        new[:, :, 12] = old[:, :, 11]
        new[:, :, 13] = old[:, :, 10]
        new[:, :, 14] = old[:, :, 8]
        new[:, :, 15] = old[:, :, 3]
        return new
    elif C==25:
        new[:, :, 0] = old[:, :, 2]
        new[:, :, 1] = old[:, :, 9]
        new[:, :, 2] = old[:, :, 14]
        new[:, :, 3] = old[:, :, 13]
        new[:, :, 4] = old[:, :, 12]
        new[:, :, 5] = old[:, :, 10]
        new[:, :, 6] = old[:, :, 11]
        new[:, :, 7] = old[:, :, 8]
        new[:, :, 8] = old[:, :, 7]
        new[:, :, 9] = old[:, :, 5]
        new[:, :, 10] = old[:, :, 6]
        new[:, :, 11] = old[:, :, 23]
        new[:, :, 12] = old[:, :, 22]
        new[:, :, 13] = old[:, :, 20]
        new[:, :, 14] = old[:, :, 21]
        new[:, :, 15] = old[:, :, 3]
        new[:, :, 16] = old[:, :, 0]
        new[:, :, 17] = old[:, :, 1]
        new[:, :, 18] = old[:, :, 18]
        new[:, :, 19] = old[:, :, 17]
        new[:, :, 20] = old[:, :, 15]
        new[:, :, 21] = old[:, :, 16]
        new[:, :, 22] = old[:, :, 24]
        new[:, :, 23] = old[:, :, 19]
        new[:, :, 24] = old[:, :, 4]
        return new

def reorder_2filter(old):
    ### reorder the multiband cube as the real pattern in MSFA
    C, _, _ = old.shape  # Extraire le nombre de canaux (C), et ignorer la hauteur et la largeur
    new = np.zeros_like(old)  # Créer un nouveau tableau vide de la même forme que 'old'
    
    if C == 16:
        order = [2, 0, 9, 1, 15, 14, 12, 13, 7, 6, 4, 5, 11, 10, 8, 3]  # Ordre de réorganisation pour 16 canaux
        for i in range(0, 16):
            new[order[i], :, :] = old[i, :, :]  # Remplir 'new' selon l'ordre spécifié
        return new
    
    elif C == 25:
        order = [2, 9, 14, 13, 12, 10, 11, 8, 7, 5, 6, 23, 22, 20, 21, 3, 0, 1, 18, 17, 15, 16, 24, 19, 4]  # Ordre pour 25 canaux
        for i in range(0, 25):
            new[order[i], :, :] = old[i, :, :]  # Remplir 'new' selon cet ordre
        return new


def input_matrix_wpn(inH, inW, msfa_size):
    h_offset_coord = torch.zeros(inH, inW, 1)  # Créer un tableau d'offsets pour les coordonnées de hauteur
    w_offset_coord = torch.zeros(inH, inW, 1)  # Créer un tableau d'offsets pour les coordonnées de largeur

    for i in range(0, msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i + 1) / msfa_size  # Remplir les offsets de hauteur
        w_offset_coord[:, i::msfa_size, 0] = (i + 1) / msfa_size  # Remplir les offsets de largeur

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)  # Combiner les coordonnées de hauteur et de largeur
    pos_mat = pos_mat.contiguous().view(1, -1, 2)  # Redimensionner la matrice pour l'adapter à la structure souhaitée
    return pos_mat  # Retourner la matrice des positions

def load_img(filepath):
    tif = TiffFile(filepath)  # Charger le fichier TIFF en utilisant la bibliothèque libtiff
    picture, _ = tiff.get_samples()  # Extraire les échantillons d'image
    img = picture[0].transpose(2, 1, 0)  # Transposer les dimensions de l'image (reordonne les axes)
    return img  # Retourner l'image chargée


def normalization(x):
    _range = np.max(x) - np.min(x)
    if _range == 0:  # Vérification pour éviter la division par zéro
        return np.zeros_like(x)  # Retourne un tableau de zéros si toutes les valeurs sont identiques
    return (x - np.min(x)) / _range

def adapt_checkpoint(checkpoint):
    """
    Adapte un checkpoint contenant d'anciennes couches FC vers les nouvelles couches MHSA.
    Transpose les poids si nécessaire.
    """
    state_dict = checkpoint['model'].state_dict() if 'model' in checkpoint else checkpoint

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
        new_key = key_mapping.get(key, key)
        if key.endswith('fc.0.weight'):
            new_state_dict[new_key] = value.t()  # Transpose pour correspondre à la nouvelle structure
        else:
            new_state_dict[new_key] = value

    return {'model_state_dict': new_state_dict}
