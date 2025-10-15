import numpy as np
import cv2
import os
import pandas as pd
import os

# --- PARAMÈTRES ET CHEMINS D'ACCÈS ---

# Chemins du dossier contenant les images de cellules
TRAIN_DIR = 'C-NMC_Leukemia/training_data' 
TEST_DIR = 'C-NMC_Leukemia/testing_data' 
VAL_DIR = 'C-NMC_Leukemia/validation_data' 

def creer_dataframe_depuis_dataset(root_dir: str) -> pd.DataFrame:
    """
    Parcourt les différents dossiers training, testing, validation 
    pour créer une DataFrame contenant les chemins de fichiers et leurs étiquettes.

    Args:
        train_dir, test_dir, val_dir: Le chemin du dossier racine en question.

    Returns:
        pd.DataFrame: Une DataFrame avec les colonnes ['filepaths', 'labels', 'set'] 
    """

    all_filepaths = []
    all_labels = []
    all_sets = []

    # print(f"-> Traitement du dossier racine : {root_dir}")
    
    ######################################################################
    #                         PARTIE TRAINING                            #
    ######################################################################

    if os.path.isdir(train_dir):
        # Parcourir les dossiers de train_dir
        for dirs, files in os.walk(train_dir):
            for root, dirs, files in os.walk(train_dir):
            base = os.path.basename(root).casefold() # prend la fin du chemin et le mets comme nom de dossier pour creer classe et convertit en minuscule
            if base == 'hem':
                labels_name = 'hem'
            elif base == 'all' :
                labels_name = 'all'
                    for file_name in os.listdir(root):
                        if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            all_filepaths.append(os.path.join(root, file_name))
                            all_labels.append(labels_name)
                            all_sets.append('train')

    ######################################################################
    #                         PARTIE TESTING                             #
    ######################################################################

    ######################################################################
    #                        PARTIE VALIDATION                           #
    ######################################################################
    # Traitement de la validation data
    val_images_dir = os.path.join(VAL_DIR, 'C-NMC_test_prelim_phase_data')
    val_labels_csv = os.path.join(VAL_DIR, 'C-NMC_test_prelim_phase_data_labels.csv')
    
    if os.path.isdir(val_images_dir) and os.path.isfile(val_labels_csv):
        print(f"  -> Traitement de la validation data")
        # Lire le CSV des labels
        labels_df = pd.read_csv(val_labels_csv)
        # Mapper les labels : 0 -> 'hem', 1 -> 'all'
        label_mapping = {0: 'hem', 1: 'all'}
        labels_df['labels'] = labels_df['labels'].map(label_mapping)
        
        # Parcourir les images
        for file_name in os.listdir(val_images_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Trouver le label correspondant via new_names
                matching_row = labels_df[labels_df['new_names'] == file_name]
                if not matching_row.empty:
                    label = matching_row['labels'].values[0]
                    all_filepaths.append(os.path.join(val_images_dir, file_name))
                    all_labels.append(label)
                    all_sets.append('validation')
                else:
                    print(f"Avertissement : Pas de label trouvé pour {file_name}")
    # Création de la DataFrame Finale
    if not all_filepaths:
        print("\nERREUR : Aucune image trouvée. Vérifiez le chemin DATA_ROOT_DIR et la structure des dossiers.")
        return pd.DataFrame(columns=['filepaths', 'labels', 'set'])

    final_df = pd.DataFrame({
        'filepaths': all_filepaths,
        'labels': all_labels,
        'set': all_sets
    })

    # Aperçu de la distribution
    print("\n--- RÉSULTAT ---")
    print("DataFrame finale créée avec succès.")
    print("Distribution des données par ensemble et par classe :")
    print(final_df.groupby('set')['labels'].value_counts().unstack(fill_value=0))
    print(f"\nTotal des images récupérées : {len(final_df)}")

    return final_df


def charger_et_normaliser_image(chemin_image: str, taille: tuple) -> np.ndarray | None:
    """
    Charge une image, la redimensionne, la convertit en RGB, et la normalise
    pour que les valeurs de pixels soient comprises entre 0.0 et 1.0 (float32).
    """
    if not os.path.exists(chemin_image):
        return None

    # Chargement de l'image en BGR (standard OpenCV)
    image = cv2.imread(chemin_image, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Redimensionnement à la taille cible (224x224)
    image_redimensionnee = cv2.resize(image, taille, interpolation=cv2.INTER_LINEAR)
    
    # Conversion BGR -> RGB (la convention standard pour les modèles de ML/DL)
    image_rgb = cv2.cvtColor(image_redimensionnee, cv2.COLOR_BGR2RGB)
    
    # Normalisation : Division par 255.0 pour mettre les valeurs entre 0.0 et 1.0
    # Le type est forcé à float32 (standard en Machine Learning)
    image_normalisee = image_rgb.astype('float32') / 255.0
    
    return image_normalisee # Forme (224, 224, 3), valeurs entre [0.0, 1.0]
    


