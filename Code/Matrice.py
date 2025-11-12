import numpy as np
import cv2
import os
import pandas as pd
from typing import Union

# --- PARAMÈTRES ET CHEMINS D'ACCÈS ---

# Chemins du dossier contenant les images de cellules
TRAIN_DIR = 'C-NMC_Leukemia/training_data' 
TEST_DIR = 'C-NMC_Leukemia/testing_data' 
VAL_DIR = 'C-NMC_Leukemia/validation_data' 

def creer_dataframe_depuis_dataset() -> pd.DataFrame:
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

    if os.path.isdir(TRAIN_DIR):
        for root, dirs, files in os.walk(TRAIN_DIR):
            base = os.path.basename(root).casefold()
            if base == 'hem':
                labels_name = 'hem'
            elif base == 'all':
                labels_name = 'all'
            else:
                continue  # Ignorer les autres dossiers
            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_filepaths.append(os.path.join(root, file_name))
                    all_labels.append(labels_name)
                    all_sets.append('train')
      
    ######################################################################
    #                         PARTIE TESTING                             #
    ######################################################################

    test_images_dir = os.path.join(TEST_DIR, 'C-NMC_test_final_phase_data')
    if os.path.isdir(test_images_dir):
        for file_name in os.listdir(test_images_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_filepaths.append(os.path.join(test_images_dir, file_name))
                all_labels.append('unknown')  # unknown car on sait pas si hem ou all
                all_sets.append('test') 
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


def charger_et_normaliser_image(chemin_image: str, taille: tuple) -> Union[np.ndarray, None]:
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

# --- PARAMÈTRES GLOBAUX (à ajuster si nécessaire) ---
# La taille que votre modèle attend (comme 224x224 dans votre code Jupyter)
IMAGE_TAILLE = (224, 224)

# --- EXÉCUTION ---
# Exemple d'appel pour créer la DataFrame
dataset_df = creer_dataframe_depuis_dataset()

# Dossier pour les données traitées
processed_dir = 'processed_data'

# Si vous voulez l'utiliser immédiatement pour le prétraitement :
if not dataset_df.empty:
    nb_images_a_traiter = len(dataset_df)  # Traiter toutes les images
    for i in range(nb_images_a_traiter):
        image_a_traiter = dataset_df['filepaths'].iloc[i]
        set_name = dataset_df['set'].iloc[i]
        label_name = dataset_df['labels'].iloc[i]
        
        print(f"\n--- Traitement de l'image {i+1}/{nb_images_a_traiter} : {image_a_traiter} ---")
        tableau_numpy = charger_et_normaliser_image(image_a_traiter, IMAGE_TAILLE)
        
        if tableau_numpy is not None:
            # Créer le dossier pour cette classe
            class_dir = os.path.join(processed_dir, set_name, label_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Affiche le type et la forme du résultat
            print("Informations sur le résultat :")
            print(f"Type de données : {tableau_numpy.dtype}")
            print(f"Forme du tableau : {tableau_numpy.shape}")
            print("Contenu (premières valeurs du premier pixel) :")
            print(tableau_numpy[0, 0, :])  # Affiche les valeurs du premier pixel
            
            # Pour la première image, afficher TOUS les pixels (sauvegardé dans un fichier texte dans processed_dir)
            if i == 0:
                print("\nMatrice complète de l'image (TOUS les pixels sauvegardés dans un fichier texte) :")
                output_txt_file = os.path.join(processed_dir, 'matrice_image_1_complete.txt')
                np.savetxt(output_txt_file, tableau_numpy.reshape(-1, 3), fmt='%.6f')  # Sauvegarder en format texte
                print(f"Matrice complète sauvegardée dans : {output_txt_file}")
                print("Vous pouvez ouvrir ce fichier pour voir tous les pixels.")
            # Pour les autres images, pas d'affichage de la matrice pour éviter la surcharge
            
            # Sauvegarder la matrice complète dans le dossier de classe
            output_file = os.path.join(class_dir, f'matrice_image_{i+1}.npy')
            np.save(output_file, tableau_numpy)
            print(f"Matrice complète sauvegardée dans le fichier : {output_file}")
            print("Vous pouvez la charger plus tard avec : np.load('{output_file}')")
            
            # Copier l'image originale non traitée dans le même dossier
            import shutil
            _, ext = os.path.splitext(image_a_traiter)
            original_output_file = os.path.join(class_dir, f'image_originale_{i+1}{ext}')
            shutil.copy(image_a_traiter, original_output_file)
            print(f"Image originale copiée dans : {original_output_file}")
        else:
            print("Erreur lors du traitement de l'image.")
else:
    print("Aucune image trouvée dans le dataset.")

