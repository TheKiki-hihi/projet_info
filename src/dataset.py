import os
import cv2
import numpy as np
import pandas as pd


###############################
#     PARAMÈTRES GLOBAUX      #
###############################

# Dossiers contenant les images brutes
TRAIN_DIR = "C-NMC_Leukemia/training_data"
TEST_DIR = "C-NMC_Leukemia/testing_data"
VAL_DIR = "C-NMC_Leukemia/validation_data"

# Dossier où seront sauvegardées les images prétraitées
PROCESSED_DIR = "processed_data"

# Taille cible des images 
IMAGE_TAILLE = (224, 224)


#########################################################
#        FONCTION PRINCIPALE DE PRÉTRAITEMENT           #
#########################################################

def preparer_donnees():
    """
    Cette fonction réalise le prétraitement des données d'images.

    Étapes :
    1. Parcours des dossiers training / validation / testing
    2. Création d'une DataFrame contenant :
       - le chemin de chaque image
       - son label
       - l'ensemble auquel elle appartient
    3. Chargement des images
    4. Redimensionnement
    5. Normalisation
    6. Sauvegarde des images prétraitées au format .npy

    Cette fonction est appelée depuis main.py.
    """

    # Listes qui serviront à construire la DataFrame finale
    all_filepaths = []
    all_labels = []
    all_sets = []
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

    ######################################################
    # CRÉATION DE LA DATAFRAME.  
    ######################################################
    
    if not chemins_images:
        print("Aucune image trouvée. Vérifiez les chemins.")
        return

    df = pd.DataFrame({
        "chemin_image": all_filepaths,
        "label": all_labels,
        "ensemble": all_sets
    })

    print("\nRépartition des images par ensemble :")
    print(df["ensemble"].value_counts())

    ########################################
    #       PRÉTRAITEMENT DES IMAGES       #
    ######################################## 
    for i, row in df.iterrows():
        # Chargement de l'image
        image = cv2.imread(row["chemin_image"])

        # Si l'image est illisible, on passe à la suivante
        if image is None:
            continue

        # Redimensionnement de l'image
        image = cv2.resize(image, IMAGE_TAILLE)

        # Normalisation des pixels entre 0 et 1
        image = image.astype(np.float32) / 255.0

        # Création du dossier de sortie
        dossier_sortie = os.path.join(
            PROCESSED_DIR,
            row["ensemble"],
            str(row["label"])
        )
        os.makedirs(dossier_sortie, exist_ok=True)

        # Nom du fichier de sortie (.npy)
        nom_image = os.path.splitext(
            os.path.basename(row["chemin_image"])
        )[0]
        chemin_sortie = os.path.join(dossier_sortie, nom_image + ".npy")

        # Sauvegarde de l'image prétraitée
        np.save(chemin_sortie, image)

        # Affichage de la progression
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(df)} images traitées")
