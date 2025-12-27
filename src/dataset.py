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
    chemins_images = []
    labels = []
    ensembles = []

    # ----------------------------------------------------
    # 1) PARCOURS DU JEU D'ENTRAÎNEMENT
    # ----------------------------------------------------
    # Les dossiers 'hem' et 'all' correspondent aux classes
    if os.path.isdir(TRAIN_DIR):
        for root, _, files in os.walk(TRAIN_DIR):
            nom_classe = os.path.basename(root).lower()

            # On ne garde que les dossiers de classes
            if nom_classe not in ["hem", "all"]:
                continue

            for fichier in files:
                if fichier.lower().endswith((".jpg", ".png", ".jpeg")):
                    chemins_images.append(os.path.join(root, fichier))
                    labels.append(nom_classe)
                    ensembles.append("training")

    # ----------------------------------------------------
    # 2) PARCOURS DU JEU DE TEST
    # ----------------------------------------------------
    # Les images de test n'ont pas de labels connus
    test_images_dir = os.path.join(TEST_DIR, "C-NMC_test_final_phase_data")

    if os.path.isdir(test_images_dir):
        for fichier in os.listdir(test_images_dir):
            if fichier.lower().endswith((".jpg", ".png", ".jpeg")):
                chemins_images.append(os.path.join(test_images_dir, fichier))
                labels.append("unknown")
                ensembles.append("testing")

    # ----------------------------------------------------
    # 3) PARCOURS DU JEU DE VALIDATION
    # ----------------------------------------------------
    # Les labels sont stockés dans un fichier CSV
    val_images_dir = os.path.join(VAL_DIR, "C-NMC_test_prelim_phase_data")
    val_labels_csv = os.path.join(
        VAL_DIR, "C-NMC_test_prelim_phase_data_labels.csv"
    )

    if os.path.isdir(val_images_dir) and os.path.isfile(val_labels_csv):
        df_labels = pd.read_csv(val_labels_csv)

        for fichier in os.listdir(val_images_dir):
            if fichier.lower().endswith((".jpg", ".png", ".jpeg")):
                ligne = df_labels[df_labels["new_names"] == fichier]

                # On vérifie que le label existe dans le CSV
                if not ligne.empty:
                    chemins_images.append(os.path.join(val_images_dir, fichier))
                    labels.append(ligne["labels"].values[0])
                    ensembles.append("validation")

    ######################################################
    # CRÉATION DE LA DATAFRAME.  
    ######################################################
    
    if not chemins_images:
        print("Aucune image trouvée. Vérifiez les chemins.")
        return

    df = pd.DataFrame({
        "chemin_image": chemins_images,
        "label": labels,
        "ensemble": ensembles
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
