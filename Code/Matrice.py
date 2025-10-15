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

    # Mapping des dossiers aux ensembles
    fold_mapping = {
        'fold_0': 'train',
        'fold_1': 'validation',
        'fold_2': 'test'
    }

    print(f"-> Traitement du dossier racine : {root_dir}")

    if os.path.isdir(root_dir):
        # Parcourir les dossiers fold_0, fold_1, fold_2
        for fold_name, set_name in fold_mapping.items():
            fold_path = os.path.join(root_dir, fold_name)
            if os.path.isdir(fold_path):
                print(f"  -> Traitement de {fold_name} ({set_name})")
                # Parcourir les dossiers de classes 'hem' et 'all'
                for class_name in ['hem', 'all']:
                    class_path = os.path.join(fold_path, class_name)
                    if os.path.isdir(class_path):
                        for file_name in os.listdir(class_path):
                            if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                all_filepaths.append(os.path.join(class_path, file_name))
                                all_labels.append(class_name)
                                all_sets.append(set_name)

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

# --- PARAMÈTRES GLOBAUX (à ajuster si nécessaire) ---
# La taille que votre modèle attend (comme 224x224 dans votre code Jupyter)
IMAGE_TAILLE = (224, 224)

def charger_et_encoder_image(chemin_image: str) -> np.ndarray | None:
    """
    Charge une image à partir d'un chemin d'accès, la redimensionne,
    et la convertit en un tableau NumPy.

    Args:
        chemin_image (str): Le chemin d'accès complet au fichier image.

    Returns:
        np.ndarray | None: Le tableau NumPy de l'image (224x224x3) ou None en cas d'erreur.
    """

    # 1. Vérification du Chemin
    if not os.path.exists(chemin_image):
        print(f"Erreur : Le fichier n'existe pas à l'emplacement : {chemin_image}")
        return None

    # 2. Lecture de l'Image avec OpenCV (cv2)
    # flag cv2.IMREAD_COLOR garantit la lecture en couleur (3 canaux)
    image = cv2.imread(chemin_image, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Erreur : Impossible de lire l'image. Vérifiez le format du fichier.")
        return None

    # 3. Traitement de l'Image

    # Redimensionnement à la taille attendue par le modèle (224x224)
    image_redimensionnee = cv2.resize(image, IMAGE_TAILLE, interpolation=cv2.INTER_LINEAR)

    # Conversion du format de couleur BGR (OpenCV par défaut) à RGB (Keras/TensorFlow)
    image_rgb = cv2.cvtColor(image_redimensionnee, cv2.COLOR_BGR2RGB)

    # Normalisation des valeurs des pixels (mise à l'échelle)
    # Les pixels sont typiquement de 0 à 255. La division par 255.0 les met entre 0.0 et 1.0.
    # C'est une étape de prétraitement essentielle pour les modèles de Deep Learning.
    image_normalisee = image_rgb.astype('float32') / 255.0

    # 4. Renvoi de l'Encodage Numérique (Tableau NumPy)

    # Le tableau aura la forme (Hauteur, Largeur, Canaux), soit (224, 224, 3)
    print(f"Image traitée avec succès. Forme du tableau NumPy : {image_normalisee.shape}")

    return image_normalisee

# --- EXÉCUTION ---
# Exemple d'appel pour créer la DataFrame
dataset_df = creer_dataframe_depuis_dataset(DATA_ROOT_DIR)

# Si vous voulez l'utiliser immédiatement pour le prétraitement :
if not dataset_df.empty:
    nb_images_a_afficher = 5  # Nombre d'images à traiter et afficher
    for i in range(min(nb_images_a_afficher, len(dataset_df))):
        image_a_traiter = dataset_df['filepaths'].iloc[i]
        print(f"\n--- Traitement de l'image {i+1}/{min(nb_images_a_afficher, len(dataset_df))} : {image_a_traiter} ---")
        tableau_numpy = charger_et_encoder_image(image_a_traiter)
        
        if tableau_numpy is not None:
            # Affiche le type et la forme du résultat
            print("Informations sur le résultat :")
            print(f"Type de données : {tableau_numpy.dtype}")
            print(f"Forme du tableau : {tableau_numpy.shape}")
            print("Contenu (premières valeurs du premier pixel) :")
            print(tableau_numpy[0, 0, :])  # Affiche les valeurs du premier pixel
            
            # Pour la première image, afficher TOUS les pixels (sauvegardé dans un fichier texte car trop volumineux pour la console)
            if i == 0:
                print("\nMatrice complète de l'image (TOUS les pixels sauvegardés dans un fichier texte) :")
                output_txt_file = 'matrice_image_1_complete.txt'
                np.savetxt(output_txt_file, tableau_numpy.reshape(-1, 3), fmt='%.6f')  # Sauvegarder en format texte
                print(f"Matrice complète sauvegardée dans : {output_txt_file}")
                print("Vous pouvez ouvrir ce fichier pour voir tous les pixels.")
            else:
                # Pour les autres images, affichage tronqué
                print("\nMatrice complète de l'image (tronquée pour lisibilité) :")
                np.set_printoptions(threshold=100, edgeitems=5)  # Limiter l'affichage
                print(tableau_numpy)
            
            # Sauvegarder la matrice complète dans un fichier
            output_file = f'matrice_image_{i+1}.npy'
            np.save(output_file, tableau_numpy)
            print(f"Matrice complète sauvegardée dans le fichier : {output_file}")
            print("Vous pouvez la charger plus tard avec : np.load('{output_file}')")
        else:
            print("Erreur lors du traitement de l'image.")
else:
    print("Aucune image trouvée dans le dataset.")
