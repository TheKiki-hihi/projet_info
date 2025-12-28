# Projet Python – Classification de cellules sanguines

Ce projet a été réalisé tout au long du semestre dans le cadre de notre 3ème année en licence.
L’objectif principal était de créer un projet mêlant programmation Python, traitement d’images et  faire apprendre une LML (large model learning).

Le projet consiste à classifier des images de cellules sanguines afin de déterminer si une
cellule est saine ou atteinte de leucémie.

Contexte du projet

Dans le domaine médical, l’analyse des cellules sanguines est un élément clé pour le diagnostic
de certaines maladies comme, dans le cas de notre projet, la leucémie. L’idée est donc de comprendre comment
des outils informatiques peuvent être utilisés pour automatiser ce type de tâche à partir
d’images.

Les données utilisées proviennent de la plateforme Kaggle et sont déjà classées en deux
catégories, ce qui nous a permis de travailler sur un cas réel de classification supervisée :

https://www.kaggle.com/datasets/andrewmvd/leukemia-classification 

Il est nécessaire de télécharger le dataset complet et de le décompresser sur son
ordinateur.

Organisation des données sur le PC

Une fois le dataset téléchargé, les dossiers doivent être placés dans le projet en
respectant l’organisation attendue par le programme.

Le dossier des données doit être organisée de la manière suivante :
```
C-NMC_Leukemia/
├── training_data/
│   └── fold_0/
│       ├── hem/
│       └── all/
├── validation_data/
│   ├── C-NMC_test_prelim_phase_data/
│   └── C-NMC_test_prelim_phase_data_labels.csv
└── testing_data/
    └── C-NMC_test_final_phase_data/
```

Après avoir téléchargé le projet en .zip depuis Github et organisé le dataset, il faut lancer le script principal : python main.py.


Données utilisées

Les données sont composées d’images de cellules sanguines réparties en deux classes :

- HEM : cellules saines  
- ALL : cellules atteintes de leucémie  

Elles sont organisées en trois ensembles :
- un ensemble d’entraînement
- un ensemble de validation
- un ensemble de test

Cette organisation permet d’entraîner le modèle, de l’évaluer, puis de tester ses performances
sur des images qu’il n’a jamais vues.

 Objectifs du projet

À travers ce projet, nous avons cherché à :
- comprendre comment préparer des données à partir d’images
- manipuler des matrices et des fichiers avec Python
- entraîner un modèle de classification
- analyser les résultats obtenus
- avoir une vision globale d’un projet de data science du début à la fin

Outils et bibliothèques utilisés

Le projet a été réalisé en Python en utilisant plusieurs bibliothèques :
- NumPy pour la manipulation des données
- OpenCV pour le traitement des images
- Pandas pour la gestion des données
- PyTorch et Torchvision pour la création et l’entraînement du modèle
- Scikit-learn pour certaines analyses complémentaires
- Matplotlib pour la visualisation

Organisation du projet

Le projet est découpé en plusieurs fichiers afin de séparer les différentes étapes du travail :

- `main.py` : fichier principal permettant de lancer le projet
- `dataset.py` : gestion et chargement des données
- `model.py` : définition du modèle de classification
- `train.py` : entraînement du modèle
- `predict.py` : prédiction sur les images de test
- `clustering.py` : analyse complémentaire des données
- `README.md` : description du projet

 Fonctionnement général

Le fonctionnement du projet se fait en plusieurs étapes.

Dans un premier temps, les images sont chargées depuis les dossiers du dataset. Elles sont
ensuite redimensionnées et normalisées afin d’être exploitables par le modèle.

Les données sont ensuite séparées en ensembles d’entraînement, de validation et de test.
Le modèle est entraîné à partir des données d’entraînement, puis évalué sur les données de
validation.

Une fois l’entraînement terminé, le modèle est utilisé pour effectuer des prédictions sur
les images de test. Les résultats obtenus sont enregistrés dans un fichier afin de pouvoir
être analysés.

 Modèle utilisé

Le modèle utilisé pour la classification est un réseau de neurones convolutionnel basé sur
l’architecture ResNet18. Ce choix a été fait afin de travailler avec un modèle déjà existant
et reconnu, tout en comprenant son fonctionnement et son adaptation à notre problème.

 Résultats

Pour chaque image de test, le modèle fournit :
- une classe prédite (cellule saine ou malade)
- un score de confiance associé à cette prédiction

Ces résultats permettent d’avoir une première estimation de la performance du modèle et de
mieux comprendre ses limites.

 Conclusion

Ce projet nous a permis de mettre en pratique de nombreuses notions vues durant le semestre
et de mieux comprendre les différentes étapes nécessaires à la réalisation d’un projet de
classification d’images. Il nous a également permis de prendre conscience de l’importance
de la préparation des données et du choix du modèle pour obtenir des résultats cohérents.

Kily Majani, Kilpéric Courilleau, Anael Atmani
