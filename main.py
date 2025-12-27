from src.dataset import preparer_donnees
from src.train import entrainer_modele
from src.predict import lancer_predictions

if __name__ == "__main__":
    preparer_donnees()
    entrainer_modele()
    lancer_predictions()
