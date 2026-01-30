Ce projet a été réalisé bénévolement pour l'association de protection animale "Le Refuge". L'objectif est d'automatiser l'indexation de leur base de données d'images en classant automatiquement les chiens par race grâce au Deep Learning.

1. Installation

# Cloner le projet
git clone https://github.com/ton-pseudo/dog-breed-classifier.git
cd dog-breed-classifier

# Installer les dépendances
pip install -r requirements.txt

2. Utilisation

Lancer via le terminal Bash la commande suivante : streamlit run app.py
Puis ouvrir si cela ne se fait pas automatiquement http://localhost:8501 dans votre navigateur

3. Notebooks

 1. Préparation des données
 2. Modèle CNN maison + test avec dropout
 3. Transfer Learning (VGG16 & MobileNetV2)  

4. Structure du projet

dog-breed-classifier/
├── app.py                          # Application Streamlit
├── Images/                         # Dataset Stanford Dogs
├── Goldstein_Ludivine_1_pretraitement.ipynb
├── Goldstein_Ludivine_2_modele_cnn_personnel.ipynb
├── Goldstein_Ludivine_3_transfer_learning.ipynb
├── model_transfer_learning.keras   # Modèle MobileNetV2 entraîné
├── dog_labels.pkl                  # Labels des classes
├── donnees_preparees.pkl           # Données prétraitées
├── resultats_maison.pkl            # Résultats CNN maison
├── requirements.txt                # Dépendances Python
└── README.md
   
