import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from PIL import Image

# CSS pour traduire l'interface
st.markdown("""
<style>
[data-testid="stFileUploadDropzone"] span {
    visibility: hidden;
}
[data-testid="stFileUploadDropzone"] span::before {
    content: "Glissez-déposez votre photo ici";
    visibility: visible;
}
[data-testid="stFileUploadDropzone"] button::before {
    content: "Parcourir";
}
[data-testid="stFileUploadDropzone"] button span {
    display: none;
}
[data-testid="stFileUploadDropzone"] small {
    visibility: hidden;
}
[data-testid="stFileUploadDropzone"] small::before {
    content: "Limite 200MB • JPG, PNG";
    visibility: visible;
}
</style>
""", unsafe_allow_html=True)

# 1. Chargement
@st.cache_resource
def load_assets():
    model = load_model('model_transfer_learning.keras')
    with open('dog_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    return model, labels

model, labels = load_assets()

SEUIL_CONFIANCE = 70

st.title(" Le Refuge : Identification de races de chiens")

# 2. Interface de test
st.markdown("**Glissez-déposez votre photo ici** ou cliquez pour parcourir vos fichiers *(formats acceptés : JPG, PNG)*")
file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if file:
    img = Image.open(file)
    st.image(img, caption="Photo à analyser", width=300)
    
    # 3. Préparation
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    # Normalisation simple (comme dans l'entraînement MobileNetV2)
    img_final = img_batch / 255.0
    # 4. Prédiction
    preds = model.predict(img_final)
    classe_id = np.argmax(preds)
    nom_race = labels[classe_id]
    confiance = np.max(preds) * 100

    # 5. Afficher le détail des probabilités
    st.write("**Détail des probabilités :**")
    for i, label in enumerate(labels):
        st.write(f"- {label} : {preds[0][i]*100:.1f}%")

    # 6. Décision avec seuil (DANS le if file: !)
    if nom_race == "Autre":
        st.info(" Ce chien ne fait pas encore parti de notre base de données (Golden Retriever, Eskimo_dog, Chihuahua).")
    elif confiance >= SEUIL_CONFIANCE:
        st.success(f" C'est un **{nom_race}** ({confiance:.1f}% de confiance)")
    else:
        st.warning(f" Résultat incertain : **{nom_race}** ({confiance:.1f}% de confiance)")