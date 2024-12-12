import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# CSS pour fixer la position du widget d'upload
st.markdown("""
    <style>
    .uploadedFile {
        position: fixed;
        top: 4rem;
        left: 1rem;
        z-index: 999;
        width: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

st.title("Détection de voitures avec YOLOv11")

# Interface pour uploader l'image (maintenant en haut)
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], key="uploadedFile")

# Création des onglets pour choisir le modèle
tab1, tab2 = st.tabs(["Premier modèle", "Deuxième modèle"])

with tab1:
    st.header("Premier modèle")
    model_a = load_model('models/model_detector/best.pt')
    st.write("Utilisant YOLOv11m")

with tab2:
    st.header("Deuxième modèle")
    model_b = load_model('models/vehicle_detector/best.pt')
    st.write("Utilisant YOLOv11m")

if uploaded_file is not None:
    # Charger et afficher l'image originale
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploadée', use_container_width =True)

    # Effectuer la prédiction avec les deux modèles
    results_a = model_a(image)
    results_b = model_b(image)

    # Afficher les résultats dans les onglets respectifs
    with tab1:
        st.image(results_a[0].plot(), caption='Résultat de la détection avec Premier modèle', use_container_width =True)
        st.subheader("Détails des détections (Premier modèle):")
        for r in results_a:
            for box in r.boxes:
                st.write(f"Classe: {r.names[int(box.cls)]}, Confiance: {box.conf.item():.2f}")

    with tab2:
        st.image(results_b[0].plot(), caption='Résultat de la détection avec Deuxième modèle', use_container_width =True)
        st.subheader("Détails des détections (Deuxième modèle):")
        for r in results_b:
            for box in r.boxes:
                st.write(f"Classe: {r.names[int(box.cls)]}, Confiance: {box.conf.item():.2f}")

    # Afficher des informations sur les modèles utilisés
    st.subheader("Informations sur les modèles:")
    st.write("Premier modèle: YOLOv11 medium avec détection de modèles de voitures")
    st.write("Deuxième modèle: YOLOv11 medium avec détection de voitures/motos")
