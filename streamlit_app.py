import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import tempfile
import base64

from model import get_model, predict_image as predict_image_model

# Configuration de la page
st.set_page_config(
    page_title="Détection de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Créer les dossiers nécessaires
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Titre principal
st.title("🗑️ Détection de Poubelles Pleines/Vides")

# Sidebar
with st.sidebar:
    st.header("ℹ️ À propos")
    st.info("""
    Cette application utilise **YOLOv8** pour détecter si une poubelle est pleine ou vide.
    
    **Instructions:**
    1. Téléchargez une image
    2. Ou utilisez votre webcam
    3. Consultez les résultats
    """)
    
    st.header("⚙️ Paramètres")
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    st.header("📊 Informations du modèle")
    try:
        model = get_model()
        st.success("✅ Modèle chargé")
        st.write(f"**Type:** YOLOv8")
        st.write(f"**Classes:** {list(model.names.values())}")
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Vidéo", "📸 Webcam"])

# Tab 1: Upload d'image
with tab1:
    st.subheader("Téléchargez une image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats supportés: JPG, PNG, BMP"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Afficher l'image originale
        with col1:
            st.markdown("### 📥 Image Originale")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Bouton de prédiction
        if st.button("🔍 Analyser l'image", key="analyze_image"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Sauvegarder temporairement
                    import uuid
                    prediction_id = str(uuid.uuid4())
                    file_path = UPLOAD_DIR / f"{prediction_id}_{uploaded_file.name}"
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Prédiction
                    result = predict_image_model(str(file_path), str(RESULTS_DIR), prediction_id)
                    
                    # Afficher l'image annotée
                    with col2:
                        st.markdown("### 🎯 Résultats de Détection")
                        annotated_path = RESULTS_DIR / f"{prediction_id}_annotated.jpg"
                        if annotated_path.exists():
                            annotated_image = Image.open(annotated_path)
                            st.image(annotated_image, use_container_width=True)
                    
                    # Afficher les statistiques
                    st.markdown("---")
                    st.subheader("📊 Statistiques")
                    
                    summary = result["summary"]
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Total Détections", summary["total_detections"])
                    
                    # Afficher par classe
                    for idx, (class_name, count) in enumerate(summary["class_counts"].items(), 1):
                        with metric_cols[idx % 3]:
                            st.metric(
                                class_name.replace("_", " ").title(),
                                count,
                                delta=None
                            )
                    
                    # Détails des détections
                    if result["detections"]:
                        st.markdown("### 🔍 Détails des Détections")
                        for i, detection in enumerate(result["detections"], 1):
                            class_name = detection["class"]
                            confidence = detection["confidence"]
                            bbox = detection["bbox"]
                            
                            with st.expander(f"Détection #{i} - {class_name} ({confidence:.1%})"):
                                st.write(f"**Classe:** {class_name}")
                                st.write(f"**Confiance:** {confidence:.1%}")
                                st.write(f"**Coordonnées:** [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                    
                    st.success("✅ Analyse terminée avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

# Tab 2: Upload de vidéo
with tab2:
    st.subheader("Téléchargez une vidéo")
    st.warning("⚠️ Le traitement vidéo peut prendre du temps selon la longueur de la vidéo.")
    
    uploaded_video = st.file_uploader(
        "Choisissez une vidéo...",
        type=["mp4", "avi", "mov"],
        help="Formats supportés: MP4, AVI, MOV"
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🔍 Analyser la vidéo", key="analyze_video"):
            with st.spinner("Traitement de la vidéo en cours... Cela peut prendre plusieurs minutes."):
                try:
                    model = get_model()
                    
                    # Sauvegarder la vidéo
                    temp_dir = Path("temp_videos")
                    temp_dir.mkdir(exist_ok=True)
                    
                    input_path = temp_dir / uploaded_video.name
                    with open(input_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                    
                    # Traiter la vidéo
                    cap = cv2.VideoCapture(str(input_path))
                    
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    output_path = temp_dir / f"output_{uploaded_video.name}"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    progress_bar = st.progress(0)
                    frame_count = 0
                    total_detections = 0
                    detection_stats = {}
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model(frame, conf=confidence_threshold, verbose=False)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)
                        
                        boxes = results[0].boxes
                        total_detections += len(boxes)
                        
                        for box in boxes:
                            class_name = results[0].names[int(box.cls[0])]
                            detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                    
                    cap.release()
                    out.release()
                    
                    # Afficher les résultats
                    st.success("✅ Traitement terminé!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Frames traitées", frame_count)
                        st.metric("Détections totales", total_detections)
                    
                    with col2:
                        avg_detections = round(total_detections / frame_count, 2) if frame_count > 0 else 0
                        st.metric("Détections/frame", avg_detections)
                    
                    st.markdown("### 📊 Statistiques par classe")
                    for class_name, count in detection_stats.items():
                        st.write(f"**{class_name}:** {count}")
                    
                    # Afficher la vidéo annotée
                    st.markdown("### 🎬 Vidéo Annotée")
                    if output_path.exists():
                        with open(output_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                    
                    # Nettoyer
                    input_path.unlink()
                    output_path.unlink()
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

# Tab 3: Webcam
with tab3:
    st.subheader("Capture depuis la webcam")
    st.info("📸 Prenez une photo avec votre webcam pour l'analyser")
    
    camera_photo = st.camera_input("Prenez une photo")
    
    if camera_photo is not None:
        if st.button("🔍 Analyser la photo", key="analyze_webcam"):
            with st.spinner("Analyse en cours..."):
                try:
                    import uuid
                    prediction_id = str(uuid.uuid4())
                    
                    # Convertir en image
                    image = Image.open(camera_photo)
                    file_path = UPLOAD_DIR / f"{prediction_id}_webcam.jpg"
                    image.save(file_path)
                    
                    # Prédiction
                    result = predict_image_model(str(file_path), str(RESULTS_DIR), prediction_id)
                    
                    # Afficher résultats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📥 Photo Originale")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🎯 Résultats")
                        annotated_path = RESULTS_DIR / f"{prediction_id}_annotated.jpg"
                        if annotated_path.exists():
                            st.image(Image.open(annotated_path), use_container_width=True)
                    
                    # Statistiques
                    st.markdown("---")
                    summary = result["summary"]
                    
                    metric_cols = st.columns(len(summary["class_counts"]) + 1)
                    with metric_cols[0]:
                        st.metric("Total", summary["total_detections"])
                    
                    for idx, (class_name, count) in enumerate(summary["class_counts"].items(), 1):
                        with metric_cols[idx]:
                            st.metric(class_name.replace("_", " ").title(), count)
                    
                    st.success("✅ Analyse terminée!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

# Footer
st.markdown("---")
st.caption("Développé avec ❤️ par Gueye Tech | YOLOv8 Detection API")
st.caption("[GitHub](https://github.com/Gueyetech)")
