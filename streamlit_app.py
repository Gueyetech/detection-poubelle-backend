import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import tempfile
import base64
import os
import traceback

from model import get_model, predict_image as predict_image_model

# Configuration de la page
st.set_page_config(
    page_title="Détection de Poubelles",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utiliser toujours le répertoire temporaire pour la compatibilité cloud
TEMP_DIR = Path(tempfile.gettempdir()) / "detection_poubelle"
UPLOAD_DIR = TEMP_DIR / "uploads"
RESULTS_DIR = TEMP_DIR / "results"
TEMP_VIDEO_DIR = TEMP_DIR / "temp_videos"

# Créer les dossiers
for directory in [UPLOAD_DIR, RESULTS_DIR, TEMP_VIDEO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Titre principal
st.title("Détection de Poubelles Pleines/Vides")


with st.sidebar:
    st.header("À propos")
    st.info("""
    Cette application utilise **YOLOv8** pour détecter si une poubelle est pleine ou vide.
    
    **Instructions:**
    1. Téléchargez une image depuis votre appareil
    2. Utilisez l'appareil photo de votre téléphone/webcam
    3. Analysez une vidéo
    4. Consultez les résultats de détection
    """)
    

    
    st.header("Informations du modèle")
    try:
        model = get_model()
        st.success("Modèle chargé")
        st.write(f"**Type:** YOLOv8")
        st.write(f"**Classes:** {list(model.names.values())}")
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

tab1, tab2, tab3 = st.tabs(["Image", "Vidéo", "Appareil Photo"])

with tab1:
    st.subheader("Téléchargez une image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats supportés: JPG, PNG, BMP"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        with col1:
            st.markdown("###  Image Originale")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with st.spinner("Analyse en cours..."):
            try:
                import uuid
                prediction_id = str(uuid.uuid4())
                file_path = UPLOAD_DIR / f"{prediction_id}_{uploaded_file.name}"
                
                # Sauvegarder le fichier
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Prédiction
                result = predict_image_model(str(file_path), str(RESULTS_DIR), prediction_id)
                
                with col2:
                    st.markdown("###  Résultats de Détection")
                    annotated_path = RESULTS_DIR / f"{prediction_id}_annotated.jpg"
                    if annotated_path.exists():
                        annotated_image = Image.open(annotated_path)
                        st.image(annotated_image, use_container_width=True)
                    else:
                        st.warning("Image annotée non trouvée")
                
                st.markdown("---")
                st.subheader(" Statistiques")
                
                summary = result["summary"]
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Total Détections", summary["total_detections"])
                
                for idx, (class_name, count) in enumerate(summary["class_counts"].items(), 1):
                    with metric_cols[idx % 3]:
                        st.metric(
                            class_name.replace("_", " ").title(),
                            count,
                            delta=None
                        )
                
                # Détails des détections
                if result["detections"]:
                    st.markdown("###  Détails des Détections")
                    for i, detection in enumerate(result["detections"], 1):
                        class_name = detection["class"]
                        confidence = detection["confidence"]
                        bbox = detection["bbox"]
                        
                        with st.expander(f"Détection #{i} - {class_name} ({confidence:.1%})"):
                            st.write(f"**Classe:** {class_name}")
                            st.write(f"**Confiance:** {confidence:.1%}")
                            st.write(f"**Coordonnées:** [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                
                st.success("Analyse terminée avec succès!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())

# Tab 2: Upload de vidéo
with tab2:
    st.subheader("Téléchargez une vidéo")
    st.warning("Le traitement vidéo peut prendre du temps selon la longueur de la vidéo.")
    
    uploaded_video = st.file_uploader(
        "Choisissez une vidéo...",
        type=["mp4", "avi", "mov"],
        help="Formats supportés: MP4, AVI, MOV"
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with st.spinner("Traitement de la vidéo en cours... Cela peut prendre plusieurs minutes."):
            try:
                model = get_model()
                confidence_threshold = 0.25
                
                input_path = TEMP_VIDEO_DIR / uploaded_video.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                cap = cv2.VideoCapture(str(input_path))
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                output_path = TEMP_VIDEO_DIR / f"output_{uploaded_video.name}"
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
                
                st.success("Traitement terminé!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Frames traitées", frame_count)
                    st.metric("Détections totales", total_detections)
                
                with col2:
                    avg_detections = round(total_detections / frame_count, 2) if frame_count > 0 else 0
                    st.metric("Détections/frame", avg_detections)
                
                st.markdown("### Statistiques par classe")
                for class_name, count in detection_stats.items():
                    st.write(f"**{class_name}:** {count}")
                
                st.markdown("### Vidéo Annotée")
                if output_path.exists():
                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                
                input_path.unlink()
                output_path.unlink()
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

with tab3:
    st.subheader("Capture avec votre appareil photo")
    st.info("Sur mobile : utilisez l'appareil photo de votre téléphone\n\n ur ordinateur : utilisez votre webcam")
    
    camera_photo = st.camera_input("Prenez une photo")
    
    if camera_photo is not None:
        # Analyse automatique
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
                    st.markdown("### Photo Originale")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("### Résultats")
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
                
                st.success("Analyse terminée!")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

# Footer
st.markdown("---")
st.caption("Développé avec par Gueye Tech | [GitHub](https://github.com/Gueyetech)")
