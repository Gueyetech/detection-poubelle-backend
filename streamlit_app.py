import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os

from ultralytics import YOLO

# Configuration de la page
st.set_page_config(
    page_title="Détection de Poubelles",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f" Modèle introuvable : {path}")
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f" Erreur chargement modèle : {e}")
        return None

model = load_model()

# Header principal
st.markdown("""
<div style='text-align: center; padding:0.5rem;background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
     border-radius: 20px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);'>
    <h1 style='font-size: 3rem; margin: 0; font-weight: 800;'> Détection de Poubelles pleine ou vide </h1>
</div>
""", unsafe_allow_html=True)

# Tabs pour les différentes options
tab1, tab2, tab3 = st.tabs(["Image", "Vidéo", "Appareil Photo"])

# TAB 1: Upload Image
with tab1:
    st.markdown("###  Téléchargez une image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
        help="Formats supportés: JPG, PNG, BMP"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("####  Image Originale")
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f" Erreur de chargement: {e}")
                image = None
        
        if image and st.button(" Analyser l'image", key="analyze_image", type="primary"):
            if model is None:
                st.error("Modèle non disponible")
            else:
                with st.spinner(" Analyse en cours..."):
                    try:
                        # Conversion
                        img_array = np.array(image)
                        
                        # Prédiction
                        results = model.predict(img_array, conf=0.25, imgsz=640)
                        
                        if results and len(results) > 0:
                            r = results[0]
                            
                            # Image annotée
                            annotated = r.plot()
                            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            
                            with col2:
                                st.markdown("####  Résultats")
                                st.image(annotated, use_container_width=True)
                            
                            # Statistiques
                            dets = r.boxes
                            if dets and len(dets) > 0:
                                st.markdown("---")
                                st.markdown("###  Statistiques")
                                
                                metric_cols = st.columns(3)
                                with metric_cols[0]:
                                    st.metric("Total Détections", len(dets))
                                
                                # Compter par classe
                                class_counts = {}
                                for box in dets:
                                    cls_idx = int(box.cls[0])
                                    cls_name = model.names[cls_idx]
                                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                
                                for idx, (cls_name, count) in enumerate(class_counts.items(), 1):
                                    with metric_cols[idx % 3]:
                                        st.metric(cls_name.replace("_", " ").title(), count)
                                
                                # Détails
                                st.markdown("###  Détails des Détections")
                                for i, box in enumerate(dets, 1):
                                    cls_idx = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    cls_name = model.names[cls_idx]
                                    bbox = box.xyxy[0].cpu().numpy()
                                    
                                    with st.expander(f"Détection #{i} - {cls_name} ({conf:.1%})"):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write(f"**Classe:** {cls_name}")
                                            st.write(f"**Confiance:** {conf:.1%}")
                                        with col_b:
                                            st.write(f"**Coordonnées:** [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                                        
                                        # Barre de confiance
                                        st.progress(conf)
                                
                                st.success(" Analyse terminée avec succès!")
                            else:
                                st.info(" Aucune détection trouvée")
                        else:
                            st.warning(" Aucun résultat")
                            
                    except Exception as e:
                        st.error(f" Erreur lors de l'analyse: {str(e)}")
                        import traceback
                        with st.expander("Détails de l'erreur"):
                            st.code(traceback.format_exc())

# TAB 2: Vidéo
with tab2:
    st.markdown("###  Analyse Vidéo")
    st.info(" Le traitement vidéo peut prendre du temps selon la longueur")
    
    uploaded_video = st.file_uploader(
        "Choisissez une vidéo",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        help="Formats supportés: MP4, AVI, MOV"
    )
    
    if uploaded_video:
        st.video(uploaded_video)
        
        if st.button(" Analyser la vidéo", key="analyze_video", type="primary"):
            if model is None:
                st.error(" Modèle non disponible")
            else:
                with st.spinner(" Traitement en cours..."):
                    try:
                        # Créer dossier temp
                        temp_dir = Path(tempfile.gettempdir()) / "detection_videos"
                        temp_dir.mkdir(exist_ok=True)
                        
                        input_path = temp_dir / uploaded_video.name
                        with open(input_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                        
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
                            
                            results = model(frame, conf=0.25, verbose=False)
                            annotated_frame = results[0].plot()
                            out.write(annotated_frame)
                            
                            boxes = results[0].boxes
                            total_detections += len(boxes)
                            
                            for box in boxes:
                                cls_idx = int(box.cls[0])
                                class_name = model.names[cls_idx]
                                detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
                            
                            frame_count += 1
                            progress_bar.progress(frame_count / total_frames)
                        
                        cap.release()
                        out.release()
                        
                        st.success(" Traitement terminé!")
                        
                        # Statistiques
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Frames traitées", frame_count)
                            st.metric("Détections totales", total_detections)
                        with col2:
                            avg = round(total_detections / frame_count, 2) if frame_count > 0 else 0
                            st.metric("Détections/frame", avg)
                        
                        if detection_stats:
                            st.markdown("###  Statistiques par classe")
                            for class_name, count in detection_stats.items():
                                st.write(f"**{class_name}:** {count}")
                        
                        # Vidéo annotée
                        if output_path.exists():
                            st.markdown("###  Vidéo Annotée")
                            with open(output_path, "rb") as f:
                                st.video(f.read())
                            
                            # Nettoyage
                            input_path.unlink(missing_ok=True)
                            output_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f" Erreur: {str(e)}")
                        import traceback
                        with st.expander("Détails"):
                            st.code(traceback.format_exc())

# TAB 3: Appareil Photo/Webcam
with tab3:
    st.markdown("###  Capture avec votre appareil photo")
    st.info(" Sur mobile : utilisez l'appareil photo de votre téléphone\n\n Sur ordinateur : utilisez votre webcam")
    
    camera_photo = st.camera_input("Prenez une photo")
    
    if camera_photo:
        if model is None:
            st.error(" Modèle non disponible")
        else:
            with st.spinner(" Analyse en cours..."):
                try:
                    # Convertir en image
                    image = Image.open(camera_photo).convert("RGB")
                    
                    # Conversion
                    img_array = np.array(image)
                    
                    # Prédiction
                    results = model.predict(img_array, conf=0.25, imgsz=640)
                    
                    if results and len(results) > 0:
                        r = results[0]
                        
                        # Image annotée
                        annotated = r.plot()
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        
                        # Afficher résultats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("####  Photo Originale")
                            st.image(image, use_container_width=True)
                        
                        with col2:
                            st.markdown("####  Résultats")
                            st.image(annotated, use_container_width=True)
                        
                        # Statistiques
                        dets = r.boxes
                        if dets and len(dets) > 0:
                            st.markdown("---")
                            
                            class_counts = {}
                            for box in dets:
                                cls_idx = int(box.cls[0])
                                cls_name = model.names[cls_idx]
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                            
                            metric_cols = st.columns(len(class_counts) + 1)
                            with metric_cols[0]:
                                st.metric("Total", len(dets))
                            
                            for idx, (cls_name, count) in enumerate(class_counts.items(), 1):
                                with metric_cols[idx]:
                                    st.metric(cls_name.replace("_", " ").title(), count)
                            
                            st.success(" Analyse terminée!")
                        else:
                            st.info(" Aucune détection trouvée")
                    else:
                        st.warning(" Aucun résultat")
                        
                except Exception as e:
                    st.error(f" Erreur: {str(e)}")
                    import traceback
                    with st.expander("Détails"):
                        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1.5rem;'>
    <p style='color: #64748b; margin: 0;'>Développé par <strong>Gueye Tech</strong> | 
    <a href='https://github.com/Gueyetech' style='color: #3b82f6; text-decoration: none;'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
