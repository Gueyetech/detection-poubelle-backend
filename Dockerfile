FROM python:3.11-slim

WORKDIR /app

# Installer dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p uploads results temp_videos

# Exposer le port (Render utilisera la variable $PORT)
EXPOSE 8000

# Lancer l'API FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
