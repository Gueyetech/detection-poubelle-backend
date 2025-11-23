# Détection de Poubelles - Guide de Déploiement Render

## 🚀 Déploiement sur Render

Ce projet peut être déployé sur [Render](https://render.com) avec deux services :
1. **Application Streamlit** - Interface utilisateur web
2. **API FastAPI** - API REST pour les prédictions

### Option 1 : Déploiement avec render.yaml (Recommandé)

Le fichier `render.yaml` configure automatiquement les deux services.

#### Étapes :

1. **Créez un compte sur [Render](https://render.com)**

2. **Créez un nouveau "Blueprint"**
   - Allez sur [Dashboard Render](https://dashboard.render.com)
   - Cliquez sur "New" → "Blueprint"
   - Connectez votre dépôt GitHub
   - Sélectionnez le repository `detection-poubelle-backend`
   - Render détectera automatiquement le fichier `render.yaml`

3. **Confirmez la configuration**
   - Vérifiez que les deux services apparaissent :
     - `detection-poubelle-streamlit` (Application Streamlit)
     - `detection-poubelle-api` (API FastAPI)
   - Cliquez sur "Apply"

4. **Attendez le déploiement**
   - Le build prend environ 5-10 minutes
   - Les deux services seront disponibles avec leurs URLs uniques

### Option 2 : Déploiement Manuel

#### Pour l'Application Streamlit :

1. Sur le [Dashboard Render](https://dashboard.render.com), cliquez "New" → "Web Service"
2. Connectez votre repository GitHub
3. Configurez :
   - **Name**: `detection-poubelle-streamlit`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: Free
4. Cliquez sur "Create Web Service"

#### Pour l'API FastAPI :

1. Cliquez "New" → "Web Service"
2. Connectez le même repository
3. Configurez :
   - **Name**: `detection-poubelle-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Cliquez sur "Create Web Service"

### 📝 Notes Importantes

- **Plan gratuit** : Les deux services peuvent tourner sur le plan gratuit de Render
- **Temps de build** : Premier déploiement ~5-10 minutes (téléchargement du modèle YOLOv8)
- **Sleep automatique** : Sur le plan gratuit, les services s'endorment après 15 min d'inactivité
- **Réveil** : Premier accès après sleep prend ~30 secondes

### 🔗 URLs après déploiement

Une fois déployés, vos services seront accessibles à :
- **Streamlit**: `https://detection-poubelle-streamlit.onrender.com`
- **API**: `https://detection-poubelle-api.onrender.com`
- **Documentation API**: `https://detection-poubelle-api.onrender.com/docs`

### 🛠️ Variables d'Environnement

Aucune variable d'environnement spéciale n'est requise. Le modèle YOLOv8 est téléchargé automatiquement depuis GitHub au premier démarrage.

### 📊 Monitoring

- Consultez les logs en temps réel depuis le Dashboard Render
- Les erreurs de build/démarrage sont affichées dans les logs
- Utilisez `/api/health` pour vérifier l'état de l'API

### 🔄 Redéploiement

Render redéploie automatiquement à chaque push sur la branche `main` de GitHub.

Pour forcer un redéploiement manuel :
1. Allez dans le service sur Render Dashboard
2. Cliquez sur "Manual Deploy" → "Deploy latest commit"

### ⚠️ Limitations du Plan Gratuit

- 750 heures/mois de runtime
- 512 MB RAM
- Sleep après 15 minutes d'inactivité
- Bande passante limitée

Pour des performances production, envisagez un plan payant.

### 🐛 Dépannage

**Erreur de build** :
- Vérifiez les logs de build
- Assurez-vous que `requirements.txt` est correct
- Python 3.11 est utilisé par défaut

**Service ne démarre pas** :
- Vérifiez les logs de démarrage
- Le modèle `best.pt` doit se télécharger automatiquement
- Vérifiez que le port `$PORT` est bien utilisé

**Performances lentes** :
- Normal sur le plan gratuit après sleep
- Considérez un plan payant pour éviter le sleep

### 📞 Support

Pour plus d'informations, consultez la [documentation Render](https://render.com/docs).
