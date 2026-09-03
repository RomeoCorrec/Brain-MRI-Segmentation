# Runbook — obtenir les résultats (entraînement Colab GPU)

Objectif : produire les métriques, courbes et figures pour la présentation.
Durée : ~1 h (early stopping — UNet et YOLO s'arrêtent quand la validation
stagne : `--patience` 12 / 15 epochs, plafond 100). Compter ~25-40 min par modèle.

> `mlflow` est volontairement épinglé `<3` dans le notebook : le serveur Docker
> tourne en 2.14.3 et un client 3.x échoue en 404 sur `/api/2.0/mlflow/logged-models`
> au moment d'enregistrer les modèles dans le registry.

## A. Sur le PC local — démarrer le suivi d'expériences

```bash
cd Brain-MRI-Segmentation
# .env doit contenir POSTGRES_PASSWORD et NGROK_AUTHTOKEN
docker compose up -d --build
docker compose ps                 # les 3 services doivent être "running"/"healthy"
```

Récupérer l'URL publique ngrok :

```bash
docker compose logs ngrok | grep -o 'https://[^ ]*\.ngrok-free\.app' | head -1
```

ou via http://localhost:4040. MLflow UI en local : http://localhost:5000

> Garder ce terminal ouvert pendant toute la session Colab.

## B. Sur Google Colab — `notebooks/colab_training.ipynb`

Runtime : **GPU** (Exécution → Modifier le type d'exécution → T4).

| Cellule | Action |
|---|---|
| 2 | Coller l'URL ngrok dans `TRACKING_URI`. Laisser les hyperparamètres par défaut. |
| 4 | Clone du repo. **Pousser d'abord la branche à jour sur GitHub** (voir C). |
| 5 | Installe les dépendances. **Puis : Exécution → Redémarrer la session** (obligatoire, conflit numpy). Ne pas rejouer cette cellule après le redémarrage. |
| 2 (bis) | **Après le redémarrage**, ré-exécuter la cellule 2 (les variables Python sont perdues). |
| 6 | Reprendre ici. Vérifie `GPU disponible : True`. |
| 8–9 | Upload `kaggle.json` (kaggle.com/settings → API → Create New Token), télécharge TCGA. |
| 12 | Vérifie la connexion MLflow (doit lister les expériences, vide au 1er run). |
| 14 | Entraîne UNet (~25 min sur T4). |
| 16 | Convertit les masques au format polygones YOLO. |
| 17 | Entraîne YOLOv8n-seg (~10 min). |
| 19 | Vérifie les runs + modèles enregistrés dans le registry. |
| 21–25 | Section 7 : courbes, tableau, prédictions visuelles. |
| 27–31 | Section 8 : évaluation métrique commune (Dice/IoU pixel) + export ONNX + latence. |

## C. Pré-requis : pousser le code à jour

Le notebook clone `github.com/RomeoCorrec/Brain-MRI-Segmentation`. Les ajouts
(`src/evaluate.py`, section 8) doivent être sur `main` avant de lancer la cellule 4 :

```bash
git push origin main
```

## D. Récupérer les livrables pour les slides

Depuis Colab (`Fichiers` dans le panneau latéral, ou `files.download(...)`) :

| Fichier | Usage slide |
|---|---|
| `/content/comparison_curves.png` | Slide 5 — courbes d'entraînement |
| `/content/eval/eval_table.md` | Slide 5 — tableau métrique commune |
| `/content/eval/eval_summary.json` | chiffres bruts (latence, params, ONNX) |
| `/content/comparison_predictions.png` | Slide 6 — prédictions qualitatives |
| `/content/eval/qualitative.png` | Slide 6 — variante (avec IoU par coupe) |
| `/content/eval/dice_distribution.png` | Slide 6 — histogramme Dice par coupe |

Captures d'écran de la **MLflow UI** (http://localhost:5000) pour le slide 4 :
- vue « Compare runs » des deux expériences
- le Model Registry avec `unet-brain-mri` et `yolov8-brain-mri`

## E. Arrêter la stack

```bash
docker compose down          # garde les volumes (données MLflow conservées)
```

## Points de vigilance

- L'URL ngrok change à chaque `docker compose up` → recoller dans la cellule 2.
- Si Colab perd le GPU en cours de route, les runs déjà loggés dans MLflow sont conservés.
- YOLO est entraîné sur un split par image (`prepare_yolo_dataset.py`), différent du
  split par patient de UNet. `evaluate.py` le signale (`summary["note"]`). Le corriger
  = réutiliser le split par patient — bon point « analyse des écarts entre jeux de
  données » pour l'entretien, à mentionner comme limite identifiée.
