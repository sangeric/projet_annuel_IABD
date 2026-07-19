# Felidae Classifier

Ce dépôt contient un classifieur de félins écrit en Rust (compilé en bibliothèque partagée) ainsi qu'un client Python pour les tests et l'application de démonstration.

## Structure du projet

```
projet_annuel_IABD_code/
├── felidae_classifier/     # Code source Rust
├── python_client/           # Notebooks de test (Jupyter)
├── requirements.txt         # Dépendances Python
└── app.py                   # Application de démonstration
```

## 1. Build de la bibliothèque Rust

Se placer dans le répertoire du projet Rust puis compiler en mode release :

```bash
cd projet_annuel_IABD_code/felidae_classifier
cargo build --release
```

Cette commande génère le fichier `.so` (Linux) ou `.dll` (Windows) dans le dossier `projet_annuel_IABD_code\felidae_classifier\target\release\`.

## 2. Mise en place de l'environnement Python (tests)

Les cas de tests se trouvent dans `projet_annuel_IABD/python_client`. Créer un environnement virtuel et installer les dépendances (le fichier `requirements.txt` se trouve à la racine de `projet_annuel_IABD`) :

```bash
cd projet_annuel_IABD/python_client
python -m venv .venv

# Activer l'environnement virtuel python
# Windows :
.venv\Scripts\activate.ps1
# Linux / macOS :
source .venv/bin/activate

# Installation des dépendances
pip install -r ../requirements.txt
```

### Lancer les notebooks de test

Une fois l'environnement activé, lancer Jupyter :

```bash
jupyter notebook
```

Puis ouvrir dans le navigateur l'adresse indiquée dans le terminal (généralement `http://localhost:8888/...`).

## 3. Lancer l'application

Se placer à la racine du projet, activer le `.venv` (créé précédemment) puis lancer l'application :

```bash
cd projet_annuel_IABD_code

# Activer l'environnement virtuel python
# Windows :
python_client\.venv\Scripts\activate.ps1
# Linux / macOS :
source python_client/.venv/bin/activate

python app.py
```

## Résumé rapide

| Étape | Répertoire | Commande |
|-------|------------|----------|
| Build Rust | `felidae_classifier` | `cargo build --release` |
| Setup venv + deps | `python_client` | `python -m venv .venv` puis `pip install -r ../requirements.txt` |
| Tests (notebooks) | `python_client` | `jupyter notebook` |
| Application | `projet_annuel_IABD_code` (racine) | `python app.py` (avec `.venv` activé) |
