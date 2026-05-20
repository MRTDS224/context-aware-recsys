# 🎯 Context-Aware Recommender System
### Projet Data Science — Neural Collaborative Filtering + Signaux Contextuels

---

## 🗺️ Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture orientée Data Science](#architecture-orientée-data-science)
3. [Par où commencer ? (Workflow)](#par-où-commencer--workflow)
4. [Datasets](#datasets)
5. [Les features contextuelles expliquées](#les-features-contextuelles-expliquées)
6. [Les modèles, simplement](#les-modèles-simplement)
7. [Comment évaluer les résultats ?](#comment-évaluer-les-résultats-)
8. [Ablation Study — tester l'impact de chaque contexte](#ablation-study--tester-limpact-de-chaque-contexte)
9. [Interprétabilité](#interprétabilité)
10. [Résultats attendus](#résultats-attendus)
11. [Installation](#installation)
12. [Reproductibilité](#reproductibilité)
13. [Références](#références)

---

## Vue d'ensemble

Ce projet construit un **système de recommandation intelligent** qui apprend non seulement *ce qu'un utilisateur aime*, mais **quand et comment** il interagit. L'idée centrale : le contexte (heure, session, appareil) améliore significativement la qualité des recommandations.

### Le problème en termes simples

Les systèmes classiques (Netflix de base, Amazon vers 2005) ignorent **quand** tu regardes ou achètes :

| Situation | Ce qu'un bon système devrait comprendre |
|-----------|----------------------------------------|
| Vendredi soir vs. dimanche matin | Films d'action ≠ documentaires calmes |
| Session longue vs. session courte | Exploration vs. achat ciblé |
| Mobile vs. Desktop | Intentions différentes |

### L'approche

```
[Qui ?]     User ID ──────────────────────────────────────────────┐
[Quoi ?]    Item ID ──────────────────────────────────────────────┤──► Score de recommandation
[Quand ?]   Heure, jour ──► Encodeur ────────────────────────────┤
[Comment ?] Session, device ──► Contextuel ──────────────────────┘
```

**Objectif mesurable** : montrer que le modèle *avec contexte* obtient de meilleurs scores (NDCG, Hit Rate) que le modèle *sans contexte*.

---

## Architecture orientée Data Science

> 💡 **Principe clé pour un débutant** : en data science, les **notebooks sont le point d'entrée**. On explore d'abord, on code ensuite. Le dossier `src/` ne contient que des fonctions utilitaires pour alléger les notebooks — pas besoin d'y toucher pour comprendre le projet.

```
context-aware-recsys/
│
├── 📓 notebooks/               ← COMMENCE ICI — c'est le cœur du projet
│   ├── 01_exploration_donnees.ipynb     # Comprendre les datasets (EDA)
│   ├── 02_feature_engineering.ipynb     # Créer les features contextuelles
│   ├── 03_modele_baseline.ipynb         # NCF sans contexte (référence)
│   ├── 04_modele_avec_contexte.ipynb    # NCF + contexte (modèle principal)
│   ├── 05_evaluation_comparaison.ipynb  # Comparer baseline vs contexte
│   └── 06_interpretabilite.ipynb        # Pourquoi le modèle décide quoi ?
│
├── 📂 data/
│   ├── raw/                    # Données brutes téléchargées (ne pas modifier)
│   │   ├── movielens/          # ML-100K et ML-1M
│   │   └── retailrocket/       # events.csv, item_properties...
│   └── processed/              # Données nettoyées — générées par notebook 02
│
├── 🐍 src/                     # Code Python de support (fonctions utilitaires)
│   ├── data_utils.py           # Chargement et nettoyage des données
│   ├── context_features.py     # Extraction des features contextuelles
│   ├── models.py               # Architectures NCF et NCF+Contexte
│   └── metrics.py              # NDCG@K, Hit Rate@K, MRR
│
├── 📊 results/                 # Sorties du projet — générées automatiquement
│   ├── figures/                # Graphiques et visualisations
│   ├── tables/                 # CSV des résultats d'expériences
│   └── models/                 # Modèles entraînés sauvegardés (.pt)
│
├── pyproject.toml              # Dépendances Python
├── .python-version             # Python 3.12
└── README.md
```

### Pourquoi cette structure est adaptée à la data science ?

| Élément | Rôle | Analogie |
|---------|------|----------|
| `notebooks/` | Exploration + expérimentation | Ton cahier de labo |
| `data/raw/` | Données sources intouchables | Tes données terrain |
| `data/processed/` | Données prêtes à l'emploi | Données nettoyées en labo |
| `src/` | Fonctions réutilisables | Ta boîte à outils |
| `results/` | Tout ce que tu produis | Tes résultats de recherche |

---

## Par où commencer ? (Workflow)

Suis les notebooks **dans l'ordre numérique**. Chacun produit des fichiers utilisés par le suivant.

```
📓 01 → explorer les données
         ↓
📓 02 → créer les features → sauvegarde dans data/processed/
         ↓
📓 03 → entraîner le baseline → sauvegarde dans results/models/
         ↓
📓 04 → entraîner le modèle contextuel → sauvegarde dans results/models/
         ↓
📓 05 → comparer les deux → sauvegarde dans results/tables/ et figures/
         ↓
📓 06 → interpréter les résultats → visualisations dans results/figures/
```

### Ce que tu apprendras à chaque étape

**Notebook 01 — Explorer les données**
- Combien d'utilisateurs, de films, d'interactions ?
- Quand les gens regardent-ils des films ? (distributions horaires)
- Y a-t-il des patterns temporels visibles à l'œil ?

**Notebook 02 — Feature Engineering**
- Nettoyer les données (retirer les utilisateurs avec trop peu d'interactions)
- Créer les features temporelles : heure, jour, week-end
- Créer les features de session : durée, position dans la session

**Notebook 03 — Modèle Baseline (NCF sans contexte)**
- Entraîner un NCF classique sur MovieLens-100K
- Mesurer NDCG@10 et Hit Rate@10 → **c'est ta référence**
- ⚠️ Commence par ML-100K : entraînement en ~5 min sur CPU

**Notebook 04 — Modèle avec Contexte**
- Ajouter les features contextuelles au NCF
- Tester différentes façons de fusionner contexte + NCF
- Observer si les courbes d'entraînement s'améliorent

**Notebook 05 — Évaluation et Comparaison**
- Tableau comparatif : baseline vs +temps vs +session vs +tout
- Ablation study : quel contexte apporte le plus ?
- Visualisations des gains

**Notebook 06 — Interprétabilité**
- t-SNE des embeddings : les genres similaires se regroupent-ils ?
- Quelle feature contextuelle compte le plus ?
- Pour un utilisateur donné : les recommandations changent-elles selon l'heure ?

---

## Datasets

### 1. MovieLens 100K / 1M

> 💡 **Commence par ML-100K** — c'est plus petit (100 000 interactions), l'entraînement est rapide. Passe à ML-1M une fois que tout fonctionne.

**Téléchargement** : https://grouplens.org/datasets/movielens/

```bash
# Télécharger manuellement les données MovieLens depuis GroupLens
# puis placer les fichiers dans :
# - data/raw/movielens/ml-100k/
# - data/raw/movielens/ml-1m/
```

| Propriété | ML-100K | ML-1M |
|-----------|---------|-------|
| Utilisateurs | 943 | 6 040 |
| Films | 1 682 | 3 706 |
| Interactions | 100 000 | 1 000 209 |
| Densité | ~6.3% | ~4.5% |
| Timestamps | ✅ | ✅ |
| Infos utilisateurs | Âge, Genre, Métier | Idem |

**Features contextuelles extraites :**
- `hour_of_day` → encodage cyclique sin/cos (voir section suivante)
- `day_of_week` → encodage cyclique sin/cos
- `is_weekend` → booléen
- `time_since_last_interaction` → proxy de session

### 2. RetailRocket (e-commerce)

**Téléchargement** : https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

| Propriété | Valeur |
|-----------|--------|
| Événements | ~2.7 millions |
| Utilisateurs | ~1.4 million |
| Items | ~235 000 |
| Période | 4.5 mois |
| Types d'events | view, addtocart, transaction |

**Features contextuelles extraites :**
- `event_type` → proxy d'intention (view < addtocart < transaction)
- `session_id` → nouvelle session si gap > 30 min
- `session_length` → nb d'événements dans la session
- `session_position` → position de l'événement dans la session
- `device_proxy` → inféré depuis les patterns temporels

---

## Les features contextuelles expliquées

### Pourquoi l'encodage cyclique pour l'heure ?

> 💡 **Problème** : si on donne directement l'heure (0 à 23) au modèle, il pense que 23h et 0h sont très éloignées. Mais en réalité, elles sont adjacentes !

**Solution** : encoder l'heure avec sin/cos pour créer une "roue" :

```
hour_sin = sin(2π × heure / 24)
hour_cos = cos(2π × heure / 24)
```

Ainsi, 23h et 0h ont des valeurs très proches → le modèle comprend la continuité temporelle. Même logique pour les jours de la semaine (0–6).

### Tableau complet des features

| Feature | Dataset | Ce que ça représente | Pourquoi c'est utile |
|---------|---------|----------------------|----------------------|
| `hour_sin / hour_cos` | Les deux | Heure de la journée | Matin ≠ soir |
| `day_sin / day_cos` | Les deux | Jour de la semaine | Semaine ≠ week-end |
| `is_weekend` | Les deux | Samedi ou dimanche ? | Comportement loisir |
| `time_since_last` | Les deux | Temps depuis la dernière interaction | Longueur de session |
| `session_length` | RetailRocket | Nb d'événements dans la session | Exploration vs. achat |
| `session_position` | RetailRocket | Position dans la session | Début = curiosité, fin = décision |
| `event_type` | RetailRocket | view / addtocart / transaction | Intention d'achat |
| `device_proxy` | RetailRocket | Inféré depuis patterns temporels | Mobile ≠ Desktop |

---

## Les modèles, simplement

### Modèle 1 — NCF Baseline (sans contexte)

Basé sur He et al. (2017). C'est ton **point de référence** : si le modèle avec contexte n'est pas meilleur que celui-ci, quelque chose ne va pas.

```
User_ID ──► [Embedding user]  ──┐
                                 ├──► [MLP 128→64→32→16] ──► Score
Item_ID ──► [Embedding item]  ──┘
```

- **Embedding** : transforme un ID en vecteur numérique (dim = 64)
- **MLP** : réseau de neurones qui apprend à combiner user + item
- **Score** : probabilité que l'utilisateur aime cet item

### Modèle 2 — NCF + Contexte (modèle principal)

On ajoute un **encodeur contextuel** qui transforme les features temporelles et de session en vecteur, puis on fusionne avec la sortie du NCF :

```
User_ID + Item_ID ──► [NCF] ──────────────────────────┐
                                                        ├──► [MLP Final] ──► Score
[Features contextuelles] ──► [Context Encoder] ────────┘
```

**Stratégies de fusion testées (ablation) :**

| Stratégie | Description simple | À tester en premier ? |
|-----------|-------------------|-----------------------|
| Concatenation | On colle les deux vecteurs | ✅ Oui, c'est la plus simple |
| FiLM | Le contexte *modifie* les embeddings NCF | En second |
| Attention | Le modèle apprend quelles features contextuelles privilegier | En dernier |

---

## Comment évaluer les résultats ?

### Les métriques de ranking (et ce qu'elles veulent dire)

> 💡 En recommandation, on ne prédit pas une valeur exacte — on génère une **liste ordonnée**. Les métriques évaluent si le bon item est bien placé dans cette liste.

**Protocole utilisé (Leave-One-Out) :**
Pour chaque utilisateur du test set, on prend sa dernière interaction comme "bonne réponse" et on tire au sort 99 items qu'il n'a jamais vus. Le modèle doit classer la bonne réponse le plus haut possible parmi ces 100 candidats.

| Métrique | Ce qu'elle mesure | Exemple concret |
|----------|-------------------|-----------------|
| **Hit Rate@10** | La bonne réponse est-elle dans le top 10 ? | Sur 100 utilisateurs, 60 ont leur film dans le top 10 → HR@10 = 0.60 |
| **NDCG@10** | Est-elle bien *classée* dans le top 10 ? | Pénalise si le film est en position 9 plutôt qu'en position 1 |
| **MRR** | En moyenne, à quel rang apparaît la bonne réponse ? | MRR = 0.30 → en moyenne rang ~3.3 |

**Valeurs K standards utilisées** : K ∈ {5, 10, 20}

---

## Ablation Study — tester l'impact de chaque contexte

L'ablation, c'est **retirer une pièce du modèle pour voir si elle manque vraiment**. C'est la méthode rigoureuse pour justifier chaque choix.

### Plan d'ablation

| Expérience | Ce qu'on teste | Variable supprimée |
|------------|----------------|-------------------|
| `baseline` | NCF pur, zéro contexte | — |
| `+temps` | NCF + features temporelles seulement | session, device |
| `+session` | NCF + features de session seulement | temps, device |
| `+device` | NCF + proxy device seulement | temps, session |
| `+temps+session` | NCF + temps + session | device |
| `+full_context` | NCF + tous les contextes | — |
| `concat_fusion` | Full model, fusion par concaténation | — |
| `film_fusion` | Full model, fusion FiLM | — |
| `attention_fusion` | Full model, fusion attention | — |

### Résultats attendus (exemple — MovieLens 100K)

| Modèle | NDCG@10 | HR@10 | MRR | Δ NDCG vs baseline |
|--------|---------|-------|-----|---------------------|
| NCF Baseline | 0.342 | 0.601 | 0.289 | — |
| NCF + Temps | 0.358 | 0.618 | 0.304 | +1.6% |
| NCF + Session | 0.371 | 0.632 | 0.315 | +2.9% |
| NCF + Device Proxy | 0.347 | 0.609 | 0.295 | +0.5% |
| NCF + Full Context (concat) | 0.381 | 0.644 | 0.325 | +3.9% |
| NCF + Full Context (FiLM) | 0.389 | 0.651 | 0.332 | +4.7% |
| NCF + Full Context (attention) | 0.392 | 0.654 | 0.335 | +5.0% |

> **Lecture** : le device proxy apporte peu seul (+0.5%), mais la session apporte beaucoup (+2.9%). La fusion attention est la meilleure mais la plus complexe.

---

## Interprétabilité

> 💡 L'interprétabilité, c'est répondre à : **"pourquoi le modèle recommande-t-il ça ?"** C'est essentiel pour avoir confiance dans les résultats.

### 1. Attention Weights — Quel contexte compte le plus ?

Si la fusion est basée sur l'attention, on visualise les poids attribués à chaque feature contextuelle. Question clé : **l'heure compte-t-elle plus que la position en session ?**

### 2. Feature Importance (Permutation)

Méthode simple et intuitive :
1. Mélanger aléatoirement les valeurs d'une feature dans le test set
2. Mesurer la dégradation de NDCG@10
3. Plus la dégradation est grande → plus la feature est importante

```
importance(feature) = NDCG_normal - NDCG_avec_feature_mélangée
```

### 3. Visualisation des Embeddings (t-SNE)

Les embeddings sont des vecteurs de dimension 64. t-SNE les projette en 2D pour qu'on puisse les visualiser. On s'attend à ce que :
- Les films du même **genre** se regroupent
- Les utilisateurs avec des **profils temporels similaires** se regroupent

### 4. Analyse de l'effet contextuel

Pour un utilisateur donné : comment changent ses recommandations selon...
- L'heure (matin 8h vs. soir 22h) ?
- Le jour (lundi vs. samedi) ?
- Sa position dans la session (1er clic vs. 10ème clic) ?

---

## Résultats attendus

### MovieLens 100K

| Modèle | NDCG@10 | HR@10 |
|--------|---------|-------|
| NCF Baseline | ~0.34 | ~0.60 |
| + Contexte complet | ~0.38–0.41 | ~0.64–0.67 |

### RetailRocket

| Modèle | NDCG@10 | HR@10 |
|--------|---------|-------|
| NCF Baseline | ~0.28 | ~0.52 |
| + Contexte complet | ~0.33–0.37 | ~0.58–0.63 |

> **Pourquoi RetailRocket bénéficie plus du contexte ?** Les patterns temporels de e-commerce sont très prononcés : heures de shopping, sessions d'achat intentionnel, comportements très différents entre mobile et desktop.

---

## Installation

### Prérequis

- Python 3.12
- Jupyter Lab ou Jupyter Notebook
- `uv` (optionnel mais pris en charge)
- GPU recommandé (CUDA 12+), mais **CPU suffit pour MovieLens-100K**

### Démarrage en 4 étapes

```bash
# 1. Cloner le projet
git clone <repo-url>
cd context-aware-recsys

# 2. Installer les dépendances
uv sync

# 3. Télécharger les données et les placer dans data/raw/
#    MovieLens 100K  -> data/raw/movielens/ml-100k/
#    MovieLens 1M    -> data/raw/movielens/ml-1m/
#    RetailRocket    -> data/raw/retailrocket/

# 4. Lancer Jupyter
uv run jupyter lab
```

> Si tu n'utilises pas `uv`, tu peux remplacer `uv sync` par `python3 -m pip install numpy pandas matplotlib seaborn scikit-learn torch jupyterlab` et `uv run jupyter lab` par `python3 -m jupyter lab`.

### Vérifier que tout fonctionne

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Reproductibilité

Toutes les expériences utilisent une graine aléatoire fixée — les résultats sont identiques à chaque exécution :

```python
SEED = 42
torch.manual_seed(SEED)
numpy.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

Les résultats (hyperparamètres + métriques) de chaque expérience sont sauvegardés automatiquement dans `results/tables/`.

---

## Références

```
[1] He, X., et al. (2017). Neural Collaborative Filtering. WWW 2017.
[2] Baltrunas, L., et al. (2012). Context-aware Matrix Factorization. CARS 2012.
[3] Liu, Q., et al. (2016). Context-aware Sequential Recommendation. ICDM 2016.
[4] Perera, D., & Zimmermann, R. (2019). Attending to Future Utility for Conversational Recommendation. CIKM 2019.
[5] Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. ACM TIIS.
```

---

*Python 3.12 · PyTorch · Jupyter Lab*
