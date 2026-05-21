import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ajouter src au path pour importer les modèles et utilitaires du projet
ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = ROOT / "src"
sys.path.append(str(SYS_PATH))

from data_utils import load_movielens_100k, encode_ids
from models import NCF, NCFContext

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RecoArena · Comparateur de Modèles",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0a0f;
        color: #e8e6f0;
    }

    .stApp { background-color: #0a0a0f; }

    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #9ca3af;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .model-card {
        background: linear-gradient(145deg, #13131f, #1a1a2e);
        border: 1px solid #2d2d4a;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .model-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px;
        height: 100%;
        border-radius: 4px 0 0 4px;
    }

    .model-card.ncf::before   { background: #a78bfa; }
    .model-card.context::before { background: #60a5fa; }

    .model-card:hover {
        border-color: #4f4f7a;
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(167,139,250,0.1);
    }

    .model-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.6rem;
    }

    .badge-ncf     { background: rgba(167,139,250,0.15); color: #a78bfa; }
    .badge-context { background: rgba(96,165,250,0.15);  color: #60a5fa; }

    .item-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        margin: 0;
        color: #f1f0f8;
    }

    .item-score {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 2px;
    }

    .score-bar {
        height: 3px;
        border-radius: 2px;
        margin-top: 6px;
    }

    .score-bar.ncf     { background: linear-gradient(90deg, #a78bfa, transparent); }
    .score-bar.context { background: linear-gradient(90deg, #60a5fa, transparent); }

    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #e8e6f0;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1f1f35;
    }

    .stat-box {
        background: #13131f;
        border: 1px solid #2d2d4a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }

    .stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }

    .stat-lbl {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .winner-banner {
        background: linear-gradient(135deg, #1a1040, #0d1f3c, #1f0d2e);
        border: 1px solid #4f4f7a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .winner-banner h2 {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.4rem 0;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d2d4a, transparent);
        margin: 1.5rem 0;
    }

    .context-tag {
        display: inline-block;
        background: rgba(255,255,255,0.05);
        border: 1px solid #2d2d4a;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: #9ca3af;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CHEMINS ET CONSTANTES
# ─────────────────────────────────────────────
RAW_ROOT = ROOT / "data" / "raw" / "movielens" / "ml-100k"
RESULTS_DIR = ROOT / "results" / "models"
BASELINE_MODEL_PATH = RESULTS_DIR / "baseline_ncf_ml100k.pt"
CONTEXT_MODEL_PATH = RESULTS_DIR / "context_ncf_ml100k.pt"
MOVIES_PATH = RAW_ROOT / "movies.csv"

@st.cache_data
def load_movie_catalog():
    if MOVIES_PATH.exists():
        movies_df = pd.read_csv(MOVIES_PATH)
        titles = movies_df['title'].astype(str).tolist()
        genres = movies_df['genres'].astype(str).tolist()
    else:
        titles = [f"Film {i}" for i in range(1, 101)]
        genres = ["Inconnu"] * len(titles)
    return titles, genres

@st.cache_data
def load_dataset_shape():
    # Les paramètres du modèle sauvegardé
    # n_users=943, n_items=1349 (items vraiment notés dans l'ensemble d'entraînement)
    return 943, 1349

@st.cache_resource
def load_models(n_users: int, n_items: int):
    """Charge les modèles baseline et contextuel depuis les fichiers sauvegardés."""
    baseline_model = None
    context_model = None
    baseline_n_items = n_items
    context_n_items = n_items

    # Baseline
    baseline_model = NCF(
        num_users=n_users,
        num_items=n_items,
        embed_dim=32,
        mlp_layers=[64, 32, 16],
        dropout=0.2,
    )
    if BASELINE_MODEL_PATH.exists():
        try:
            baseline_state = torch.load(BASELINE_MODEL_PATH, map_location='cpu')
            baseline_n_items = baseline_state['embedding_item_gmf.weight'].shape[0]
            baseline_n_users = baseline_state['embedding_user_gmf.weight'].shape[0]
            baseline_model = NCF(
                num_users=baseline_n_users,
                num_items=baseline_n_items,
                embed_dim=32,
                mlp_layers=[64, 32, 16],
                dropout=0.2,
            )
            baseline_model.load_state_dict(baseline_state)
            baseline_model.eval()
            st.success(f"✅ Baseline chargé depuis {BASELINE_MODEL_PATH.name}")
        except Exception as exc:
            st.warning(f"⚠️ Impossible de charger le baseline : {exc}. Mode démo activé.")
            baseline_model.eval()
    else:
        st.info(f"ℹ️ Fichier {BASELINE_MODEL_PATH.name} introuvable : mode démo activé.")
        baseline_model.eval()

    # Contextuel
    if CONTEXT_MODEL_PATH.exists():
        try:
            context_state = torch.load(CONTEXT_MODEL_PATH, map_location='cpu')
            context_n_items = context_state['ncf_core.embedding_item_gmf.weight'].shape[0]
            context_n_users = context_state['ncf_core.embedding_user_gmf.weight'].shape[0]
            context_model = NCFContext(
                num_users=context_n_users,
                num_items=context_n_items,
                context_dim=8,
                embed_dim=32,
                mlp_layers=[64, 32, 16],
                context_embed_dim=32,
                dropout=0.2,
                fusion_type='concat'
            )
            context_model.load_state_dict(context_state)
            context_model.eval()
            st.success(f"✅ Modèle contextuel chargé depuis {CONTEXT_MODEL_PATH.name}")
        except Exception as exc:
            st.warning(f"⚠️ Impossible de charger le modèle contextuel : {exc}. Mode démo activé.")
            context_model = None
    else:
        st.info(f"ℹ️ Fichier {CONTEXT_MODEL_PATH.name} introuvable : modèle contextuel indisponible.")

    return baseline_model, context_model, baseline_n_items, context_n_items


def build_context_vector(hour: int, day_of_week: int, minutes_since_last: int, session_length: int, session_position: int):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = int(day_of_week in {5, 6})
    time_since_last = np.log1p(minutes_since_last * 60)
    if session_length <= 1:
        session_position_norm = 1.0
    else:
        session_position_norm = float(session_position - 1) / float(session_length - 1)
    return [
        float(hour_sin), float(hour_cos),
        float(dow_sin), float(dow_cos),
        float(is_weekend), float(time_since_last),
        float(session_length), float(session_position_norm),
    ]


def get_recommendations(model, user_id: int, context_vec: list[float], n_items_trained: int, top_k: int = 5):
    """Génère des recommandations pour un utilisateur et un contexte donnés.
    
    Args:
        model: Modèle NCF
        user_id: ID de l'utilisateur
        context_vec: Vecteur de contexte (non utilisé par NCF pur, mais pour compatibilité)
        n_items_trained: Nombre d'items sur lesquels le modèle a été entraîné
        top_k: Nombre de recommandations à retourner
    """
    u_tensor = torch.tensor([user_id] * n_items_trained, dtype=torch.long)
    i_tensor = torch.arange(n_items_trained, dtype=torch.long)

    with torch.no_grad():
        if isinstance(model, NCFContext):
            ctx_tensor = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).repeat(n_items_trained, 1)
            preds = model(u_tensor, i_tensor, ctx_tensor)
        else:
            preds = model(u_tensor, i_tensor)

    preds = preds.cpu().numpy()
    top_idx = np.argsort(preds)[::-1][:top_k]
    
    # Retourne les films correspondants (avec fallback si l'index dépasse la liste)
    results = []
    for i in top_idx:
        if i < len(MOVIES):
            results.append((MOVIES[i], float(preds[i]), GENRES[i]))
        else:
            results.append((f"Film {i}", float(preds[i]), "Inconnu"))
    return results


# ─────────────────────────────────────────────
# CONTENU
# ─────────────────────────────────────────────
MOVIES, GENRES = load_movie_catalog()
N_USERS, N_ITEMS = load_dataset_shape()
BASELINE_MODEL, CONTEXT_MODEL, BASELINE_N_ITEMS, CONTEXT_N_ITEMS = load_models(N_USERS, N_ITEMS)

st.markdown('<div class="hero-title">RecoArena 🎬</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Explorez les recommandations NCF et NCF contextuels MovieLens 100K</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Contexte utilisateur")
    model_options = ["Baseline NCF"]
    if CONTEXT_MODEL is not None:
        model_options.append("NCF + contexte")

    model_option = st.radio("Modèle", model_options, index=0)
    selected_model = CONTEXT_MODEL if model_option == "NCF + contexte" else BASELINE_MODEL
    selected_n_items = CONTEXT_N_ITEMS if model_option == "NCF + contexte" else BASELINE_N_ITEMS
    if selected_model is None:
        st.warning("Le modèle sélectionné n'est pas disponible. Baseline utilisé par défaut.")
        selected_model = BASELINE_MODEL
        selected_n_items = BASELINE_N_ITEMS

    user_id = st.slider("ID utilisateur", 0, max(0, N_USERS - 1), min(42, max(0, N_USERS - 1)))
    hour = st.slider("Heure de la journée", 0, 23, 20)
    day_map = {"Lun": 0, "Mar": 1, "Mer": 2, "Jeu": 3, "Ven": 4, "Sam": 5, "Dim": 6}
    day_name = st.selectbox("Jour", list(day_map.keys()), index=4)
    day_of_week = day_map[day_name]
    minutes_since_last = st.slider("Minutes depuis la dernière interaction", 0, 240, 30)

    st.markdown("---")
    st.markdown("### 🎞️ Session actuelle")
    session_movies = st.multiselect(
        "Films vus dans cette session",
        MOVIES,
        default=MOVIES[:2] if len(MOVIES) >= 2 else MOVIES,
        max_selections=5,
    )
    session_length = max(1, len(session_movies))
    session_position = st.slider(
        "Position dans la session",
        1,
        session_length,
        session_length,
        disabled=(session_length <= 1),
    )

    run_btn = st.button("🚀 Lancer les recommandations", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

if run_btn or st.session_state.history:
    ctx_vec = build_context_vector(hour, day_of_week, minutes_since_last, session_length, session_position)
    recommendations = get_recommendations(selected_model, user_id, ctx_vec, selected_n_items)
    st.session_state.last_recos = recommendations

if st.session_state.get("last_recos"):
    recos = st.session_state.last_recos
    st.markdown(
        f"""
        <div style='margin-bottom:1.2rem; display:flex; gap:8px; flex-wrap:wrap; align-items:center;'>
            <span style='color:#6b7280; font-size:0.8rem;'>Contexte :</span>
            <span class='context-tag'>👤 User #{user_id}</span>
            <span class='context-tag'>🕐 {hour:02d}h00</span>
            <span class='context-tag'>📅 {day_name}</span>
            <span class='context-tag'>⏱ {minutes_since_last} min depuis la dernière action</span>
            <span class='context-tag'>🎞 {session_length} film(s) en session</span>
            <span class='context-tag'>📍 position {session_position}/{session_length}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='section-label'>
            🎬 Recommandations {model_option}
            <br><span style='font-size:0.75rem; font-weight:400; color:#6b7280;'>Neural Collaborative Filtering</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    for rank, (title, score, genre) in enumerate(recos, 1):
        width = int((score - min(r[1] for r in recos)) * 100 + 10)
        st.markdown(
            f"""
            <div class='model-card ncf'>
                <span class='model-badge badge-ncf'>#{rank} · {genre}</span>
                <p class='item-title'>{title}</p>
                <p class='item-score'>Score : {score:.3f}</p>
                <div class='score-bar ncf' style='width:{min(100, width)}%'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Recommandation pertinente", use_container_width=True):
            st.session_state.history.append({
                "user": user_id,
                "heure": hour,
                "jour": day_name,
                "feedback": "👍 Pertinent",
            })
            st.success("Merci pour ton retour !")
    
    with col2:
        if st.button("👎 Recommandation non pertinente", use_container_width=True):
            st.session_state.history.append({
                "user": user_id,
                "heure": hour,
                "jour": day_name,
                "feedback": "👎 Non pertinent",
            })
            st.warning("Retour enregistré, on s'améliore !")

    if st.session_state.history:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander("📋 Historique des retours"):
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
