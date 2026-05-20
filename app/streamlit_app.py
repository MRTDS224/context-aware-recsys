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
BASELINE_PATH = RESULTS_DIR / "baseline_ncf_ml100k.pt"
CONTEXT_PATH = RESULTS_DIR / "context_ncf_ml100k.pt"
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
    if RAW_ROOT.exists():
        try:
            df = load_movielens_100k(RAW_ROOT)
            df = encode_ids(df)
            n_users = int(df['user_id_encoded'].nunique())
            n_items = int(df['item_id_encoded'].nunique())
            return n_users, n_items
        except Exception:
            pass
    return 943, 1682

@st.cache_resource
def load_models(n_users: int, n_items: int):
    baseline = NCF(
        num_users=n_users,
        num_items=n_items,
        embed_dim=32,
        mlp_layers=[64, 32, 16],
        dropout=0.2,
    )
    context = NCFContext(
        num_users=n_users,
        num_items=n_items,
        context_dim=8,
        embed_dim=32,
        mlp_layers=[64, 32, 16],
        context_embed_dim=32,
        dropout=0.2,
        fusion_type='concat',
    )
    models = {
        'NCF Baseline': (BASELINE_PATH, baseline),
        'NCF Context':  (CONTEXT_PATH, context),
    }

    loaded_models = {}
    for name, (path, model) in models.items():
        if path.exists():
            try:
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                loaded_models[name] = model
            except Exception as exc:
                st.warning(f"⚠️ Impossible de charger {name} depuis {path.name} : {exc}")
                loaded_models[name] = model
        else:
            st.info(f"ℹ️ Fichier {path.name} introuvable : mode démo activé pour {name}.")
            loaded_models[name] = model
    return loaded_models


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


def get_recommendations(model, user_id: int, context_vec: list[float], top_k: int = 5):
    n_items = len(MOVIES)
    u_tensor = torch.tensor([user_id] * n_items, dtype=torch.long)
    i_tensor = torch.arange(n_items, dtype=torch.long)
    ctx_tensor = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).repeat(n_items, 1)

    with torch.no_grad():
        if isinstance(model, NCFContext):
            preds = model(u_tensor, i_tensor, ctx_tensor)
        else:
            preds = model(u_tensor, i_tensor)

    preds = preds.cpu().numpy()
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [(MOVIES[i], float(preds[i]), GENRES[i]) for i in top_idx]


# ─────────────────────────────────────────────
# CONTENU
# ─────────────────────────────────────────────
MOVIES, GENRES = load_movie_catalog()
N_USERS, N_ITEMS = load_dataset_shape()
MODELS = load_models(N_USERS, N_ITEMS)

st.markdown('<div class="hero-title">RecoArena 🎬</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Comparez des modèles de recommandation avec le contexte MovieLens 100K</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Contexte utilisateur")
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
        default=MOVIES[:2],
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

if "votes" not in st.session_state:
    st.session_state.votes = {name: 0 for name in MODELS}

if run_btn or st.session_state.history:
    ctx_vec = build_context_vector(hour, day_of_week, minutes_since_last, session_length, session_position)
    recommendations = {
        name: get_recommendations(model, user_id, ctx_vec)
        for name, model in MODELS.items()
    }
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

    cols = st.columns(len(MODELS))
    for col, (model_name, model) in zip(cols, MODELS.items()):
        with col:
            badge = "badge-ncf" if model_name == "NCF Baseline" else "badge-context"
            color = "ncf" if model_name == "NCF Baseline" else "context"
            desc = "NCF pur" if model_name == "NCF Baseline" else "NCF + contexte"
            st.markdown(
                f"""
                <div class='section-label'>
                    {model_name}
                    <br><span style='font-size:0.75rem; font-weight:400; color:#6b7280;'>{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for rank, (title, score, genre) in enumerate(recos[model_name], 1):
                width = int((score - np.min([r[1] for r in recos[model_name]])) * 100 + 10)
                st.markdown(
                    f"""
                    <div class='model-card {color}'>
                        <span class='model-badge {badge}'>#{rank} · {genre}</span>
                        <p class='item-title'>{title}</p>
                        <p class='item-score'>Score : {score:.3f}</p>
                        <div class='score-bar {color}' style='width:{min(100, width)}%'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if st.button(f"✅ Voter pour {model_name}", key=f"vote_{model_name}", use_container_width=True):
                st.session_state.votes[model_name] += 1
                st.session_state.history.append({
                    "user": user_id,
                    "heure": hour,
                    "jour": day_name,
                    "choix": model_name,
                })
                st.experimental_rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Votes</div>', unsafe_allow_html=True)
    total_votes = sum(st.session_state.votes.values())
    stat_cols = st.columns(3)
    for col, name in zip(stat_cols, MODELS):
        with col:
            st.markdown(
                f"""
                <div class='stat-box'>
                    <div class='stat-num' style='color:#a78bfa'>{st.session_state.votes[name]}</div>
                    <div class='stat-lbl'>{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if total_votes > 0:
        winner = max(st.session_state.votes, key=st.session_state.votes.get)
        st.markdown(f"<div class='winner-banner'><h2>🏆 {winner}</h2><p style='color:#6b7280;'>Modèle préféré par les votes</p></div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander("📋 Historique des votes"):
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
