import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ajouter src au path pour importer les modèles et utilitaires du projet
ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = ROOT / "src"
sys.path.append(str(SYS_PATH))

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
RESULTS_DIR = ROOT / "results" / "models"
CONTEXT_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "time_since_last_interaction_log",
    "session_length",
    "session_position_norm",
]


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    processed_path: Path
    baseline_file: str
    context_file: str
    raw_movies_path: Path | None
    movie_format: str | None
    item_label: str
    default_users: int
    default_items: int


@dataclass
class ModelBundle:
    key: str
    label: str
    path: Path
    model: torch.nn.Module | None
    n_users: int
    n_items: int
    error: str | None = None


DATASETS = {
    "ml100k": DatasetConfig(
        label="MovieLens 100K",
        processed_path=ROOT / "data" / "processed" / "movielens_100k" / "interactions.parquet",
        baseline_file="baseline_ncf_ml100k.pt",
        context_file="context_ncf_ml100k.pt",
        raw_movies_path=ROOT / "data" / "raw" / "movielens" / "ml-100k" / "movies.csv",
        movie_format="csv",
        item_label="Films vus dans cette session",
        default_users=943,
        default_items=1682,
    ),
    "ml1m": DatasetConfig(
        label="MovieLens 1M",
        processed_path=ROOT / "data" / "processed" / "movielens_1m" / "interactions.parquet",
        baseline_file="baseline_ncf_ml1m.pt",
        context_file="context_ncf_ml1m.pt",
        raw_movies_path=ROOT / "data" / "raw" / "movielens" / "ml-1m" / "movies.dat",
        movie_format="dat",
        item_label="Films vus dans cette session",
        default_users=6040,
        default_items=3416,
    ),
    "retailrocket": DatasetConfig(
        label="RetailRocket",
        processed_path=ROOT / "data" / "processed" / "retailrocket" / "interactions.parquet",
        baseline_file="baseline_ncf_retailrocket.pt",
        context_file="context_ncf_retailrocket.pt",
        raw_movies_path=None,
        movie_format=None,
        item_label="Articles consultés dans cette session",
        default_users=65968,
        default_items=36126,
    ),
}


def infer_checkpoint_shape(state_dict: dict[str, torch.Tensor], model_key: str) -> tuple[int, int]:
    if model_key == "baseline":
        return (
            state_dict["embedding_user_gmf.weight"].shape[0],
            state_dict["embedding_item_gmf.weight"].shape[0],
        )
    return (
        state_dict["ncf_core.embedding_user_gmf.weight"].shape[0],
        state_dict["ncf_core.embedding_item_gmf.weight"].shape[0],
    )


def build_model(model_key: str, n_users: int, n_items: int) -> torch.nn.Module:
    if model_key == "baseline":
        return NCF(
            num_users=n_users,
            num_items=n_items,
            embed_dim=32,
            mlp_layers=[64, 32, 16],
            dropout=0.2,
        )

    return NCFContext(
        num_users=n_users,
        num_items=n_items,
        context_dim=len(CONTEXT_COLS),
        embed_dim=32,
        mlp_layers=[64, 32, 16],
        context_embed_dim=32,
        dropout=0.2,
        fusion_type="concat",
    )


@st.cache_resource
def load_model_bundle(dataset_key: str, model_key: str) -> ModelBundle:
    config = DATASETS[dataset_key]
    filename = config.baseline_file if model_key == "baseline" else config.context_file
    path = RESULTS_DIR / filename
    label = "Baseline NCF" if model_key == "baseline" else "NCF + contexte"

    if path.name == "last_model.pt" or not path.exists():
        return ModelBundle(model_key, label, path, None, 0, 0, "checkpoint introuvable")

    try:
        state_dict = torch.load(path, map_location="cpu")
        n_users, n_items = infer_checkpoint_shape(state_dict, model_key)
        model = build_model(model_key, n_users, n_items)
        model.load_state_dict(state_dict)
        model.eval()
        return ModelBundle(model_key, label, path, model, n_users, n_items)
    except Exception as exc:
        return ModelBundle(model_key, label, path, None, 0, 0, str(exc))


@st.cache_data
def load_movie_metadata(path: Path, movie_format: str | None) -> dict[int, tuple[str, str]]:
    if path is None or not path.exists():
        return {}

    if movie_format == "csv":
        movies_df = pd.read_csv(path)
        id_col = "movieId" if "movieId" in movies_df.columns else "item_id"
        return {
            int(row[id_col]): (str(row["title"]), str(row.get("genres", "Film")))
            for _, row in movies_df.iterrows()
        }

    if movie_format == "dat":
        movies_df = pd.read_csv(
            path,
            sep="::",
            names=["movieId", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
        return {
            int(row["movieId"]): (str(row["title"]), str(row["genres"]))
            for _, row in movies_df.iterrows()
        }

    return {}


@st.cache_data
def load_item_catalog(dataset_key: str) -> tuple[list[str], list[str], int, int]:
    config = DATASETS[dataset_key]
    titles = [f"Item encodé {idx}" for idx in range(config.default_items)]
    genres = ["Inconnu"] * config.default_items
    n_users = config.default_users
    n_items = config.default_items

    if config.processed_path.exists():
        df = pd.read_parquet(config.processed_path)
        if "user_id_encoded" in df.columns:
            n_users = int(df["user_id_encoded"].nunique())
        if "item_id_encoded" in df.columns:
            n_items = max(n_items, int(df["item_id_encoded"].max()) + 1)
            titles = [f"Item encodé {idx}" for idx in range(n_items)]
            genres = ["Inconnu"] * n_items

            item_map = (
                df[["item_id_encoded", "item_id"]]
                .drop_duplicates("item_id_encoded")
                .sort_values("item_id_encoded")
            )
            metadata = load_movie_metadata(config.raw_movies_path, config.movie_format)
            event_by_item = {}
            if "event" in df.columns:
                event_by_item = (
                    df.groupby("item_id_encoded")["event"]
                    .agg(lambda values: values.mode().iat[0] if not values.mode().empty else "event")
                    .to_dict()
                )

            for _, row in item_map.iterrows():
                encoded_id = int(row["item_id_encoded"])
                original_id = int(row["item_id"])
                if encoded_id >= len(titles):
                    continue
                if metadata and original_id in metadata:
                    title, genre = metadata[original_id]
                elif dataset_key == "retailrocket":
                    title = f"Article #{original_id}"
                    genre = f"RetailRocket · {event_by_item.get(encoded_id, 'item')}"
                else:
                    title = f"Item #{original_id}"
                    genre = "Inconnu"
                titles[encoded_id] = title
                genres[encoded_id] = genre

    return titles, genres, n_users, n_items


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


def get_recommendations(
    model,
    user_id: int,
    context_vec: list[float],
    n_items_trained: int,
    titles: list[str],
    genres: list[str],
    top_k: int = 5,
    batch_size: int = 8192,
):
    """Génère des recommandations pour un utilisateur et un contexte donnés.
    
    Args:
        model: Modèle NCF
        user_id: ID de l'utilisateur
        context_vec: Vecteur de contexte (non utilisé par NCF pur, mais pour compatibilité)
        n_items_trained: Nombre d'items sur lesquels le modèle a été entraîné
        top_k: Nombre de recommandations à retourner
    """
    preds_batches = []
    safe_user_id = min(max(0, user_id), model.num_users - 1 if hasattr(model, "num_users") else user_id)
    if isinstance(model, NCFContext):
        safe_user_id = min(max(0, user_id), model.ncf_core.num_users - 1)

    with torch.no_grad():
        for start in range(0, n_items_trained, batch_size):
            end = min(start + batch_size, n_items_trained)
            u_tensor = torch.full((end - start,), safe_user_id, dtype=torch.long)
            i_tensor = torch.arange(start, end, dtype=torch.long)
            if isinstance(model, NCFContext):
                ctx_tensor = torch.tensor(context_vec, dtype=torch.float32).unsqueeze(0).repeat(end - start, 1)
                preds_batches.append(model(u_tensor, i_tensor, ctx_tensor).cpu())
            else:
                preds_batches.append(model(u_tensor, i_tensor).cpu())

    preds = torch.cat(preds_batches).numpy()
    top_idx = np.argsort(preds)[::-1][:top_k]
    
    results = []
    for i in top_idx:
        if i < len(titles):
            results.append((titles[i], float(preds[i]), genres[i]))
        else:
            results.append((f"Item encodé {i}", float(preds[i]), "Inconnu"))
    return results


# ─────────────────────────────────────────────
# CONTENU
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">RecoArena 🎬</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Explorez les recommandations NCF et NCF contextuels sur MovieLens et RetailRocket</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Contexte utilisateur")
    dataset_key = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        format_func=lambda key: DATASETS[key].label,
    )

MOVIES, GENRES, DATA_N_USERS, DATA_N_ITEMS = load_item_catalog(dataset_key)
MODEL_BUNDLES = {
    "baseline": load_model_bundle(dataset_key, "baseline"),
    "context": load_model_bundle(dataset_key, "context"),
}
AVAILABLE_MODELS = {
    key: bundle for key, bundle in MODEL_BUNDLES.items()
    if bundle.model is not None
}

with st.sidebar:
    for bundle in MODEL_BUNDLES.values():
        if bundle.model is None:
            st.warning(f"{bundle.label} indisponible : {bundle.path.name}")
        else:
            st.success(f"{bundle.label} chargé : {bundle.n_users} users · {bundle.n_items} items")

    if not AVAILABLE_MODELS:
        st.error("Aucun checkpoint utilisable pour ce dataset.")
        st.stop()

    model_key = st.radio(
        "Modèle",
        list(AVAILABLE_MODELS.keys()),
        format_func=lambda key: AVAILABLE_MODELS[key].label,
        index=0,
    )
    selected_bundle = AVAILABLE_MODELS[model_key]
    selected_model = selected_bundle.model
    selected_n_items = selected_bundle.n_items
    max_user_id = max(0, selected_bundle.n_users - 1)

    if max_user_id > 0:
        user_id = st.slider(
            "ID utilisateur encodé",
            0,
            max_user_id,
            min(42, max_user_id),
            key=f"user_{dataset_key}_{model_key}",
        )
    else:
        user_id = 0
        st.write("ID utilisateur encodé : 0")

    hour = st.slider("Heure de la journée", 0, 23, 20)
    day_map = {"Lun": 0, "Mar": 1, "Mer": 2, "Jeu": 3, "Ven": 4, "Sam": 5, "Dim": 6}
    day_name = st.selectbox("Jour", list(day_map.keys()), index=4)
    day_of_week = day_map[day_name]
    minutes_since_last = st.slider("Minutes depuis la dernière interaction", 0, 240, 30)

    st.markdown("---")
    st.markdown("### 🎞️ Session actuelle")
    session_options = MOVIES[:500] if dataset_key == "retailrocket" else MOVIES
    session_movies = st.multiselect(
        DATASETS[dataset_key].item_label,
        session_options,
        default=session_options[:2] if len(session_options) >= 2 else session_options,
        max_selections=5,
    )
    session_length = max(1, len(session_movies))
    if session_length > 1:
        session_position = st.slider(
            "Position dans la session",
            1,
            session_length,
            session_length,
        )
    else:
        session_position = 1
        st.write("Position dans la session : 1 (session unique)")

    run_btn = st.button("🚀 Lancer les recommandations", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

current_signature = (
    dataset_key, model_key, user_id, hour, day_of_week,
    minutes_since_last, session_length, session_position,
)

if run_btn or st.session_state.get("last_signature") != current_signature:
    ctx_vec = build_context_vector(hour, day_of_week, minutes_since_last, session_length, session_position)
    recommendations = get_recommendations(selected_model, user_id, ctx_vec, selected_n_items, MOVIES, GENRES)
    st.session_state.last_recos = recommendations
    st.session_state.last_signature = current_signature

if st.session_state.get("last_recos"):
    recos = st.session_state.last_recos
    st.markdown(
        f"""
        <div style='margin-bottom:1.2rem; display:flex; gap:8px; flex-wrap:wrap; align-items:center;'>
            <span style='color:#6b7280; font-size:0.8rem;'>Contexte :</span>
            <span class='context-tag'>{escape(DATASETS[dataset_key].label)}</span>
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
            🎬 Recommandations {escape(selected_bundle.label)}
            <br><span style='font-size:0.75rem; font-weight:400; color:#6b7280;'>Neural Collaborative Filtering</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    min_score = min(r[1] for r in recos)
    max_score = max(r[1] for r in recos)
    score_span = max(max_score - min_score, 1e-9)
    card_class = "context" if model_key == "context" else "ncf"
    badge_class = "badge-context" if model_key == "context" else "badge-ncf"
    for rank, (title, score, genre) in enumerate(recos, 1):
        width = int(((score - min_score) / score_span) * 90 + 10)
        st.markdown(
            f"""
            <div class='model-card {card_class}'>
                <span class='model-badge {badge_class}'>#{rank} · {escape(str(genre))}</span>
                <p class='item-title'>{escape(str(title))}</p>
                <p class='item-score'>Score : {score:.3f}</p>
                <div class='score-bar {card_class}' style='width:{min(100, width)}%'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Recommandation pertinente", use_container_width=True):
            st.session_state.history.append({
                "dataset": DATASETS[dataset_key].label,
                "modele": selected_bundle.label,
                "user": user_id,
                "heure": hour,
                "jour": day_name,
                "feedback": "👍 Pertinent",
            })
            st.success("Merci pour ton retour !")
    
    with col2:
        if st.button("👎 Recommandation non pertinente", use_container_width=True):
            st.session_state.history.append({
                "dataset": DATASETS[dataset_key].label,
                "modele": selected_bundle.label,
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
