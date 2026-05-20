import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

def load_movielens_100k(data_dir: str | Path) -> pd.DataFrame:
    """Charge le dataset MovieLens 100K."""
    path = Path(data_dir) / "u.data"
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {path} est introuvable. Avez-vous lancé download_data.sh ?")
    
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=names)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def load_movielens_1m(data_dir: str | Path) -> pd.DataFrame:
    """Charge le dataset MovieLens 1M."""
    path = Path(data_dir) / "ratings.dat"
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {path} est introuvable. Avez-vous lancé download_data.sh ?")
    
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='::', names=names, engine='python', encoding='latin-1')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def load_retailrocket(data_dir: str | Path) -> pd.DataFrame:
    """Charge le dataset RetailRocket (events)."""
    path = Path(data_dir) / "events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {path} est introuvable. Avez-vous lancé download_data.sh ?")
    
    df = pd.read_csv(path)
    # RetailRocket timestamps sont en millisecondes
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Renommer les colonnes pour matcher l'interface
    df = df.rename(columns={
        'visitorid': 'user_id',
        'itemid': 'item_id'
    })
    
    return df

def filter_interactions(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Filtre les utilisateurs et les items ayant moins de K interactions."""
    print(f"Début du filtrage itératif (min {min_interactions} interactions)...")
    while True:
        start_len = len(df)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        if len(df) == start_len:
            break
            
    return df

def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les IDs en entiers contigus."""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['item_id_encoded'] = item_encoder.fit_transform(df['item_id'])
    
    return df
