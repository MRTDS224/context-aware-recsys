import pandas as pd
import numpy as np

def extract_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Extrait les features temporelles : heure du jour, jour de la semaine, et encodage cyclique.
    """
    df = df.copy()
    
    # Heure et jour
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Encodage cyclique pour l'heure (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # Encodage cyclique pour le jour (0-6)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def extract_session_features(df: pd.DataFrame, user_col: str = 'user_id', 
                             timestamp_col: str = 'timestamp', gap_minutes: int = 30) -> pd.DataFrame:
    """
    Construit les features de session basées sur un gap temporel.
    """
    df = df.copy()
    df = df.sort_values([user_col, timestamp_col])
    
    # Calculer la différence de temps en secondes avec l'interaction précédente
    diffs = df.groupby(user_col)[timestamp_col].diff().dt.total_seconds().fillna(0)
    df['time_since_last_interaction'] = diffs
    
    # Une nouvelle session si gap > gap_minutes
    new_session = (df['time_since_last_interaction'] > gap_minutes * 60).astype(int)
    
    # Session ID cumulatif
    df['session_id'] = new_session.groupby(df[user_col]).cumsum()
    
    # Calcul des longueurs de session
    session_lengths = df.groupby([user_col, 'session_id']).size().reset_index(name='session_length')
    df = df.merge(session_lengths, on=[user_col, 'session_id'], how='left')
    
    # Position dans la session (1-indexed)
    df['session_position'] = df.groupby([user_col, 'session_id']).cumcount() + 1
    
    # Normalisation de la position [0, 1]
    # Si longueur = 1, la position normalisée est 1.0
    df['session_position_norm'] = np.where(
        df['session_length'] > 1,
        (df['session_position'] - 1) / (df['session_length'] - 1),
        1.0
    )
    
    # Log-normalisation des temps
    df['time_since_last_interaction_log'] = np.log1p(df['time_since_last_interaction'])
    
    return df

def extract_device_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy pour le device (desktop vs mobile) basé sur la longueur de session.
    """
    df = df.copy()
    if 'session_length' in df.columns:
        df['is_desktop_proxy'] = (df['session_length'] > 15).astype(int)
    else:
        df['is_desktop_proxy'] = 0
    return df

def extract_all_context_features(df: pd.DataFrame, is_retail_rocket: bool = False) -> pd.DataFrame:
    """
    Pipeline complète d'extraction de contexte.
    """
    df = extract_temporal_features(df)
    df = extract_session_features(df)
    
    if is_retail_rocket:
        df = extract_device_proxy(df)
        
    return df