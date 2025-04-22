import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_scaler():
    """Cria e retorna um objeto MinMaxScaler"""
    return MinMaxScaler()

def fit_scaler(scaler, X):
    """Ajusta o scaler aos dados de treino"""
    return scaler.fit(X)

def transform_data(scaler, X):
    """Aplica a transformação aos dados"""
    return scaler.transform(X)

def fit_transform_data(scaler, X):
    """Ajusta e transforma os dados em uma única operação"""
    return scaler.fit_transform(X)

def normalize_features_for_final_model(input_file):
    """Função para normalizar o conjunto completo para o modelo final"""
    try:
        df = pd.read_csv(input_file)
        print(f"Carregando features para {len(df)} trials")
        
        metadata_cols = ['trial_number', 'label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        features_data = df[feature_cols]
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(features_data)
        
        normalized_df = pd.DataFrame(normalized_data, columns=feature_cols)
        
        for col in metadata_cols:
            normalized_df[col] = df[col]
            
        final_cols = metadata_cols + feature_cols
        normalized_df = normalized_df[final_cols]
        
        return normalized_df, scaler, feature_cols
    except Exception as e:
        print(f"Erro durante a normalização: {e}")
        return None, None, None
