import pandas as pd
import numpy as np
import joblib

def load_pipeline(model_path, scaler_path, selector_path):
    """Carrega os componentes do pipeline"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)
        return model, scaler, selector
    except Exception as e:
        print(f"Erro ao carregar pipeline: {e}")
        return None, None, None

def predict_new_data(data_file, model, scaler, selector):
    """Faz previsões em novos dados usando o pipeline completo"""
    try:
        # Carregar dados
        df = pd.read_csv(data_file)
        print(f"Carregando {len(df)} amostras para predição")
        
        # Extrair features (excluindo metadados)
        metadata_cols = ['trial_number']
        if 'label' in df.columns:
            metadata_cols.append('label')
            
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        X = df[feature_cols].values
        
        # Aplicar o pipeline
        X_scaled = scaler.transform(X)
        X_selected = selector.transform(X_scaled)
        predictions = model.predict(X_selected)
        
        # Adicionar previsões ao dataframe
        df['predicted_label'] = predictions
        
        # Se houver labels reais, calcular acurácia
        if 'label' in df.columns:
            accuracy = (df['label'] == df['predicted_label']).mean()
            print(f"Acurácia nas previsões: {accuracy:.4f}")
        
        # Salvar resultados
        output_file = f"predictions_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Previsões salvas em: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return None

if __name__ == "__main__":
    # Caminhos para os componentes do pipeline
    model_path = "svm_model_final_20250422-123456.joblib"
    scaler_path = "scaler_final_20250422-123456.joblib"
    selector_path = "selector_final_20250422-123456.joblib"
    
    # Carregar pipeline
    model, scaler, selector = load_pipeline(model_path, scaler_path, selector_path)
    
    if model is not None:
        # Arquivo de dados para predição
        data_file = "/Users/joaomachado/Desktop/IC_V3/new_eeg_features.csv"
        
        # Fazer previsões
        predictions_df = predict_new_data(data_file, model, scaler, selector)
