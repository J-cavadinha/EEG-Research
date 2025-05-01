import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import Counter
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Importar as funções modulares
from normalize import create_scaler, fit_scaler, transform_data, normalize_features_for_final_model
from feature_selection import create_selector, fit_selector, transform_features, get_selected_feature_indices, get_feature_ranking, apply_rfe_for_final_model

def train_and_evaluate_models(input_file, n_features_to_select=10):
    try:
        # Carregar os dados
        data = pd.read_csv(input_file)
        print(f"Carregando dados com {len(data)} trials")
        
        # Separar features e target
        metadata_cols = ['trial_number', 'label']
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        
        X = data[feature_cols].values
        y = data['label'].values
        
        print(f"Usando {len(feature_cols)} features para treinamento")
        
        # Definir os modelos e seus hiperparâmetros
        models = {
            'SVM': (SVC(probability=True), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }),
            'Random Forest': (RandomForestClassifier(random_state=42), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            })
        }
        
        # Configurar a validação cruzada
        kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
        
        # Dicionário para armazenar resultados
        results = {}
        
        # Dicionário para armazenar contagem de features selecionadas
        feature_selection_counts = {feature: 0 for feature in feature_cols}
        
        # Para cada modelo
        for model_name, (model, param_grid) in models.items():
            print(f"\n{'-'*50}")
            print(f"Treinando {model_name}...")
            
            fold_scores = []
            fold_selected_features = []
            
            # Para cada fold da validação cruzada
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                print(f"\nFold {fold+1}/{kfold.n_splits}")
                
                # Separar dados de treino e teste para este fold
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 1. Normalizar os dados (apenas com dados de treino)
                scaler = create_scaler()
                scaler = fit_scaler(scaler, X_train)
                X_train_scaled = transform_data(scaler, X_train)
                X_test_scaled = transform_data(scaler, X_test)
                
                # 2. Selecionar features (apenas com dados de treino)
                selector = create_selector(n_features_to_select)
                selector = fit_selector(selector, X_train_scaled, y_train)
                X_train_selected = transform_features(selector, X_train_scaled)
                X_test_selected = transform_features(selector, X_test_scaled)
                
                # Obter índices e nomes das features selecionadas
                selected_indices, selected_names = get_selected_feature_indices(selector, feature_cols)
                fold_selected_features.append(selected_names)
                
                # Atualizar contagem de features selecionadas
                for feature in selected_names:
                    feature_selection_counts[feature] += 1
                
                # 3. Treinar o modelo com GridSearchCV
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=3,  # CV interno para otimização de hiperparâmetros
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_selected, y_train)
                
                # Avaliar no conjunto de teste
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_selected)
                fold_accuracy = accuracy_score(y_test, y_pred)
                fold_scores.append(fold_accuracy)
                
                print(f"  Acurácia no fold {fold+1}: {fold_accuracy:.4f}")
                print(f"  Features selecionadas no fold {fold+1}:")
                for i, feature in enumerate(selected_names, 1):
                    print(f"    {i}. {feature}")
            
            # Calcular média e desvio padrão das acurácias
            mean_accuracy = np.mean(fold_scores)
            std_accuracy = np.std(fold_scores)
            
            # Armazenar resultados
            results[model_name] = {
                'fold_scores': fold_scores,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'fold_selected_features': fold_selected_features
            }
            
            print(f"\nResultados para {model_name}:")
            print(f"Acurácia média da validação cruzada: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Identificar as features mais frequentemente selecionadas
        sorted_features = sorted(feature_selection_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeatures mais frequentemente selecionadas:")
        for feature, count in sorted_features[:20]:
            print(f"{feature}: {count} vezes")
        
        # Visualizar as features mais selecionadas
        top_features = [f for f, c in sorted_features[:20]]
        top_counts = [c for f, c in sorted_features[:20]]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_counts, y=top_features)
        plt.title('Features mais frequentemente selecionadas')
        plt.xlabel('Número de vezes selecionada')
        plt.tight_layout()
        plt.savefig(f"feature_selection_counts_{time.strftime('%Y%m%d-%H%M%S')}.png")
        
        # Comparar modelos
        print("\nComparação de Modelos:")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"{model_name:20} Acurácia: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        
        # Identificar o melhor modelo
        best_model_name = max(results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
        print(f"\nMelhor modelo: {best_model_name}")
        
        # Treinar o modelo final com todo o conjunto de dados
        print("\nTreinando modelo final com todo o conjunto de dados...")
        
        # 1. Normalizar todo o conjunto
        scaler_final = create_scaler()
        X_scaled_final = scaler_final.fit_transform(X)
        
        # 2. Selecionar features com todo o conjunto
        selector_final, selected_indices_final, selected_names_final, feature_ranking_final = apply_rfe_for_final_model(
            X_scaled_final, y, feature_cols, n_features_to_select
        )
        
        print("\nFeatures finais selecionadas:")
        for i, feature in enumerate(selected_names_final, 1):
            print(f"{i}. {feature}")
        
        # Transformar os dados com as features selecionadas
        X_selected_final = selector_final.transform(X_scaled_final)
        
        # 3. Treinar o modelo final
        best_model_class, best_param_grid = models[best_model_name]
        
        final_grid_search = GridSearchCV(
            best_model_class,
            best_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        final_grid_search.fit(X_selected_final, y)
        
        print(f"\nMelhores parâmetros para o modelo final: {final_grid_search.best_params_}")
        print(f"Acurácia do modelo final (CV): {final_grid_search.best_score_:.4f}")
        
        # Salvar o modelo final, scaler e selector
        model_filename = f"{best_model_name.lower().replace(' ', '_')}_model_final_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
        scaler_filename = f"scaler_final_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
        selector_filename = f"selector_final_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
        
        joblib.dump(final_grid_search.best_estimator_, model_filename)
        joblib.dump(scaler_final, scaler_filename)
        joblib.dump(selector_final, selector_filename)
        
        print(f"Modelo final salvo como: {model_filename}")
        print(f"Scaler final salvo como: {scaler_filename}")
        print(f"Selector final salvo como: {selector_filename}")
        
        # Salvar as features selecionadas e seus rankings
        feature_ranking_df = pd.DataFrame(feature_ranking_final, columns=['Feature', 'Ranking'])
        feature_ranking_df = feature_ranking_df.sort_values('Ranking')
        feature_ranking_filename = f"feature_rankings_final_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        feature_ranking_df.to_csv(feature_ranking_filename, index=False)
        
        print(f"Rankings de features salvos como: {feature_ranking_filename}")
        
        # Adicionar análise detalhada das features selecionadas
        print("\nAnálise detalhada das features selecionadas:")
        print("-" * 50)
        print("\nRanking completo das features:")
        for i, (feature, rank) in enumerate(zip(feature_cols, selector_final.ranking_), 1):
            status = "Selecionada" if rank == 1 else f"Eliminada (Ranking: {rank})"
            print(f"{i}. {feature}: {status}")

        # Estatísticas das features
        print("\nEstatísticas de seleção de features:")
        print(f"Total de features originais: {len(feature_cols)}")
        print(f"Total de features selecionadas: {len(selected_names_final)}")
        print(f"Porcentagem de redução: {((len(feature_cols) - len(selected_names_final)) / len(feature_cols) * 100):.1f}%")

        # Análise por canal
        ch3_features = [f for f in selected_names_final if 'channel3' in f]
        ch4_features = [f for f in selected_names_final if 'channel4' in f]
        print(f"\nFeatures do Canal 3: {len(ch3_features)}")
        for i, feat in enumerate(ch3_features, 1):
            print(f"  {i}. {feat}")
        print(f"\nFeatures do Canal 4: {len(ch4_features)}")
        for i, feat in enumerate(ch4_features, 1):
            print(f"  {i}. {feat}")
        
        return results, feature_selection_counts, final_grid_search.best_estimator_, scaler_final, selector_final
        
    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def predict_with_pipeline(X, model, scaler, selector):
    """Função para fazer previsões usando o pipeline completo"""
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    return model.predict(X_selected)

if __name__ == "__main__":
    input_file = '/Users/joaomachado/Desktop/IC_V3/eeg_features_reorganized_20250422-191539.csv'
    n_features = 12  # Número de features a serem selecionadas
    
    results, feature_counts, final_model, final_scaler, final_selector = train_and_evaluate_models(input_file, n_features)
    
    if results is not None:
        print("\nTodos os modelos foram treinados com sucesso!")
