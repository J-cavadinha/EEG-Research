# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import joblib
import os # Added for path joining

# Assuming normalize.py and feat_selection.py are in the same directory or Python path
from normalize import create_scaler, fit_scaler, transform_data 
from feat_selection import create_selector, fit_selector, transform_features, \
                           get_selected_feature_indices, get_feature_ranking, \
                           apply_rfe_for_final_model

# Output folder for saved models and artifacts
MODEL_ARTIFACTS_FOLDER = "model_artifacts"

def train_and_evaluate_svm_model(input_file, n_features_to_select=8): # Default n_features for optimal set
    try:
        # Create model artifacts folder if it doesn't exist
        if not os.path.exists(MODEL_ARTIFACTS_FOLDER):
            os.makedirs(MODEL_ARTIFACTS_FOLDER)
            print(f"[INFO] Created output folder for model artifacts: {MODEL_ARTIFACTS_FOLDER}")

        data = pd.read_csv(input_file)
        # The new CSV has 'epoch_index' and 'label'.
        # 'epoch_index' can be treated like 'trial_number' if needed for metadata, but not as a feature.
        print(f"Carregando dados com {len(data)} trials.")

        # Define metadata columns based on the new CSV format
        metadata_cols = ['epoch_index', 'label'] 
        
        # Dynamically get feature columns
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        if not feature_cols:
            print("ERRO: Nenhuma coluna de feature encontrada no arquivo de entrada!")
            return None, None, None, None, None
            
        print(f"Features originais (antes da seleção RFE, se aplicável): {len(feature_cols)} -> {feature_cols}")

        X = data[feature_cols].values
        y = data['label'].values

        # Adjust n_features_to_select if it's more than available features
        actual_n_features = X.shape[1]
        if n_features_to_select > actual_n_features:
            print(f"Aviso: n_features_to_select ({n_features_to_select}) é maior que o número de features disponíveis ({actual_n_features}).")
            print(f"Ajustando para usar todas as {actual_n_features} features (RFE selecionará todas).")
            n_features_to_select = actual_n_features
        elif n_features_to_select <= 0 : # Ensure at least 1 or all features are selected
             n_features_to_select = actual_n_features


        model_name = 'SVM'
        model_instance = SVC(probability=True, random_state=42) # Added random_state for SVC
        param_grid = {
            'C': [0.01, 0.1, 1], # Slightly expanded C range
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01] # Slightly reduced gamma
        }

        # For very small datasets (like 4 samples), KFold can be tricky.
        # Adjust n_splits based on dataset size, especially smallest class size.
        n_samples = len(y)
        unique_labels, counts = np.unique(y, return_counts=True)
        min_class_count = min(counts) if len(counts) > 0 else 0
        
        cv_splits = 7 # Default
        if n_samples < 5 or (min_class_count > 0 and min_class_count < cv_splits) : # If any class has fewer samples than splits
            cv_splits = max(2, min_class_count) # At least 2, or limited by smallest class
            print(f"Aviso: Número de splits da CV ajustado para {cv_splits} devido ao tamanho pequeno do dataset ou classes.")
        
        # If still too small for any meaningful CV (e.g. if a class has only 1 sample)
        if cv_splits < 2 and n_samples > 1: # Cannot do CV if splits < 2
            print("ERRO: Dataset muito pequeno para StratifiedKFold com pelo menos 2 splits. Consiga mais dados.")
            # Fallback: train on all, test on all (not ideal, just for code to run)
            # Or simply return and state more data is needed. For now, let's try to proceed carefully.
            # If you have only 4 samples, cv_splits will be 2.
            if n_samples <= 1: # Cannot proceed if 0 or 1 sample
                 print("ERRO: Dataset com 1 ou 0 amostras. Impossível treinar.")
                 return None, None, None, None, None


        kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        print(f"\nIniciando avaliação do modelo {model_name} com {kfold.n_splits}-Fold CV...")
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            print(f"  Avaliando Fold {fold+1}/{kfold.n_splits}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = create_scaler()
            scaler = fit_scaler(scaler, X_train) # Fit ONLY on training data
            X_train_scaled = transform_data(scaler, X_train)
            X_test_scaled = transform_data(scaler, X_test) # Transform test data with training fit

            # Apply RFE within each fold if n_features_to_select < actual_n_features
            if n_features_to_select < actual_n_features:
                print(f"    Aplicando RFE para selecionar {n_features_to_select} features no Fold {fold+1}...")
                # Pass actual_n_features from this fold's training set
                selector = create_selector(n_features_to_select=n_features_to_select, actual_n_features=X_train_scaled.shape[1])
                selector = fit_selector(selector, X_train_scaled, y_train)
                X_train_selected = transform_features(selector, X_train_scaled)
                X_test_selected = transform_features(selector, X_test_scaled)
            else: # Use all features (RFE step is effectively skipped for selection)
                print(f"    Usando todas as {actual_n_features} features no Fold {fold+1} (RFE não está reduzindo features).")
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
                selector = None # No actual selector used for transformation

            # Inner CV for hyperparameter tuning within this fold
            inner_cv_splits = max(2, min(3, min_class_count -1 if min_class_count >1 else 2)) # Adjust inner CV splits too
            if X_train_selected.shape[0] < inner_cv_splits * 2 : # Basic check
                 inner_cv_splits = max(2, X_train_selected.shape[0] // 2 if X_train_selected.shape[0] >1 else 2)


            if X_train_selected.shape[0] < 2 or (hasattr(y_train, 'shape') and y_train.shape[0] < 2):
                print("    Aviso: Conjunto de treino do fold muito pequeno para GridSearchCV. Pulando o fold.")
                fold_scores.append(0.0) # Or handle as NaN or skip
                continue

            # Ensure y_train has enough samples for inner_cv_splits if it's small
            unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
            min_class_count_y_train = min(counts_y_train) if len(counts_y_train) > 0 else 0
            
            effective_inner_cv_splits = inner_cv_splits
            if min_class_count_y_train > 0 and min_class_count_y_train < inner_cv_splits:
                effective_inner_cv_splits = max(2, min_class_count_y_train)


            if X_train_selected.shape[0] < effective_inner_cv_splits : # Check if enough samples for GridSearch CV
                 print(f"    Aviso: Não há amostras suficientes ({X_train_selected.shape[0]}) para GridSearchCV com {effective_inner_cv_splits} splits. Usando modelo padrão.")
                 best_model_fold = model_instance # Use default params
                 best_model_fold.fit(X_train_selected, y_train)

            else:
                grid_search = GridSearchCV(
                    estimator=model_instance,
                    param_grid=param_grid,
                    cv= StratifiedKFold(n_splits=effective_inner_cv_splits, shuffle=True, random_state=123), # Use StratifiedKFold for inner CV
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                try:
                    grid_search.fit(X_train_selected, y_train)
                    best_model_fold = grid_search.best_estimator_
                except ValueError as gsv_e:
                    print(f"    Erro no GridSearchCV no Fold {fold+1}: {gsv_e}. Usando modelo padrão.")
                    best_model_fold = model_instance # Fallback to default params
                    best_model_fold.fit(X_train_selected, y_train)


            y_pred = best_model_fold.predict(X_test_selected)
            fold_accuracy = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_accuracy)
            print(f"    Acurácia do Fold {fold+1}: {fold_accuracy:.4f}")

        mean_accuracy_cv = np.mean(fold_scores) if fold_scores else 0.0
        std_accuracy_cv = np.std(fold_scores) if fold_scores else 0.0

        print(f"\nResultados da Avaliação para {model_name}:")
        print(f"Acurácia média da validação cruzada ({kfold.n_splits} folds): {mean_accuracy_cv:.4f} ± {std_accuracy_cv:.4f}")

        print("\nTreinando modelo SVM final com todo o conjunto de dados (ou o subconjunto de features selecionado)...")
        
        scaler_final = create_scaler()
        X_scaled_final = scaler_final.fit_transform(X) # Fit and transform on the entire dataset X

        selector_final_obj = None # Initialize
        selected_names_final = feature_cols # Default to all original feature names
        X_for_final_model = X_scaled_final # Default to all scaled features

        if n_features_to_select < actual_n_features:
            print(f"Aplicando RFE final para selecionar {n_features_to_select} features...")
            selector_final_obj, selected_indices_final, selected_names_final, feature_ranking_final = apply_rfe_for_final_model(
                X_scaled_final, y, feature_cols, n_features_to_select=n_features_to_select
            )
            if selector_final_obj is None:
                print("Erro durante a seleção final de features (RFE). O modelo final será treinado com todas as features.")
                # Keep X_for_final_model as X_scaled_final and selected_names_final as feature_cols
            else:
                print(f"Número de features selecionadas para o modelo final: {len(selected_names_final)}")
                print("Features selecionadas para o modelo final:", selected_names_final)
                X_for_final_model = selector_final_obj.transform(X_scaled_final)
        else:
            print(f"Usando todas as {actual_n_features} features para o modelo final (RFE não está reduzindo features).")
            # If keeping RFE structure for saving, fit it to learn rankings even if all are selected
            if actual_n_features > 0 : # Only if features exist
                selector_final_obj, _, selected_names_final, feature_ranking_final = apply_rfe_for_final_model(
                    X_scaled_final, y, feature_cols, n_features_to_select=actual_n_features # Select all
                )
            else: # No features to select
                selector_final_obj, feature_ranking_final = None, None


        # Final Hyperparameter Tuning on the (potentially selected) full dataset
        # Adjust final CV splits based on full dataset size
        final_cv_splits = 5
        if n_samples < 10 or (min_class_count > 0 and min_class_count < final_cv_splits) :
             final_cv_splits = max(2, min_class_count)
        
        if X_for_final_model.shape[0] < final_cv_splits : # Check if enough samples for GridSearch CV
             final_cv_splits = max(2, X_for_final_model.shape[0] // 2 if X_for_final_model.shape[0] > 1 else 2 )
             if X_for_final_model.shape[0] < 2 : # Cannot do CV if less than 2 samples
                print("ERRO: Conjunto de dados final muito pequeno para GridSearchCV. Treinando com parâmetros padrão.")
                final_svm_model = model_instance.fit(X_for_final_model, y)
                final_model_accuracy = accuracy_score(y, final_svm_model.predict(X_for_final_model)) # Train accuracy
                print(f"Acurácia do modelo final (treinamento, sem CV no GridSearch): {final_model_accuracy:.4f}")
             else : # Fallback for very small data for GridSearchCV
                final_grid_search = model_instance # Use default params, no grid search CV
                final_svm_model = final_grid_search.fit(X_for_final_model, y)
                final_model_accuracy = accuracy_score(y, final_svm_model.predict(X_for_final_model))
                print(f"Acurácia do modelo final (treinamento, GridSearchCV CV pulado): {final_model_accuracy:.4f}")

        else:
            final_grid_search = GridSearchCV(
                estimator=model_instance,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=final_cv_splits, shuffle=True, random_state=321),
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            try:
                final_grid_search.fit(X_for_final_model, y)
                final_svm_model = final_grid_search.best_estimator_
                final_model_accuracy = final_grid_search.best_score_
                print(f"\nMelhores parâmetros para o modelo SVM final: {final_grid_search.best_params_}")
                print(f"Acurácia do modelo SVM final (estimada pelo CV interno do GridSearchCV): {final_model_accuracy:.4f}")
            except ValueError as fgsv_e:
                print(f"Erro no GridSearchCV Final: {fgsv_e}. Treinando modelo com parâmetros padrão.")
                final_svm_model = model_instance.fit(X_for_final_model, y)
                final_model_accuracy = accuracy_score(y, final_svm_model.predict(X_for_final_model))
                print(f"Acurácia do modelo final (treinamento, GridSearchCV CV falhou): {final_model_accuracy:.4f}")


        # Save artifacts
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        model_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"svm_model_final_{timestamp}.joblib")
        scaler_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"svm_scaler_final_{timestamp}.joblib")
        
        joblib.dump(final_svm_model, model_filename)
        joblib.dump(scaler_final, scaler_filename)
        print(f"\nModelo SVM final salvo como: {model_filename}")
        print(f"Scaler final salvo como: {scaler_filename}")

        # Save selector and feature order only if RFE was meaningfully applied or fitted
        if selector_final_obj is not None:
            selector_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"svm_selector_final_{timestamp}.joblib")
            joblib.dump(selector_final_obj, selector_filename)
            print(f"Selector RFE final salvo como: {selector_filename}")
            
            # Save the names of the features *selected by RFE* for the final model
            final_selected_feature_names_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"FINAL_SELECTED_feature_names_{timestamp}.joblib")
            joblib.dump(selected_names_final, final_selected_feature_names_filename) 
            print(f"Nomes das features selecionadas para o modelo final salvos como: {final_selected_feature_names_filename}")

            if feature_ranking_final:
                try:
                    feature_ranking_df = pd.DataFrame(feature_ranking_final, columns=['Feature', 'Ranking'])
                    feature_ranking_df = feature_ranking_df.sort_values('Ranking')
                    feature_ranking_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"svm_feature_rankings_final_{timestamp}.csv")
                    feature_ranking_df.to_csv(feature_ranking_filename, index=False)
                    print(f"Rankings de features do modelo final salvos como: {feature_ranking_filename}")
                except Exception as rank_e:
                    print(f"Aviso: Não foi possível salvar o ranking de features: {rank_e}")
        else: # If RFE was not used for selection (e.g. using all features)
             print("Selector RFE não foi usado para redução de features no modelo final ou não foi ajustado.")


        # Save the original feature column names (input to RFE if used, or input to model if RFE skipped)
        original_feature_names_for_pipeline_filename = os.path.join(MODEL_ARTIFACTS_FOLDER, f"PIPELINE_INPUT_feature_names_{timestamp}.joblib")
        joblib.dump(feature_cols, original_feature_names_for_pipeline_filename)
        print(f"Nomes das features de entrada da pipeline salvos como: {original_feature_names_for_pipeline_filename}")

        return final_svm_model, scaler_final, selector_final_obj, mean_accuracy_cv, final_model_accuracy

    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de entrada de features não encontrado - {input_file}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Erro crítico durante o treinamento do modelo SVM: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None