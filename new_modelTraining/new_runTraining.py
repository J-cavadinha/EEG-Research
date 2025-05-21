# run_training.py
import new_train_model
import os # Added for path joining if needed

# !!! UPDATE THIS to the actual path of your NEW features CSV file !!!
# This file is the output from the "optimal" extract_features.py script
# Example: 'eeg_features_optimal/optimal_logbandpower_features_YYYYMMDD-HHMMSS.csv'
INPUT_FEATURES_CSV = "/Users/joaomachado/Desktop/pipeline/eeg_features_optimal/processed_epochs_20250515-235629_optimal_logbandpower_features.csv"

# Number of features to select using RFE.
# Since we have 4 "optimal" features (C3_log_mu, C3_log_beta, C4_log_mu, C4_log_beta),
# setting this to 4 means RFE will use all of them (which is fine if you want to keep the RFE step).
# If you wanted to test if a subset (e.g., 2 or 3) is better, you could change this.
# If N_FEATURES > actual number of features, the code in train_model.py will adjust.
N_FEATURES_TO_SELECT = 4 # Using all 4 optimal features

# --- Execution ---
print("--- Iniciando Pipeline de Treinamento de Modelo SVM (com features otimizadas) ---")
print(f"Arquivo de entrada de features: {INPUT_FEATURES_CSV}")
print(f"Número de features a serem usadas/selecionadas pelo RFE: {N_FEATURES_TO_SELECT}")
print("-" * 50)

if not os.path.exists(INPUT_FEATURES_CSV):
    print(f"ERRO: Arquivo de features de entrada não encontrado: {INPUT_FEATURES_CSV}")
    print("Por favor, execute o script de extração de features otimizadas primeiro e atualize o caminho.")
else:
    final_model, final_scaler, final_selector, cv_mean_acc, final_model_acc = \
        new_train_model.train_and_evaluate_svm_model(
            input_file=INPUT_FEATURES_CSV,
            n_features_to_select=N_FEATURES_TO_SELECT # Pass this to the training function
        )

    if final_model is not None:
        print("\n--- Pipeline de Treinamento Concluído com Sucesso ---")
        print(f"Acurácia média da CV (Cross-Validation): {cv_mean_acc:.4f}")
        if final_model_acc is not None: # final_model_acc might be from inner CV of GridSearchCV
             print(f"Acurácia do modelo final (estimada pelo CV interno do GridSearchCV na totalidade dos dados selecionados): {final_model_acc:.4f}")
        print("Modelo, Scaler, Selector (se RFE foi efetivamente usado para seleção) e ordem das features foram salvos.")
    else:
        print("\n--- Pipeline de Treinamento Falhou ---")