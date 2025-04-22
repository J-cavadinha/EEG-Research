import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import time

def create_selector(n_features_to_select=10):
    """Cria e retorna um seletor RFE"""
    estimator = SVC(kernel="linear")
    return RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)

def fit_selector(selector, X, y):
    """Ajusta o seletor aos dados de treino"""
    return selector.fit(X, y)

def transform_features(selector, X):
    """Aplica a seleção de features aos dados"""
    return selector.transform(X)

def get_selected_feature_indices(selector, feature_names):
    """Retorna os índices e nomes das features selecionadas"""
    selected_indices = np.where(selector.support_)[0]
    selected_names = [feature_names[i] for i in selected_indices]
    return selected_indices, selected_names

def get_feature_ranking(selector, feature_names):
    """Retorna o ranking das features"""
    return [(feature, rank) for feature, rank in zip(feature_names, selector.ranking_)]

def apply_rfe_for_final_model(X, y, feature_names, n_features_to_select=10):
    """Aplica RFE ao conjunto completo para o modelo final"""
    try:
        estimator = SVC(kernel="linear")
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X, y)
        
        selected_indices, selected_names = get_selected_feature_indices(selector, feature_names)
        feature_ranking = get_feature_ranking(selector, feature_names)
        
        return selector, selected_indices, selected_names, feature_ranking
    except Exception as e:
        print(f"Erro durante a seleção de features: {e}")
        return None, None, None, None
