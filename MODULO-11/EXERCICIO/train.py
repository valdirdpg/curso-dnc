import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Carrega dados
def dataframe_data():
    data = load_iris()
    return data

# Separa dados em X e y
def dataframe_target_feature(data):    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])
    return X, y

# Treina modelo
def cassifier_model(X, y):
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    return model

# Serializa modelo
def serialize_model(model):
    with open('model_trainer_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

# Carrega modelo
def load_model():
    data = dataframe_data()
    X, y = dataframe_target_feature(data)
    model = cassifier_model(X, y)
    serialize_model(model)

# Executa
if __name__ == '__main__':
    load_model()

