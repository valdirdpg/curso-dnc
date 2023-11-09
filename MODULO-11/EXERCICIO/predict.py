import pickle
import pandas as pd

# Carrega modelo
def load_model():
    with open('model_trainer_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Carrega dados
def load_data():
    new_data = pd.DataFrame([[5.1,4.5,6,3.8]])
    return new_data

#constuir predição
def predict(data,model):
    return model.predict(data)

# Escrecer resultado
def write_result(result):
    print(result)

#Orquestrar
def run():
    model = load_model()
    data = load_data()
    result = predict(data,model)
    write_result(result)

# Executa
if __name__ == '__main__':
    run()
