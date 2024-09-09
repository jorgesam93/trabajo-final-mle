# Código de Scoring - Clasificación de Texto
############################################################################

import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report

# Cargar la tabla transformada y realizar predicciones
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, 'cargado correctamente')

    # Leemos el modelo entrenado para usarlo
    package = '../models/text_classifier.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Separar características y etiquetas verdaderas
    x_test = df['preprocessed_text']
    y_test = df['label']

    # Predecimos sobre el conjunto de datos de prueba
    y_pred = model.predict(x_test)
    
    # Guardamos las predicciones
    pred = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred})
    pred.to_csv(os.path.join('../data/scores/', scores), index=False)
    print(scores, 'exportado correctamente en la carpeta scores')

    # Imprimir métricas de evaluación
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

# Scoring desde el inicio
def main():
    score_model('processed_val.csv', 'final_score.csv')  # Usar la nueva matriz de validación
    print('Finalizó el Scoring del Modelo')

if __name__ == "__main__":
    main()
