# Código de Evaluación - Clasificación de Texto
############################################################################

import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix

# Cargar la tabla transformada y evaluar el modelo
def eval_model(filename):
    # Cargar el archivo de evaluación
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, 'cargado correctamente')

    # Cargar el modelo entrenado
    package = '../models/text_classifier.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Obtener un ejemplo de prueba para evaluar
    test_text = df['preprocessed_text'][10]
    true_label = df['label'][10]
    print(f"Texto de Prueba: {test_text} ===> Etiqueta Verdadera: {true_label}")

    # Preprocesar el texto de prueba (si es necesario)
    test_text_processed = [test_text]  # Ya está preprocesado en este contexto

    # Realizar la predicción
    predicted_label = model.predict(test_text_processed)

    # Definir las clases de etiquetas
    classes = ['Irrelevant', 'Natural', 'Negative', 'Positive']

    # Mostrar resultados de la evaluación
    print(f"Etiqueta Verdadera: {true_label}")
    print(f"Etiqueta Predicha: {classes[predicted_label[0]]}")

# Evaluación desde el inicio
def main():
    eval_model('processed_score.csv')  # Usar la nueva matriz de scoring
    print('Finalizó la Evaluación del Modelo')

if __name__ == "__main__":
    main()
