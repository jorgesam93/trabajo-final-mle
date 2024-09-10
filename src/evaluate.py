import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Cargar la tabla transformada y evaluar el modelo
def eval_model(filename):
    # Cargar el archivo de evaluación
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, 'cargado correctamente')

    # Verificar si la columna 'label' existe
    if 'label' in df.columns:
        # Evaluación con etiquetas verdaderas
        print("Evaluación utilizando etiquetas verdaderas...")
        
        # Eliminar filas con NaN en 'preprocessed_text' o 'label'
        df.dropna(subset=['preprocessed_text', 'label'], inplace=True)

        # Cargar el modelo entrenado
        package = '../models/text_classifier.pkl'
        model = pickle.load(open(package, 'rb'))
        print('Modelo importado correctamente')

        # Codificar las etiquetas verdaderas usando el mismo encoder utilizado en el entrenamiento
        le_model = LabelEncoder()
        df['label'] = le_model.fit_transform(df['label'])
        
        # Obtener los textos de prueba y las etiquetas verdaderas
        X_test = df['preprocessed_text']
        y_true = df['label']

        # Realizar las predicciones
        y_pred = model.predict(X_test)

        # Decodificar las etiquetas predichas al mismo formato que las etiquetas verdaderas
        y_pred = le_model.inverse_transform(y_pred)
        y_true = le_model.inverse_transform(y_true)

        # Mostrar el reporte de clasificación completo
        print("Reporte de Clasificación:")
        print(classification_report(y_true, y_pred))

        # Mostrar la matriz de confusión
        print("Matriz de Confusión:")
        print(confusion_matrix(y_true, y_pred))

        # Mostrar precisión, recall y puntaje F1
        print(f"Precisión: {accuracy_score(y_true, y_pred):.2f}")
        print(f"Precisión (Precision): {precision_score(y_true, y_pred, average='weighted'):.2f}")
        print(f"Sensibilidad (Recall): {recall_score(y_true, y_pred, average='weighted'):.2f}")

    else:
        # Evaluación sin etiquetas verdaderas (solo predicción)
        print("Evaluación sin etiquetas verdaderas, solo predicción...")
        
        # Eliminar filas con NaN en 'preprocessed_text'
        df.dropna(subset=['preprocessed_text'], inplace=True)

        # Cargar el modelo entrenado
        package = '../models/text_classifier.pkl'
        model = pickle.load(open(package, 'rb'))
        print('Modelo importado correctamente')

        # Obtener los textos de prueba
        X_test = df['preprocessed_text']

        # Realizar las predicciones
        y_pred = model.predict(X_test)

        # Mostrar algunas predicciones como ejemplo
        print("Algunas predicciones de ejemplo:")
        for i in range(min(5, len(X_test))):  # Mostrar solo las primeras 5 predicciones
            print(f"Texto: {X_test.iloc[i]}")
            print(f"Etiqueta Predicha: {y_pred[i]}")
            print("-----")

# Evaluación desde el inicio
def main():
    eval_model('processed_val.csv')  # Usar la nueva matriz de validación
    print('Finalizó la Evaluación del Modelo')

if __name__ == "__main__":
    main()
