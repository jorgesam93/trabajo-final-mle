import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, 'cargado correctamente con columnas:', df.columns.tolist())  # Verificar las columnas cargadas
    return df

# Entrenamiento del modelo
def model_training(df):
    # Verificar que la columna 'label' exista
    if 'label' not in df.columns:
        raise ValueError("La columna 'label' no está presente en el DataFrame. Verifique el archivo de entrada.")
    
    # Codificación de etiquetas
    le_model = LabelEncoder()
    df['label'] = le_model.fit_transform(df['label'])
    
    # División del conjunto de datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        df['preprocessed_text'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Creación del clasificador con RandomForest
    clf = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer(ngram_range=(1, 3))),
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Entrenamiento del modelo
    clf.fit(x_train, y_train)
    print('Modelo entrenado con éxito')
    
    # Guardar el modelo entrenado
    package = '../models/text_classifier.pkl'
    pickle.dump(clf, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')
    return clf

# Ejecución completa del entrenamiento
def main():
    df = read_file_csv('processed_train.csv')  # Usar la nueva matriz de entrenamiento
    model_training(df)
    print('Finalizó el entrenamiento del Modelo')

if __name__ == "__main__":
    main()
