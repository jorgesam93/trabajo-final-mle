import pandas as pd
import numpy as np
import os
import spacy
from csv import QUOTE_ALL

# Leemos los archivos csv
def read_file_csv(filename):
    # Leer el archivo CSV ignorando las comas dentro del texto
    columns = ['id', 'country', 'label', 'text']
    df = pd.read_csv(
        os.path.join('../data/raw/', filename),
        header=None,
        names=columns,
        quotechar='"',  # Manejar las comillas dobles
        quoting=QUOTE_ALL,  # Ignorar comas dentro de comillas
        on_bad_lines='skip',  # Ignorar líneas problemáticas
        engine='python'  # Usar el motor de Python para mayor flexibilidad
    )
    print(filename, 'cargado correctamente con columnas:', df.columns.tolist())
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    print('Iniciando la preparación de datos...')
    df.dropna(subset=['text', 'label'], inplace=True)  # Eliminar filas con NaN en 'text' o 'label'
    print('Datos vacíos eliminados.')

    # Cargar idioma inglés
    try:
        nlp = spacy.load("en_core_web_sm")
        print('Modelo de SpaCy cargado correctamente.')
    except Exception as e:
        print(f'Error al cargar el modelo de SpaCy: {e}')
        return df

    # Preprocesamiento de texto
    def preprocess(text):
        doc = nlp(text)
        filtered_token = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_token.append(token.lemma_)
        return " ".join(filtered_token)

    # Aplicamos el preprocesamiento a la columna 'text'
    try:
        df['preprocessed_text'] = df['text'].apply(preprocess)
        print('Preprocesamiento de texto completado.')
    except Exception as e:
        print(f'Error durante el preprocesamiento de texto: {e}')
        return df

    # Reemplazar o eliminar cualquier fila con NaN en 'preprocessed_text'
    df['preprocessed_text'].replace('', np.nan, inplace=True)  # Reemplazar cadenas vacías con NaN
    df.dropna(subset=['preprocessed_text'], inplace=True)  # Eliminar filas con NaN en 'preprocessed_text'
    
    print('Transformación de datos completa')
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    try:
        dfp = df[features]
        dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
        print(filename, 'exportado correctamente en la carpeta processed')
    except Exception as e:
        print(f'Error al exportar el archivo {filename}: {e}')

# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('dataset.csv')
    tdf1 = data_preparation(df1)
    # Incluimos la columna 'label' para entrenamiento
    data_exporting(tdf1, ['text', 'preprocessed_text', 'label'], 'processed_train.csv')

    # Matriz de Validación
    df2 = read_file_csv('dataset_validation.csv')
    tdf2 = data_preparation(df2)
    # Incluimos la columna 'label' para validación
    data_exporting(tdf2, ['text', 'preprocessed_text', 'label'], 'processed_val.csv')

    # Matriz de Scoring
    df3 = read_file_csv('dataset_score.csv')
    tdf3 = data_preparation(df3)
    # La columna 'label' no es necesaria para scoring
    data_exporting(tdf3, ['text', 'preprocessed_text'], 'processed_score.csv')

if __name__ == "__main__":
    main()

