# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os
import spacy

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, 'cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Eliminamos vacíos
    df.dropna(inplace=True)
    # Cargar idioma inglés
    nlp = spacy.load("en_core_web_sm")
    
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
    df['preprocessed_text'] = df['text'].apply(preprocess)
    
    print('Transformación de datos completa')
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('dataset.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['text', 'preprocessed_text'], 'processed_train.csv')
    
    # Matriz de Validación
    df2 = read_file_csv('dataset_validation.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['text', 'preprocessed_text'], 'processed_val.csv')
    
    # Matriz de Scoring
    df3 = read_file_csv('dataset_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['text', 'preprocessed_text'], 'processed_score.csv')

if __name__ == "__main__":
    main()
