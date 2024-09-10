import pandas as pd
import pickle
import os

# Cargar la tabla transformada y realizar el scoring
def score_model(filename, output_filename):
    # Cargar el archivo de scoring
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, 'cargado correctamente')

    # Eliminar filas con NaN en 'preprocessed_text'
    df.dropna(subset=['preprocessed_text'], inplace=True)

    # Cargar el modelo entrenado
    package = '../models/text_classifier.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Obtener los textos para scoring
    X_score = df['preprocessed_text']

    # Realizar las predicciones
    y_pred = model.predict(X_score)

    # Guardar las predicciones en un archivo CSV
    output_path = os.path.join('../data/scores/', output_filename)
    pred = pd.DataFrame({'preprocessed_text': X_score, 'predicted_label': y_pred})
    pred.to_csv(output_path, index=False)
    print(output_filename, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    score_model('processed_score.csv', 'scoring_results.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()

