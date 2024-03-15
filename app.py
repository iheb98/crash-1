from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Charger les modèles entraînés
final_model_injured = joblib.load('final_model_injured.pkl')
final_model_pedestrians = joblib.load('final_model_pedestrians.pkl')
final_model_cyclists = joblib.load('final_model_cyclists.pkl')
final_model_motorists = joblib.load('final_model_motorists.pkl')

@app.route('/estimate_injuries', methods=['GET'])
def estimate_injuries():
    # Charger les données une fois au démarrage du serveur Flask
    data = pd.read_csv('datacollision.csv')

    # Supprimer les lignes avec des valeurs manquantes dans les colonnes cruciales
    columns_to_clean = ['CRASH DATE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED']
    data_cleaned = data.dropna(subset=columns_to_clean).copy()

    # Convertir la colonne 'CRASH DATE' en format de date
    data_cleaned['CRASH DATE'] = pd.to_datetime(data_cleaned['CRASH DATE'])

    # Récupérer les dates de début et de fin à partir des paramètres de requête
    start_date = pd.to_datetime(request.args.get('start_date'))
    end_date = pd.to_datetime(request.args.get('end_date'))

    # Créer un ensemble de dates pour la prédiction
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prediction_data = pd.DataFrame({'DAYS_SINCE_START': (prediction_dates - data_cleaned['CRASH DATE'].min()).days})
    
    # Faire des prédictions pour chaque type de blessure
    predictions_injured = final_model_injured.predict(prediction_data.values)
    predictions_pedestrians = final_model_pedestrians.predict(prediction_data.values)
    predictions_cyclists = final_model_cyclists.predict(prediction_data.values)
    predictions_motorists = final_model_motorists.predict(prediction_data.values)
    
    # Somme des prédictions pour obtenir une estimation du nombre total de blessures pour chaque type
    total_injured = int(round(predictions_injured.sum()))
    total_pedestrians = int(round(predictions_pedestrians.sum()))
    total_cyclists = int(round(predictions_cyclists.sum()))
    total_motorists = int(round(predictions_motorists.sum()))
    
    # Retourner le résultat en format JSON
    return jsonify({'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'estimated_total_injured': total_injured,
                    'estimated_total_pedestrians': total_pedestrians,
                    'estimated_total_cyclists': total_cyclists,
                    'estimated_total_motorists': total_motorists})

if __name__ == '__main__':
    app.run(debug=True)
