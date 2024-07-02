import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tkinter import *
import numpy as np
import joblib
import json
import os
from tkinter import messagebox  # Para mostrar mensajes de error

# Definir 'features' globalmente para la predicción meteorológica
features_weather = [
    'Temperatura del aire HC [°C] - promedio',
    'Punto de Rocío [°C] - promedio',
    'Radiación solar [W/m2] - promedio',
    'Humedad relativa HC [%] - promedio',
    'Velocidad de Viento [m/s] - promedio'
]

def load_data_from_json(file_path):
    print("Cargando datos desde JSON...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convertir los datos JSON en un DataFrame de pandas
    records = []
    for record in data:
        flat_record = {
            'Fecha / Hora': pd.to_datetime(record['date']['$date'], utc=True),
            'Temperatura del aire HC [°C] - promedio': record['sensors']['hCAirTemperature']['avg'],
            'Punto de Rocío [°C] - promedio': record['sensors']['dewPoint']['avg'],
            'Radiación solar [W/m2] - promedio': record['sensors']['solarRadiation']['avg'],
            'Humedad relativa HC [%] - promedio': record['sensors']['hCRelativeHumidity']['avg'],
            'Velocidad de Viento [m/s] - promedio': record['sensors']['uSonicWindSpeed']['avg'],
            'Dirección de Viento [deg]': record['sensors']['uSonicWindDir']['last'],
            'Precipitación [mm]': record['sensors']['precipitation']['sum']
        }
        records.append(flat_record)
    
    df = pd.DataFrame(records)
    print(f"Datos cargados: {df.shape}")
    return df

def preprocess_data_weather(df):
    print("Preprocesando datos...")
    df = df.dropna(subset=features_weather + ['Dirección de Viento [deg]'])
    X = df[features_weather].values
    y_temp = df['Temperatura del aire HC [°C] - promedio'].values
    y_precipitation = df['Precipitación [mm]'].values
    y_humidity = df['Humedad relativa HC [%] - promedio'].values
    y_wind_direction = df['Dirección de Viento [deg]'].values
    print(f"Datos preprocesados: {X.shape}")
    return X, y_temp, y_precipitation, y_humidity, y_wind_direction

def train_models(X_train, y_train_temp, y_train_precipitation, y_train_humidity, y_train_wind_direction):
    print("Entrenando modelos...")
    models = {}
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X_train, y_train_temp)
    models['temperatura'] = model_temp
    joblib.dump(model_temp, 'model_temp.pkl')
    print("Modelo de temperatura entrenado y guardado.")

    model_precipitation = RandomForestRegressor(n_estimators=100, random_state=42)
    model_precipitation.fit(X_train, y_train_precipitation)
    models['precipitacion'] = model_precipitation
    joblib.dump(model_precipitation, 'model_precip.pkl')
    print("Modelo de precipitación entrenado y guardado.")

    model_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
    model_humidity.fit(X_train, y_train_humidity)
    models['humedad'] = model_humidity
    joblib.dump(model_humidity, 'model_humidity.pkl')
    print("Modelo de humedad entrenado y guardado.")

    model_wind_direction = RandomForestRegressor(n_estimators=100, random_state=42)
    model_wind_direction.fit(X_train, y_train_wind_direction)
    models['direccion_viento'] = model_wind_direction
    joblib.dump(model_wind_direction, 'model_wind_direction.pkl')
    print("Modelo de dirección del viento entrenado y guardado.")

    return models

def predict_future_weather(models, df, future_date):
    print(f"Prediciendo para la fecha {future_date}...")
    future_date = pd.to_datetime(future_date, format='%d-%m-%Y', utc=True)
    df['Fecha / Hora'] = df['Fecha / Hora'].dt.tz_localize(None).dt.tz_localize('UTC')  # Convertir a UTC si no está tz-aware
    last_data = df[df['Fecha / Hora'] <= future_date].tail(1)

    if last_data.empty:
        raise ValueError("No hay datos disponibles para la fecha seleccionada.")

    last_features = last_data[features_weather].values
    predictions = {}

    for key, model in models.items():
        prediction = model.predict(last_features)[0]
        predictions[key] = prediction

    print(f"Predicciones: {predictions}")
    return predictions

def save_predictions_to_json(predictions, future_date, file_name="predicciones.json"):
    predictions['fecha'] = future_date
    with open(file_name, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    print(f"Predicciones guardadas en {file_name}")

def main():
    file_path = 'RAFstationdata.json'
    df = load_data_from_json(file_path)
    X_weather, y_temp, y_precipitation, y_humidity, y_wind_direction = preprocess_data_weather(df)

    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X_weather, y_temp, test_size=0.2, random_state=42)
    _, _, y_train_precipitation, y_test_precipitation = train_test_split(X_weather, y_precipitation, test_size=0.2, random_state=42)
    _, _, y_train_humidity, y_test_humidity = train_test_split(X_weather, y_humidity, test_size=0.2, random_state=42)
    _, _, y_train_wind_direction, y_test_wind_direction = train_test_split(X_weather, y_wind_direction, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train_temp, y_train_precipitation, y_train_humidity, y_train_wind_direction)

    root = Tk()
    root.title("Predicción Meteorológica Futura")

    Label(root, text="Ingrese una fecha futura (DD-MM-YYYY):").pack()
    entry_date = Entry(root)
    entry_date.pack()

    result_label = Label(root, text="")
    result_label.pack()

    def predict_button_clicked():
        future_date = entry_date.get()
        try:
            # Verificar el formato de la fecha
            pd.to_datetime(future_date, format='%d-%m-%Y')
            predictions = predict_future_weather(models, df, future_date)
            result_text = f"Predicción para {future_date}:\n"
            result_text += f"Temperatura: {predictions['temperatura']:.2f} °C\n"
            result_text += f"Precipitación: {predictions['precipitacion']:.2f} mm\n"
            result_text += f"Humedad: {predictions['humedad']:.2f} %\n"
            result_text += f"Dirección del Viento: {predictions['direccion_viento']:.2f} grados"
            result_label.config(text=result_text)
            save_predictions_to_json(predictions, future_date)
        except ValueError:
            messagebox.showerror("Error de Formato", "Por favor, ingrese la fecha en el formato DD-MM-YYYY.")

    Button(root, text="Predecir", command=predict_button_clicked).pack()

    root.mainloop()

if __name__ == "__main__":
    main()
