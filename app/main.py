from models import SensorInput
from fastapi import FastAPI
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.linalg import inv
import math

app = FastAPI()

class ExtendedKalmanFilter:
    def __init__(self):
        # Estado inicial: [posición, velocidad, aceleración]
        self.state = np.zeros(3)
        
        # Matriz de covarianza inicial del estado
        self.P = np.eye(3) * 1000
        
        # Matriz de covarianza del ruido de proceso
        self.Q = np.eye(3) * 0.1
        
        # Matriz de covarianza del ruido de medición
        self.R = np.eye(3) * 1.0
        
        # Matriz de transición de estado (linealizada)
        self.F = np.eye(3)
        
        # Matriz de medición (linealizada)
        self.H = np.eye(3)
    
    def predict(self, dt):
        """
        Etapa de predicción del Filtro de Kalman Extendido
        dt: intervalo de tiempo entre mediciones
        """
        # Actualizar matriz de transición de estado 
        self.F[0, 1] = dt
        self.F[0, 2] = 0.5 * dt**2
        self.F[1, 2] = dt
        
        # Predicción del estado
        self.state = np.dot(self.F, self.state)
        
        # Predicción de la covarianza
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.state
    
    def update(self, measurement):
        """
        Etapa de actualización del Filtro de Kalman Extendido
        measurement: vector de mediciones [ax, ay, az]
        """
        # Calcular la ganancia de Kalman
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), inv(S))
        
        # Innovación (diferencia entre medición y predicción)
        z = measurement - np.dot(self.H, self.state)
        
        # Actualizar estado
        self.state = self.state + np.dot(K, z)
        
        # Actualizar covarianza
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        return self.state

def calculate_orientation(ax, ay, az):
    """
    Calcular ángulos de orientación (roll, pitch, yaw) 
    a partir de las lecturas del acelerómetro
    """
    # Calcular ángulo de pitch (inclinación hacia adelante/atrás)
    pitch = math.atan2(ax, math.sqrt(ay**2 + az**2))
    
    # Calcular ángulo de roll (inclinación lateral)
    roll = math.atan2(ay, math.sqrt(ax**2 + az**2))
    
    return roll, pitch

def process_ekf_sensor_data(sensor_data):
    """
    Procesar datos de sensores usando Filtro de Kalman Extendido
    """
    # Inicializar Filtro de Kalman Extendido
    ekf = ExtendedKalmanFilter()
    
    # Resultados de procesamiento
    processed_results = {
        'sensor_states': [],
        'orientations': []
    }
    
    # Procesar datos de cada sensor
    for sensor_name, sensor_samples in sensor_data.sensors.items():
        sensor_states = []
        sensor_orientations = []
        
        # Ordenar muestras por timestamp
        sorted_samples = sorted(
            sensor_samples.items(), 
            key=lambda x: x[1].timestamp
        )
        
        prev_timestamp = None
        
        for sample_key, sample in sorted_samples:
            # Extraer datos del acelerómetro
            ax, ay, az = sample.ax, sample.ay, sample.az
            
            # Calcular orientación
            roll, pitch = calculate_orientation(ax, ay, az)
            sensor_orientations.append({
                'roll': roll,
                'pitch': pitch
            })
            
            # Calcular intervalo de tiempo
            if prev_timestamp is not None:
                dt = (sample.timestamp - prev_timestamp) / 1000.0  # convertir a segundos
            else:
                dt = 0.05  # valor por defecto si es el primer punto
            
            # Etapa de predicción del EKF
            predicted_state = ekf.predict(dt)
            
            # Preparar medición para actualización
            measurement = np.array([ax, ay, az])
            
            # Etapa de actualización del EKF
            updated_state = ekf.update(measurement)
            
            sensor_states.append({
                'predicted_state': predicted_state.tolist(),
                'updated_state': updated_state.tolist()
            })
            
            prev_timestamp = sample.timestamp
        
        processed_results['sensor_states'].append({
            'sensor_name': sensor_name,
            'states': sensor_states
        })
        processed_results['orientations'].append({
            'sensor_name': sensor_name,
            'orientations': sensor_orientations
        })
    
    return processed_results


@app.post("/process-sensor-data/")
async def process_sensor_data(data: SensorInput):
    # Extraemos todos los valores de `ax` para el análisis FFT
    signal = []
    for sensor, samples in data.sensors.items():
        for sample_key, sample_data in samples.items():
            signal.append(sample_data.ax)

    # Convertimos a numpy array y procesamos la señal
    signal = np.array(signal)
    fs = 20  # Frecuencia de muestreo en Hz (ajustar según el caso)

    # Centrado de la señal eliminando el promedio
    signal_mean = np.mean(signal)
    signal = signal - signal_mean

    # Cálculo de la FFT
    spectrum = fft(signal)
    freqs = fftfreq(len(signal), 1 / fs)  # Calcular frecuencias en Hz

    # Obtenemos la mitad de las frecuencias (componente positiva)
    start = len(spectrum) // 2
    freqs_values = freqs[:start].tolist()
    spectrum_values = np.abs(spectrum[:start]).tolist()
    
    # Realizar análisis EKF
    ekf_results = process_ekf_sensor_data(data)

    # Devuelve las frecuencias y el espectro de amplitud
    return {
        "fft_analysis": {
            "frequencies": freqs_values, 
            "spectrum": spectrum_values
        },
        "ekf_analysis": ekf_results
    }
