from models import SensorInput
from fastapi import FastAPI
from scipy.fft import fft, fftfreq
import numpy as np

app = FastAPI()


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

    # Devuelve las frecuencias y el espectro de amplitud
    return {"frequencies": freqs_values, "spectrum": spectrum_values}
