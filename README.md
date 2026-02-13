# Clasificación de arritmias mediante CNN y escalogramas (CWT)
Este repositorio contiene el desarrollo de mi **Trabajo Fin de Grado (UHU)**, centrado en la detección automática de patologías cardíacas utilizando técnicas avanzadas de **Deep Learning** y procesamiento de señales.

## 📊 Resumen del proyecto
El sistema transforma señales ECG 1D en **escalogramas 2D** mediante la Transformada Wavelet Continua (CWT), permitiendo que una red neuronal convolucional (**Inception-v3**) extraiga patrones complejos de tiempo-frecuencia.

### 🚀 Logros principales:
* **Precisión (Accuracy): 95.67%** utilizando una estrategia de votación multicanal.

* **Arquitectura:** Fine-tuning sobre Inception-v3 con capas densas personalizadas.

* **Procesamiento:** Uso de Wavelet *db4* para la generación de imágenes espectrales.

* **Dataset:** Entrenamiento con más de 10.000 registros del estudio de la Chapman University (Nature Sci Rep).

## 📂 Estructura del repositorio
* ***/code***: Scripts de preprocesamiento, entrenamiento y evaluación.

* ***/evaluacion_multicanal_votacion***: Implementación de la lógica de decisión multicanal.

* ***Memoria.pdf***: Documentación técnica completa, metodología y estado del arte.

## 🛠️ Tecnologías utilizadas
* **Lenguaje:** Python 3.10

* **Librerías:** TensorFlow, Keras, PyWavelets, Scikit-learn, Pandas, NumPy.
