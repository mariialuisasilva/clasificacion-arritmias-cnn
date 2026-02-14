# Clasificación de arritmias mediante CNN y escalogramas (CWT)
Este repositorio contiene el desarrollo de mi **Trabajo Fin de Grado (UHU)**, centrado en la detección automática de patologías cardíacas utilizando técnicas avanzadas de **Deep Learning** y procesamiento de señales.

## 📊 Resumen del proyecto
El sistema transforma señales ECG 1D en **escalogramas 2D** mediante la Transformada Wavelet Continua (CWT), permitiendo que una red neuronal convolucional (**Inception-v3**) extraiga patrones complejos de tiempo-frecuencia.

### 🚀 Logros principales:
* **Precisión (Accuracy): 95,67%** utilizando una estrategia de votación multicanal.
* **Arquitectura:** Fine-tuning sobre Inception-v3 con capas densas personalizadas.
* **Procesamiento:** Uso de Wavelet *db4* para la generación de imágenes espectrales.
* **Dataset:** Entrenamiento con más de 10.000 registros del estudio de la Chapman University (Nature Sci Rep).

## 📂 Estructura del repositorio
* **Modelos y experimentación:** Notebooks (.ipynb) que incluyen las diferentes pruebas realizadas:
  - Arquitecturas **simple** vs. **compleja**.
  - Entrenamiento **con y sin aumento de datos (AD)**.
  - Modelos específicos por derivación (Lead II, Lead III, V1).
* **📁 evaluación multicanal votación:** Contiene los archivos de predicciones por derivación (*.npy*) y el script *utils.py* para la lógica de votación final.
* **📁 modelos:** Carpeta destinada a guardar los pesos del modelo.
* **📁 tablas resultados:** Almacena los archivos Excel y resultados de las métricas.
* **📄 Memoria.pdf:** Documento completo del Trabajo Fin de Grado.

## 🛠️ Tecnologías utilizadas
* **Lenguaje:** Python 3.10
* **Librerías:** TensorFlow, Keras, PyWavelets, Scikit-learn, Pandas, NumPy.
