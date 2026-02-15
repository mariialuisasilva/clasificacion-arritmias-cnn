# Clasificación de arritmias mediante CNN y escalogramas (CWT)
Este repositorio contiene el desarrollo de mi **Trabajo Fin de Grado (UHU)**, centrado en la detección automática de patologías cardíacas utilizando técnicas avanzadas de **Deep Learning** y procesamiento de señales.

## 📊 Resumen del proyecto
El sistema transforma señales ECG 1D en **escalogramas 2D** mediante la Transformada Wavelet Continua (CWT), permitiendo que una red neuronal convolucional (**Inception-v3**) extraiga patrones complejos de tiempo-frecuencia.

### 🚀 Logros principales:
* **Alta precisión en clasificación:** Se alcanzó un **95,67% de accuracy** en el análisis de derivaciones individuales, demostrando la eficacia de la arquitectura Inception-v3 para este tipo de señales.
* **Sistema de decisión robusto:** Implementación de un algoritmo de **votación multicanal** que consolida las predicciones de diferentes leads para obtener un diagnóstico global con un **92,81% de precisión**.
* **Aprovechamiento de Deep Learning y CWT:** Uso avanzado de imágenes obtenidas mediante la **Transformada Wavelet Continua (CWT)**, aplicando técnicas de visión por computador para la detección automática de arritmias.
* **Optimización y experimentación:** Comparativa detallada entre arquitecturas simples y complejas, incluyendo pruebas con y sin aumento de datos (Data Augmentation) para mejorar la generalización del modelo.

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

## 📊 Resultados Obtenidos
El proyecto evalúa el rendimiento tanto en canales individuales como en el sistema integrado:
* **Modelo monocanal (Best Case):** **95,67% de precisión** utilizando la arquitectura Inception-v3 con la Transformada Wavelet Continua.
* **Sistema de votación multicanal:** **92,81% de precisión**, integrando las predicciones de todas las derivaciones para un diagnóstico global más sólido.
