### IMPORTS ###
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

### VARIABLES ###

# Definir el tamaño de la imagen
IMG_SIZE = (299, 299)  # Tamaño de la imagen para el modelo InceptionV3

# Canal a utilizar (1-12)
CANAL = 2

# Definir el tamaño del batch
BATCH_SIZE_TRAIN = 8  # Tamaño del batch para el conjunto de entrenamiento
BATCH_SIZE_VAL = 128 # Tamaño del batch para el conjunto de validación
BATCH_SIZE_TEST = 128 # Tamaño del batch para el conjunto de prueba

# Hiperparámetos iniciales del entrenamiento
LEARNING_RATE = 1e-4
BETA_1 = 0.9
BETA_2 = 0.999
EPOCH = 500 # Número de épocas para el entrenamiento
NUM_CLASES = 4

# Definir el número de neuronas para las capas densas
NUM_NEURONAS_DENSE1 = 512  # Número de neuronas para la primera capa densa
NUM_NEURONAS_DENSE2 = 256  # Número de neuronas para la segunda capa densa
NUM_NEURONAS_DENSE3 = 128  # Número de neuronas para la tercera capa densa

# Porcentajes de Dropout para las capas densas
DROPOUT1 = 0.5  # Dropout para la primera capa densa
DROPOUT2 = 0.4  # Dropout para la segunda capa densa
DROPOUT3 = 0.3  # Dropout para la tercera capa densa

# Definir la paciencia para Early Stopping
PATIENCE_FASE1 = 20  # Paciencia para Early Stopping en la primera fase
PATIENCE_FASE2 = 40  # Paciencia para Early Stopping en la segunda fase (fine-tuning)
                        # Aquí el modelo ajusta todas las capas, afinando las características
                        # Este proceso suele ser más lento, por lo que hay que aumentar la
                        # paciencia.
                        # Número razonable: 30-50

### FUNCIONES ###

# Función para separar los canales en arrays
def separar_por_canales(df):
    # Crear un diccionario para almacenar los arrays de cada canal
    canales = {f"{i:02d}": [] for i in range(1, 13)}
    
    # Iterar sobre cada canal y filtrar las filas correspondientes
    for canal in canales.keys():
        canales[canal] = df[df['Filepath'].str.endswith(f"_{canal}.png")].reset_index(drop=True)
    
    # Convertir el diccionario en una lista de DataFrames
    return [canales[canal] for canal in canales.keys()]

# Función para contar instancias por clase
def plot_class_distribution(df, title):
    class_counts = df['Label'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(title)
    plt.xlabel('Clases')
    plt.ylabel('Número de instancias')
    plt.xticks(rotation=90)
    plt.show()

def load_model_safely(model_path):
    try:
        # Comprobar si el archivo del modelo existe
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Modelo cargado desde {model_path}")
            return model
        else:
            print(f"El modelo no se encuentra en {model_path}")
            return None  # Retorna None si no se encuentra el modelo
    except Exception as e:
        print(f"Error al cargar el modelo desde {model_path}: {e}")
        return None  # Retorna None si ocurre cualquier otro error

# Definir F1-Score
def f1_score(recall, precision):
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
