### IMPORTS ###
import os

from tensorflow.keras.models import load_model

### VARIABLES ###

# Definir el tamaño de la imagen
IMG_SIZE = (299, 299)  # Tamaño de la imagen para el modelo InceptionV3

# Canal a utilizar (1-12)
CANALES = [2, 3, 7]

# Tamaño del batch para el generador de datos
BATCH_SIZE = 128

### FUNCIONES ###

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
    
# Función para separar los canales en arrays
def separar_por_canales(df):
    # Crear un diccionario para almacenar los arrays de cada canal
    canales = {f"{i:02d}": [] for i in range(1, 13)}
    
    # Iterar sobre cada canal y filtrar las filas correspondientes
    for canal in canales.keys():
        canales[canal] = df[df['Filepath'].str.endswith(f"_{canal}.png")].reset_index(drop=True)
    
    # Convertir el diccionario en una lista de DataFrames
    return [canales[canal] for canal in canales.keys()]