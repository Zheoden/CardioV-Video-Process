# ------------------------
# Cargando modelo de disco
# ------------------------

import tensorflow as tf
from keras.models import load_model
from model.utilities import focal_tversky, tversky

def cargarModelo():
    FILENAME_MODEL_TO_LOAD = "model_full.h5"
    MODEL_PATH = "../model"

    # Cargar la RNA desde disco
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD, custom_objects={"focal_tversky": focal_tversky,"tversky": tversky})
    print("Modelo cargado de disco << ", loaded_model)

    #graph = tf.compat.v1.get_default_graph()
    return loaded_model
