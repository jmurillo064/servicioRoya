from flask import Flask, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from keras.models import load_model
import base64
import io
from PIL import Image
from datetime import datetime
#from os import remove
from utiles import crear_modeloEmbeddings
import pandas as pd
import joblib

app = Flask(__name__)
cors = CORS(app, resources={r"/predecir": {"origins": "*"}})

@app.route("/")
def index():
	return 'La página está funcionando bien'

@app.route("/predecir", methods=["POST"])
def predecir():
    #try:
        #datos en formato json
        json = request.get_json(force=True)

        #obtener datos svm
        nuevos_datos = {
            "Altura_planta" : [json['Altura_planta']],
            "Numero_ramas" : [json['Numero_ramas']],
            "Nivel_produccion" : [json['Nivel_produccion']],
            "Plagas" : [json['Plagas']],
            "Nivel_plagas" : [json['Nivel_plagas']],
            "Nivel_otras_enfermedades" : [json['Nivel_otras_enfermedades']],
            "Produccion_gramos" : [json['Produccion_gramos']]
            }
        data_fenotipica = pd.DataFrame.from_dict(nuevos_datos)
        #data_fenotipica = pd.concat([data_fenotipica], ignore_index=True)

        #obtener datos rnn
        imgB64 = json["base64img"]
        image_64_decode = base64.b64decode(imgB64)
        img = Image.open(io.BytesIO(image_64_decode))
        date = datetime.now()
        f1_str = date.strftime('%d%m%Y%H%M%S')
        cadena = f1_str+".jpg"
        #lugar = "imagenes/" + cadena
        lugar = cadena
        img.save(lugar)

        #predecir svm
        model_entrenado = joblib.load('modelo_entrenado.pkl') # Carga del modelo.
        prediccion_modelo = model_entrenado.predict(data_fenotipica)
        resultado_svm = prediccion_modelo[0]

        #predecir rnn
        longitud, altura = 250, 250
        pesos_modelo = 'pesos4Capas.h5'
        cnn = crear_modeloEmbeddings()
        cnn.load_weights(pesos_modelo)
        x = load_img(lugar, target_size=(longitud, altura))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array = cnn.predict(x)
        result = array[0]
        answer = np.argmax(result)
        if answer == 0:
            estado = "SANA"
        elif answer == 1:
            estado = "ENFERMA"

        cd = {"Code":"200", "Descripción":estado, "Nivel":resultado_svm, "Nombre":cadena}
        return cd
    #except:
    #    cd = {"Code":"503","Descripción":"Algo en el servidor no funciona"
    #    return cd

#if __name__ == '__main__':
#   app.run()