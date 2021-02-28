from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.models import Model
from threading import Thread
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import numpy as np
from numpy import ndarray
import base64
import flask
from flask import Flask, Response
import redis
from redis import Redis
import uuid
import time
import json
import sys
import io
from typing import Optional, Tuple, Dict, List
from helpers import base64_encode_image
from run_model_server import classify_process
import settings

# initialize our Flask application, Redis server, and Keras model
app: Flask = flask.Flask(__name__)
db: Redis = redis.StrictRedis(host=settings.REDIS_HOST,
                              port=settings.REDIS_PORT, db=settings.REDIS_DB)


def prepare_image(image: PngImageFile, target: Tuple[int]) -> ndarray:
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image: Image.Image = image.convert("RGB")

    # resize the input image and preprocess it
    image: Image.Image = image.resize(target)
    image: ndarray = img_to_array(image)
    image: ndarray = np.expand_dims(image, axis=0)
    image: ndarray = preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=['POST'])
def predict() -> Response:
    # initialize the data dictionary that will be returned from the
    # view
    data: Dict = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image: bytes = flask.request.files["image"].read()
            image: PngImageFile = Image.open(io.BytesIO(image))
            image: ndarray = prepare_image(image, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image: ndarray = image.copy(order="C")
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k: str = str(uuid.uuid4())
            d: Dict = {"id": k, "image": base64_encode_image(image)}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output: Optional[bytes] = db.get(k)
                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output: str = output.decode("utf-8")
                    data["predictions"]: List = json.loads(output)
                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"]: bool = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
