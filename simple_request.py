import requests
from typing import Dict

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH1 = "jemma.png"
IMAGE_PATH2 = "spaceshuttle.jpg"

# load the input image and construct the payload for the request
image: bytes = open(IMAGE_PATH2, "rb").read()
payload: Dict = {"image": image}
# submit the request
r: Dict = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
                                      result["probability"]))
# otherwise, the request failed
else:
    print("Request failed")
