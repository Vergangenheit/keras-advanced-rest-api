import numpy as np
import base64
import sys

def base64_encode_image(a: bytes) -> str:
    # serialize the input images to be stored in redis
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a: bytes, dtype:Optional[object], shape: Tuple) -> ndarray:
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a: ndarray = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a: ndarray = a.reshape(shape)

    # return the decoded image
    return a

