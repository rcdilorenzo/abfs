from toolz import pipe
from keras.models import model_from_json
import numpy as np

class KerasModel():
    def __init__(self, model_path, weights_path):
        with open(model_path) as f:
            self.model_raw_json = f.read()

        self._model = model_from_json(self.model_raw_json)
        self._model.load_weights(weights_path, by_name=False)

    @property
    def model(self):
        return self._model

    def predict(self, scaled_pixel_array):
        post_process = lambda batch: np.squeeze(batch[0])

        return pipe(
            np.array([scaled_pixel_array]),
            self.model.predict,
            post_process
        )

