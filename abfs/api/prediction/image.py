from skimage import io
from toolz import curry, pipe
from lenses import lens
from PIL import Image
import numpy as np
import json

import tempfile

shape_from_model = lens['config']['layers'][0]['config']['batch_input_shape'][1].get()

class ImagePrediction():
    def __init__(self, url, keras_model, temp_file=False):
        self.image = io.imread(url)
        self.temp_file = temp_file
        self.model_json = json.loads(keras_model.model_raw_json)
        self.keras_model = keras_model

    def run(self):
        image_height, image_width = self.image.shape[0:2]

        prediction = pipe(
            self.image,
            self._prepare_image,
            self.keras_model.predict,
            self._resize_image((image_width, image_height))
        )

        if self.temp_file:
            file_path = tempfile.NamedTemporaryFile('w+b', suffix='.png', delete=False)
            io.imsave(file_path.name, prediction)
            return file_path.name
        else:
            return prediction

    def _prepare_image(self, image):
        image_size = shape_from_model(self.model_json)
        return self._resize_image((image_size, image_size), image) / 255

    @curry
    def _resize_image(self, shape, image):
        return np.array(Image.fromarray(image).resize(shape))
