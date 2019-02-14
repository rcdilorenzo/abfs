from pyramid.response import Response
from skimage import io
from toolz import curry, pipe
from PIL import Image
from lenses import lens
import numpy as np
import json

import tempfile
from pyramid.response import FileResponse

shape_from_model = lens['config']['layers'][0]['config']['batch_input_shape'][1].get()

class MaskPrediction():
    def __init__(self, request, keras_model):
        self.request = request
        self.json_body = json.loads(request.body)
        self.model_json = json.loads(keras_model.model_raw_json)
        self.keras_model = keras_model

    def respond(self):
        original_image = io.imread(self.json_body['image_url'])
        image_height, image_width = original_image.shape[0:2]

        prediction = pipe(
            original_image,
            self._prepare_image,
            self.keras_model.predict,
            self._resize_image((image_width, image_height))
        )

        file_path = tempfile.NamedTemporaryFile('w+b', suffix='.png', delete=True)
        io.imsave(file_path.name, prediction)

        return FileResponse(file_path.name, self.request)

    def _prepare_image(self, image):
        image_size = shape_from_model(self.model_json)
        return self._resize_image((image_size, image_size), image) / 255

    @curry
    def _resize_image(self, shape, image):
        return np.array(Image.fromarray(image).resize(shape))




