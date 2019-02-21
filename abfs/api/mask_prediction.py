from pyramid.response import FileResponse
from abfs.api.prediction.image import ImagePrediction
import json

class MaskPrediction():
    def __init__(self, request, keras_model):
        self.request = request
        self.keras_model = keras_model
        self.json_body = json.loads(request.body)

    def respond(self):
        prediction = ImagePrediction(self.json_body['image_url'],
                                     self.keras_model,
                                     temp_file=True)

        return FileResponse(prediction.run(), self.request)
