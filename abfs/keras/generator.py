import math
from keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, params_f, shape, max_batches=math.inf):
        self.params_f = params_f
        self.shape = shape
        self.max_batches = max_batches
        self.on_epoch_end()

    def __len__(self):
        return min(self.len_f(), self.max_batches)

    def __getitem__(self, batch_id):
        return (self.data_for_batch_id(batch_id)
                .to_nn(self.shape, scale_pixels=True))

    def on_epoch_end(self):
        len_f, data_f = self.params_f()
        self.len_f = len_f
        self.data_f = data_f

    def data_for_batch_id(self, batch_id):
        return self.data_f(batch_id)
