from keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, params_f, shape):
        self.params_f = params_f
        self.shape = shape
        self.on_epoch_end()

    def __len__(self):
        return self.len_f()

    def __getitem__(self, batch_id):
        return self.data_for_batch_id(batch_id).to_nn(self.shape)

    def on_epoch_end(self):
        len_f, data_f = self.params_f()
        self.len_f = len_f
        self.data_f = data_f

    def data_for_batch_id(self, batch_id):
        return self.data_f(batch_id)
