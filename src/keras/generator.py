from keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, len_f, data_f, shape):
        self.len_f = len_f
        self.data_f = data_f
        self.shape = shape

    def __len__(self):
        return self.len_f()

    def __getitem__(self, batch_id):
        return self.data_f(batch_id).to_nn(self.shape)
