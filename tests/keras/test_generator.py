import numpy as np
from src.keras.generator import Generator

def test_creating_train_generator_from_data(data):
    generator = data.train_generator(Generator, (200, 200))

    assert len(generator) == data.train_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.train_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs.shape == expected_inputs.shape)
    assert np.all(actual_outputs.shape == expected_outputs.shape)

def test_creating_val_generator_from_data(data):
    generator = data.val_generator(Generator, (200, 200))

    assert len(generator) == data.val_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.val_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs.shape == expected_inputs.shape)
    assert np.all(actual_outputs.shape == expected_outputs.shape)

def test_creating_test_generator_from_data(data):
    generator = data.test_generator(Generator, (200, 200))

    assert len(generator) == data.test_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.test_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs.shape == expected_inputs.shape)
    assert np.all(actual_outputs.shape == expected_outputs.shape)
