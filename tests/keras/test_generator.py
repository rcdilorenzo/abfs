import numpy as np
from abfs.keras.generator import Generator

def test_creating_train_generator_from_data(data):
    generator = data.train_generator(Generator, (200, 200))

    assert len(generator) == data.train_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.train_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs == expected_inputs)
    assert np.all(actual_outputs == expected_outputs)

def test_train_generator_randomizes_only_on_epoch_end(data):
    generator = data.train_generator(Generator, (200, 200))

    batch_0_data = generator.data_for_batch_id(0)

    assert batch_0_data == generator.data_for_batch_id(0)

    generator.on_epoch_end()

    assert batch_0_data != generator.data_for_batch_id(0)


def test_creating_val_generator_from_data(data):
    generator = data.val_generator(Generator, (200, 200))

    assert len(generator) == data.val_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.val_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs.shape == expected_inputs.shape)
    assert np.all(actual_outputs.shape == expected_outputs.shape)

def test_val_generator_does_not_randomize(data):
    generator = data.val_generator(Generator, (200, 200))

    batch_0_data = generator.data_for_batch_id(0)

    assert generator.data_for_batch_id(0) == data.val_batch_data(0)

    generator.on_epoch_end()

    assert batch_0_data == generator.data_for_batch_id(0)


def test_creating_test_generator_from_data(data):
    generator = data.test_generator(Generator, (200, 200))

    assert len(generator) == data.test_batch_count()

    actual_inputs, actual_outputs = generator.__getitem__(0)
    expected_inputs, expected_outputs = data.test_batch_data(0).to_nn((200, 200))

    assert np.all(actual_inputs.shape == expected_inputs.shape)
    assert np.all(actual_outputs.shape == expected_outputs.shape)

def test_test_generator_does_not_randomize(data):
    generator = data.test_generator(Generator, (200, 200))

    batch_0_data = generator.data_for_batch_id(0)

    assert generator.data_for_batch_id(0) == data.test_batch_data(0)

    generator.on_epoch_end()

    assert batch_0_data == generator.data_for_batch_id(0)
