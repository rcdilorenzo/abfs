import numpy as np
from src.group_data_split import DataSplitConfig

def test_loading_data_frame(data):
    assert len(data.df) == 99

def test_sq_ft_column_created(data):
    assert data.df.sq_ft.sum() > 0

def test_grouping_by_image_ids(data):
    assert len(data.image_ids) == 3

def test_image_from_id(data):
    image_id = data.image_ids[0]
    image = data.image_for(image_id)

    assert image.shape == (406, 439, 3)

# ===========
# Image Masks
# ===========

def test_mask_from_image(data):
    image_id = data.image_ids[0]
    image = data.mask_for(image_id)

    assert image.shape == (406, 439)
    assert np.sum(image) == 17118

def test_green_mask(data):
    image_id = data.image_ids[0]
    image = data.green_mask_for(image_id)

    assert image.shape == (406, 439, 3)
    assert (image[:, :, 1] == data.mask_for(image_id)).all()

def test_filtering_then_creating_mask_from_image(data):
    image_id = data.image_ids[0]

    data.data_filter = lambda df: df.sq_ft > 10
    larger_buildings_only = data.mask_for(image_id)

    data.reset_filter()
    all_buildings = data.mask_for(image_id)

    assert np.sum(all_buildings) > np.sum(larger_buildings_only)

# ==============
# Data Filtering
# ==============

def test_filtering_data(data):
    data.data_filter = lambda df: df.sq_ft > 10
    assert len(data.df) < 99

    data.reset_filter()
    assert len(data.df) == 99

# ====================
# Split Train/Val/Test
# ====================

def test_split_data(data):
    assert data.split_data.total == len(data.df)

def test_filtering_then_accessing_split_data(data):
    original_count = data.split_data.total

    data.data_filter = lambda df: df.sq_ft > 10

    assert data.split_data.total < original_count

# =======
# Batches
# =======

def test_train_batch(data):
    data.split_config = DataSplitConfig(0, 0, 0)
    data.batch_size = 2
    data.augment = False

    assert len(data.train_data(0).df.ImageId.unique()) == 2
    assert len(data.train_data(1).df.ImageId.unique()) == 1

def test_train_batch_count_with_partial_last_batch(data):
    data.split_config = DataSplitConfig(0, 0, 0)
    data.batch_size = 2
    data.augment = False

    assert data.train_batch_count() == 2

def test_train_batch_count_with_whole_last_batch(data):
    data.split_config = DataSplitConfig(0, 0, 0)
    data.batch_size = 3
    data.augment = False

    assert data.train_batch_count() == 1

def test_train_batch_count_with_augmentation(data):
    data.split_config = DataSplitConfig(0, 0, 0)
    data.batch_size = 2
    data.augment = True

    assert data.train_batch_count() == 2

    # Number of images has been halved since each with have an associated
    # augmented counterpart
    assert len(data.train_data(0).df.ImageId.unique()) == 1

def test_val_batch(data):
    data.split_config = DataSplitConfig(0.99, 0, 0)
    data.batch_size = 2
    data.augment = True # No-op
    data._split_data = None

    assert len(data.val_data(0).df.ImageId.unique()) == 2
    assert len(data.val_data(1).df.ImageId.unique()) == 1

def test_val_batch_count(data):
    data.split_config = DataSplitConfig(0.99, 0, 0)
    data.batch_size = 1
    data.augment = True # No-op

    assert data.val_batch_count() == 3

def test_test_batch(data):
    data.split_config = DataSplitConfig(0, 0.6, 0)
    data.batch_size = 1

    assert len(data.test_data(0).df.ImageId.unique()) == 1
    assert len(data.test_data(1).df.ImageId.unique()) == 1

def test_test_batch_count(data):
    data.split_config = DataSplitConfig(0, 0.4, 0)
    data.batch_size = 1
    data.augment = True # No-op

    assert data.test_batch_count() == 2

# ============================
# Neural Network Input/Output
# ============================

def test_neural_network_input_output(data):
    IMAGE_COUNT = 3
    data.augment = False

    inputs, outputs = data.to_nn((500, 500))

    assert inputs.shape == (IMAGE_COUNT, 500, 500, 3)
    assert outputs.shape == (IMAGE_COUNT, 500, 500)

def test_neural_network_input_output_with_augmentation(data):
    data.split_config = DataSplitConfig(0, 0, 0)
    data.batch_size = 2
    data.augment = True

    inputs, outputs = data.train_batch_data(0).to_nn((500, 500))

    assert inputs.shape == (2, 500, 500, 3)
    assert outputs.shape == (2, 500, 500)
