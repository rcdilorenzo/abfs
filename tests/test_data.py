import numpy as np

def test_loading_data_frame(data):
    assert len(data.df) == 50

def test_sq_ft_column_created(data):
    assert data.df.sq_ft.sum() > 0

def test_grouping_by_image_ids(data):
    assert len(data.image_ids) == 1

def test_image_from_id(data):
    image_id = data.image_ids[0]
    image = data.image_for(image_id)

    assert image.shape == (406, 439, 3)

def test_mask_from_image(data):
    image_id = data.image_ids[0]
    image = data.mask_for(image_id)

    assert image.shape == (406, 439)
    assert np.sum(image) == 16692

def test_filtering_data(data):
    data.data_filter = lambda df: df.sq_ft > 10
    assert len(data.df) < 50

    data.reset_filter()
    assert len(data.df) == 50

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

