from scipy.misc import imsave
from src.segmentation_augmentation import SegmentationAugmentation, SegAugmentationConfig
import numpy as np

def test_move_rotate_augmentation(sample_image, box_mask):
    config = SegAugmentationConfig(200, 30, 1)
    aug = SegmentationAugmentation(config, seed=0)
    mask = box_mask(sample_image.shape[0:2])

    aug_image, aug_mask = aug.run_single(sample_image, mask)

    # Mask gets cropped off with a seed of 0
    assert np.sum(mask) > np.sum(aug_mask)

    # Image is only partially visible
    assert np.sum(sample_image) > np.sum(aug_image)





