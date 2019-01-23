import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from collections import namedtuple as Struct

SegAugmentationConfig = Struct('SegAugmentationConfig', ['translate', 'degrees', 'max_scale'])

MOVE_SCALE_ROTATE = SegAugmentationConfig(40, 30, 1.5)

class SegmentationAugmentation():
    def __init__(self, config=MOVE_SCALE_ROTATE, seed=None):
        self.random_seed = seed

        translate = config.translate
        degrees = config.degrees
        scale = config.max_scale
        self.sequence = iaa.Sequential([
            iaa.Affine(translate_px={'x': (-translate, translate),
                                     'y': (-translate, translate)},
                       rotate=(degrees, degrees),
                       scale=(1, scale))
        ])

    def run(self, images, masks):
        if self.random_seed is not None: ia.seed(self.random_seed)

        sequence = self.sequence.to_deterministic()

        return (sequence.augment_images(images),
                sequence.augment_images(masks))

    def run_single(self, image, mask):
        aug_images, aug_masks = self.run(np.array([image]),
                                         np.array([mask]))
        return aug_images[0], aug_masks[0]
