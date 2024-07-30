import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )# total_patches:1568 total_masks:1408
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class RandomMaskingGenerator:
    # input_size：传入的为window_size=input_size//patch_size，即224/16=14
    # mask_ratio：mask的比例，默认为0.75
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width  # patch的总数即196
        self.num_mask = int(mask_ratio * self.num_patches)  # 196 * 0.75

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([  # 水平方向叠起来
            np.zeros(self.num_patches - self.num_mask),  # 25%为0
            np.ones(self.num_mask),  # mask的部分设为1
        ])
        np.random.shuffle(mask)
        return mask  # [196]
# mask = RandomMaskingGenerator(14, 0.75)
# print(mask())
