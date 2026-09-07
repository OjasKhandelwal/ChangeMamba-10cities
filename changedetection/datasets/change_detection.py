import os

import numpy as np

from . import imutils
from .common import BaseChangeDataset


class ChangeDetectionDataset(BaseChangeDataset):
    def transform(self, pre_img, post_img, label):

    # No online augmentation.
    # Augmentation has already been performed offline.

      pre_img = imutils.to_channel_first(
        imutils.normalize_img(pre_img)
    )

      post_img = imutils.to_channel_first(
        imutils.normalize_img(post_img)
    )

      return pre_img, post_img, np.asarray(label)

    def __getitem__(self, index):
        item_name = self.data_list[index]

        pre_path = os.path.join(
            self.dataset_path, "T1", item_name
        )
        post_path = os.path.join(
            self.dataset_path, "T2", item_name
        )
        label_path = os.path.join(
            self.dataset_path, "GT", item_name
        )

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)

        label = self.loader(label_path)

        # GT images are RGBA; use only the first channel
        if label.ndim == 3:
            label = label[:, :, 0]

        # Convert 0/255 mask to 0/1 float32
        label = label.astype(np.float32) / 255.0

        pre_img, post_img, label = self._transform(
            pre_img, post_img, label
        )

        return pre_img, post_img, label, item_name