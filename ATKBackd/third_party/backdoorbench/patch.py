# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2022, CUHK(SZ), SRIBD. All rights reserved.
# From SCLBD/BackdoorBench at f02e3534645f0ee63d6848653062cd6c0d6c400d.
# See LICENSE and NOTICE.md here; supplied as-is without warranties.
# Source: utils/bd_img_transform/patch.py, AddMaskPatchTrigger only.
# Modification: omitted unused imports and optional type annotation. Math unchanged.

class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array,
                 ):
        self.trigger_array = trigger_array

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)
