# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2022, CUHK(SZ), SRIBD. All rights reserved.
# From SCLBD/BackdoorBench at f02e3534645f0ee63d6848653062cd6c0d6c400d.
# See LICENSE and NOTICE.md here; supplied as-is without warranties.
# Source: utils/bd_img_transform/blended.py; implementation unchanged.

# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
class blendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image
