# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAugmentedImage:
    """Loads pre-augmented images.
    
    Args:
        aug_dir (str): Directory path where augmented images are stored
    """
    def __init__(self, aug_dir):
        self.aug_dir = aug_dir
        
    def __call__(self, results):
        """Load augmented images."""
        # Extract class info from original image path
        original_path = results['img_info']['filename']
        class_name = os.path.basename(os.path.dirname(original_path))
        image_name = os.path.basename(original_path)
        
        # Augmented image path
        aug_path = os.path.join(self.aug_dir, class_name, image_name)
        
        if os.path.exists(aug_path):
            # Load augmented image if it exists
            results['img'] = mmcv.imread(aug_path)
        else:
            # Use original image if augmented image does not exist
            pass
            
        return results
        
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(aug_dir={self.aug_dir})'
        return repr_str


@PIPELINES.register_module()
class LoadRotatedImage:
    """Loads only pre-rotated images.
    Args:
        rot_dir (str): Directory path where rotated images are stored
    """
    def __init__(self, rot_dir):
        self.rot_dir = rot_dir

    def __call__(self, results):
        # Extract class info from original image path
        original_path = results['img_info']['filename']
        class_name = os.path.basename(os.path.dirname(original_path))
        image_name = os.path.basename(original_path)

        # Apply rotated image filename pattern
        if '_rot' not in image_name:
            # Convert original filename to rotated filename (e.g., xxx.jpg -> xxx_rot30.jpg)
            raise FileNotFoundError(f"Rotated image filename must contain '_rot': {image_name}")

        rot_path = os.path.join(self.rot_dir, class_name, image_name)

        if os.path.exists(rot_path):
            results['img'] = mmcv.imread(rot_path)
        else:
            raise FileNotFoundError(f"Rotated image does not exist: {rot_path}")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rot_dir={self.rot_dir})'
        return repr_str
