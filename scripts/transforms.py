from pathlib import Path
from functools import wraps

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, remove_small_objects
from skimage.restoration import denoise_tv_chambolle

from dtoolbioimage import Image3D

output_base = Path('scratch')


def stack_to_path(stack):
    return output_base/stack.metadata.image_name/stack.metadata.series_name


def transformation_3D(transform_func):

    @wraps(transform_func)
    def annotated_transform(*args, **kwargs):

        stack = args[0]
        result = transform_func(*args, **kwargs).view(Image3D)

        output_fname = '{}.tif'.format(transform_func.__name__)
        output_dirpath = stack_to_path(stack)
        output_dirpath.mkdir(parents=True, exist_ok=True)
        result.save(output_dirpath/output_fname)

        result.name = stack.name
        result.metadata = stack.metadata
        
        return result

    return annotated_transform


def largest_label(label_img):
    areas_labels = [(r.area, r.label) for r in regionprops(label_img)]
    return sorted(areas_labels, reverse=True)[0][1]


def iter_plane(stack):

    _, _, zdim = stack.shape

    for z in range(zdim):
        yield(stack[:,:,z])


@transformation_3D
def convex_hull_per_plane(binary_image_stack):
    plane_chulls = map(convex_hull_image, iter_plane(binary_image_stack))
    return np.dstack(plane_chulls)


@transformation_3D
def blur_3D(stack, sigma):
    return gaussian_filter(stack, sigma)


@transformation_3D
def threshold_otsu_3D(stack, mult=1):
    thresh_value = threshold_otsu(stack)
    return stack > (mult * thresh_value)


@transformation_3D
def find_largest_object(stack):
    connected_components = label(stack)
    ll = largest_label(connected_components)
    largest_region = np.where(connected_components == ll, True, False)
    return largest_region


@transformation_3D
def apply_mask_3D(stack, mask):
    return np.multiply(stack, mask.astype(np.uint8))


@transformation_3D
def denoise_tv_3D(stack, weight=0.1):
    return denoise_tv_chambolle(stack, weight=weight)


@transformation_3D
def remove_small_objects_3D(stack, min_size=3000):
    return remove_small_objects(stack, min_size=min_size)


@transformation_3D
def subtract_bg_clipped_masked_median(stack, mask):

    def subtract_masked_med_and_clip(im_mask):
        im, mask = im_mask
        med = np.median(im[np.where(mask)])
        return (im - med).clip(min=0)

    im_mask_gen = zip(iter_plane(stack), iter_plane(mask))

    return np.dstack(map(subtract_masked_med_and_clip, im_mask_gen))
