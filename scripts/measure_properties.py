from pathlib import Path

import click

import numpy as np

from dtoolbioimage import ImageDataSet, Image3D

from imageio import imsave

from skimage.morphology import label, remove_small_objects
from skimage.filters import threshold_otsu
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation
from skimage.restoration import denoise_tv_chambolle


from transforms import (
    blur_3D,
    convex_hull_per_plane,
    threshold_otsu_3D,
    find_largest_object
)


from transforms import apply_mask_3D, denoise_tv_3D, iter_plane, transformation_3D, remove_small_objects_3D


@transformation_3D
def find_cell_mask(cell_stack):

    pz = float(cell_stack.metadata.PhysicalSizeZ)
    px = float(cell_stack.metadata.PhysicalSizeX)
    vr = pz / px

    # Find cell
    blurred = blur_3D(cell_stack, sigma=[2*vr, 2*vr, 2])
    thresholded = threshold_otsu_3D(blurred, mult=0.5)
    largest_object = find_largest_object(thresholded)
    chulls = convex_hull_per_plane(largest_object)

    return chulls


@transformation_3D
def subtract_bg_clipped_masked_median(stack, mask):

    def subtract_masked_med_and_clip(im_mask):
        im, mask = im_mask
        med = np.median(im[np.where(mask)])
        return (im - med).clip(min=0)

    im_mask_gen = zip(iter_plane(stack), iter_plane(mask))

    return np.dstack(map(subtract_masked_med_and_clip, im_mask_gen))


def measure_object_volume(stack):

    px = float(stack.metadata.PhysicalSizeX)
    py = float(stack.metadata.PhysicalSizeY)
    pz = float(stack.metadata.PhysicalSizeZ)

    voxel_volume = px * py * pz

    return np.sum(stack) * voxel_volume


def find_nuclei_mask(nuclear_stack, cell_mask):

    masked_nuclear_stack = apply_mask_3D(nuclear_stack, cell_mask)
    denoised = denoise_tv_3D(masked_nuclear_stack, weight=0.1)
    nobg = subtract_bg_clipped_masked_median(denoised, cell_mask)
    thresholded = threshold_otsu_3D(nobg)
    nosmall = remove_small_objects_3D(thresholded)

    return nosmall


def estimate_cell_volume_from_mask(cell_mask):
    px = float(cell_mask.metadata.PhysicalSizeX)
    largest_disc = np.sum(cell_mask, axis=(0, 1)).max()

    # Estimate volume from radius
    radius = np.sqrt(largest_disc/np.pi)
    radius_microns = radius * px
    estimated_volume = (4 / 3) * np.pi * (radius_microns ** 3)

    return estimated_volume


def measure_cell_and_nuclear_volumes(imageds, image_name, series_name):
    cell_stack = imageds.get_stack(image_name, series_name, 1)
    cell_mask = find_cell_mask(cell_stack)
    estimated_cell_volume_in_microns = estimate_cell_volume_from_mask(cell_mask)

    nuclear_stack = imageds.get_stack(image_name, series_name, 0)
    nuclei_mask = find_nuclei_mask(nuclear_stack, cell_mask)

    nuclear_volume_in_microns = measure_object_volume(nuclei_mask)  

    print("{},{},{},{}".format(image_name, series_name, estimated_cell_volume_in_microns, nuclear_volume_in_microns))


def measure_all_items(imageds):

    ins_sns = (
        (image_name, series_name)
        for image_name in imageds.get_image_names()
        for series_name in imageds.get_series_names(image_name)
    )

    selected_ins_sns = filter(lambda x: 'Series' in x[1], ins_sns)

    i = next(selected_ins_sns)
    i = next(selected_ins_sns)


    print(*i)

    measure_cell_and_nuclear_volumes(imageds, *i)

    # # image_name = '050218_WT_tapetum_protoplasts_Hoechst_GFP'
    # # series_name = 'Series001'
    # # # image_name = '240819_amiRILP1_1.5hwrolling_Hoechst'
    # # # series_name = 'Series036'

    # for image_name in imageds.get_image_names():
    #     series_names = [sn for sn in imageds.get_series_names(image_name) if 'Series' in sn]
    #     for series_name in sorted(series_names):
    #         measure_single_series_nuclei_volume(imageds, image_name, series_name)


@click.command()
@click.option('--dataset-uri', default=None)
def main(dataset_uri):

    if dataset_uri is None:
        with open('data.yml') as fh:
            dataset_uri = fh.readline().strip()

    imageds = ImageDataSet(dataset_uri)

    measure_all_items(imageds)


if __name__ == "__main__":
    main()  # NOQA
