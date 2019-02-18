import math

import click

import numpy as np

from dtoolcore import DataSet

from imageio import volread, imsave, mimsave

from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import label, convex_hull_image

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation

from dtoolbioimage import ImageDataSet, Image3D


def largest_label(label_img):
    areas_labels = [(r.area, r.label) for r in regionprops(label_img)]
    return sorted(areas_labels, reverse=True)[0][1]


def iter_plane(stack):

    _, _, zdim = stack.shape

    for z in range(zdim):
        yield(stack[:,:,z])


def transformation_3D(transform_func):

    def annotated_transform(*args, **kwargs):

        stack = args[0]
        result = transform_func(*args, **kwargs).view(Image3D)
        output_fpath = 'scratch/{}_{}.tif'.format(
                stack.name,
                transform_func.__name__
        )
        result.save(output_fpath)
        result.name = stack.name
        return result

    return annotated_transform


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


def get_estimated_volume_from_cell_stack(cell_stack, debug=False):

    pz = float(cell_stack.metadata.PhysicalSizeZ)
    px = float(cell_stack.metadata.PhysicalSizeX)
    vr = pz / px

    # Find cell
    blurred = blur_3D(cell_stack, sigma=[2*vr, 2*vr, 2])
    thresholded = threshold_otsu_3D(blurred, mult=0.5)
    largest_object = find_largest_object(thresholded)
    chulls = convex_hull_per_plane(largest_object)

    largest_disc = np.sum(chulls, axis=(0, 1)).max()

    # Estimate volume from radius
    radius = math.sqrt(largest_disc/math.pi)
    radius_microns = radius * px
    estimated_volume = (4 / 3) * math.pi * math.pow(radius_microns, 3)

    return estimated_volume


def measure_single_series_cell_volume(imageds, image_name, series_name):

    cell_stack = imageds.get_stack(image_name, series_name, 1)
    cell_stack.name = series_name

    cell_volume = get_estimated_volume_from_cell_stack(cell_stack)

    print('{},{},{:02f}'.format(image_name, series_name, cell_volume))


def measure_all_cell_volumes(imageds):

    ins_sns = (
        (image_name, series_name)
        for image_name in imageds.get_image_names()
        for series_name in imageds.get_series_names(image_name)
    )

    selected_ins_sns = filter(lambda x: 'Series' in x[1], ins_sns)

    for image_name, series_name in selected_ins_sns:
        try:
            measure_single_series_cell_volume(imageds, image_name, series_name)
        except KeyError:
            print("ERROR thrown while processing {},{}".format(image_name, series_name))
            continue


@click.command()
@click.option('--dataset-uri', default=None)
def main(dataset_uri):

    if dataset_uri is None:
        with open('data.yml') as fh:
            dataset_uri = fh.readline().strip()

    imageds = ImageDataSet(dataset_uri)

    measure_all_cell_volumes(imageds)


if __name__ == '__main__':
    main()
