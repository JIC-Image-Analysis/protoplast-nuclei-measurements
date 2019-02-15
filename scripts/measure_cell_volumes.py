import math

import click

import numpy as np

from dtoolcore import DataSet

from imageio import volread, imsave, mimsave

from skimage.filters import threshold_otsu
from skimage.morphology import label

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation

from dtoolbioimage import ImageDataSet, Image3D

def find_objects(im3d):

    blurred = gaussian_filter(im3d, sigma=[5, 5, 2])
    global_thresh = threshold_otsu(blurred)
    thresholded = 255 * (blurred > global_thresh).astype(np.uint8)
    mimsave('thresholded.tif', thresholded)
    connected_components = label(thresholded)

    return connected_components


def dump_properties(cell_objects):

    labels = list(np.unique(cell_objects))
    labels.remove(0)

    for l in labels:
        coords = np.where(cell_objects == l)
        print(len(coords[0]))


def largest_object_label(cell_objects):

    labels = list(np.unique(cell_objects))
    labels.remove(0)

    def object_size(l):
        coords = np.where(cell_objects == l)
        return len(coords[0])

    sizes_labels = [(object_size(l), l) for l in labels]

    sizes_labels.sort(reverse=True)

    return sizes_labels[0][1]


def clipped_subtraction(im1, im2):

    diff = im1.astype(np.int16) - im2.astype(np.int16)

    diff.clip(min=0, max=255)

    return diff.astype(np.uint8)


def measure_mixed_channel_image(dataset, identifier):

    item_fpath = dataset.item_content_abspath(identifier)

    im3d = volread(item_fpath)

    im3d_blue = im3d[:,:,:,2]
    im3d_green = im3d[:,:,:,1]
    im3d_red = im3d[:,:,:,0]

    im3d_cell = clipped_subtraction(im3d_green, im3d_blue)
    im3d_nuclei = clipped_subtraction(im3d_red, im3d_blue)

    binary_dilated_mask = find_mask_from_cell_channel(im3d_cell)

    im3d_nuclei[np.where(np.logical_not(binary_dilated_mask))] = 0

    mimsave('masked_red.tif', im3d_nuclei)


def find_mask_from_cell_channel(im3d_cell):

    cell_objects = find_objects(im3d_cell)
    l_cell = largest_object_label(cell_objects)

    mask = np.zeros(im3d_cell.shape, dtype=np.uint8)
    mask[np.where(cell_objects == l_cell)] = 255
    binary_dilated_mask = binary_dilation(mask, iterations=10)

    return binary_dilated_mask


def measure_single_item(dataset, identifier):

    item_fpath = dataset.item_content_abspath(identifier)

    im3d = volread(item_fpath)

    im3d_cell = im3d[:,:,:,1]
    im3d_nuclei = im3d[:,:,:,0]
    mimsave('red.tif', im3d_nuclei)

    im3d_nuclei[np.where(np.logical_not(binary_dilated_mask))] = 0

    mimsave('masked_red.tif', im3d_nuclei)

    filtered = median_filter(im3d_nuclei, size=[3, 5, 5])
    mimsave('filtered.tif', filtered)

    print(threshold_otsu(filtered))

    for p in range(11):
        plane = filtered[p,:,:]
        maskplane = binary_dilated_mask[p,:,:]
        mo = threshold_otsu(plane[np.where(maskplane)])
        o = threshold_otsu(plane)
        print(o, mo)

    thresholded = filtered > 60

    mimsave('thresholded.tif', 255 * thresholded.astype(np.uint8))


def largest_label(connected_components):

    labels = set(np.unique(connected_components)) - set([0])

    size_by_label = 0


def get_estimated_volume_from_cell_stack(cell_stack):

    pz = float(cell_stack.metadata.PhysicalSizeZ)
    px = float(cell_stack.metadata.PhysicalSizeX)
    vr = pz / px

    # Find cell
    blurred = gaussian_filter(cell_stack, sigma=[2*vr, 2*vr, 2]).view(Image3D)
    # blurred.save('blurred.tif')
    thresholded = (blurred > threshold_otsu(blurred)).view(Image3D)
    # thresholded.save('thresholded.tif')
    connected_components = label(thresholded)
    assert len(np.unique(connected_components)) == 2

    # Find the largest disc in the stack
    cell_label = 1
    def size_of_label_in_plane(z):
        return len(np.where(connected_components[:,:,z] == cell_label)[0])

    _, _, zdim = connected_components.shape
    largest_disc = max(size_of_label_in_plane(z) for z in range(zdim))

    # Estimate volume from radius
    radius = math.sqrt(largest_disc/math.pi)
    radius_microns = radius * px
    estimated_volume = (4 / 3) * math.pi * math.pow(radius_microns, 3)

    return estimated_volume


def measure_single_series_cell_volume(imageds, image_name, series_name):

    cell_stack = imageds.get_stack(image_name, series_name, 1)

    cell_volume = get_estimated_volume_from_cell_stack(cell_stack)

    print('{},{},{:02f}'.format(image_name, series_name, cell_volume))


def measure_all_cell_volumes(imageds):

    for image_name in imageds.get_image_names():
        series_names = [sn for sn in imageds.get_series_names(image_name) if 'Series' in sn]
        for series_name in sorted(series_names):
            measure_single_series_cell_volume(imageds, image_name, series_name)


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
