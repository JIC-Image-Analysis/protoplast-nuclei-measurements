import click

import numpy as np

from dtoolcore import DataSet

from imageio import volread, imsave, mimsave

from skimage.filters import threshold_otsu
from skimage.morphology import label

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation


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


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    ds = DataSet.from_uri(dataset_uri)

    identifier = "230d9b0a03c361010c680134e4ac743a4396ced2"
    mixed_identifier = "dd3c76d172e3d366b44051faaaff268990ed16f1"

    # measure_single_item(ds, identifier)

    measure_mixed_channel_image(ds, mixed_identifier)


if __name__ == '__main__':
    main()
