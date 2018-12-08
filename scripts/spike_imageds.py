import os

import click

import numpy as np

from dtoolcore import DataSet

from imageio import imread, imsave, mimsave

from skimage.filters import threshold_otsu
from skimage.morphology import label

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation


def find_objects(im3d):

    blurred = gaussian_filter(im3d, sigma=[5, 5, 2])
    global_thresh = threshold_otsu(blurred)
    thresholded = 255 * (blurred > global_thresh).astype(np.uint8)
    im3d_to_tiff('thresholded.tif', thresholded)
    connected_components = label(thresholded)

    return connected_components


def get_series_names(dataset):

    def extract_name(idn):
        relpath = dataset.item_properties(idn)['relpath']
        _, name, _ = relpath.split('/')
        return name

    names = set(extract_name(idn) for idn in dataset.identifiers)

    return sorted(list(names))


def im3d_to_tiff(fpath, im3d):

    root, ext = os.path.splitext(fpath)
    assert ext in ['.tif', '.tiff']

    # We use row, col, z, but mimsave expects z, row, col
    transposed = np.transpose(im3d, axes=[2, 0, 1])
    mimsave(fpath, transposed)


def get_stack(dataset, series_name, channel):

    def extract_name(idn):
        relpath = dataset.item_properties(idn)['relpath']
        _, name, _ = relpath.split('/')
        return name

    selected_idns = [
        idn
        for idn in dataset.identifiers
        if extract_name(idn) == series_name
    ]

    coords_overlay = dataset.get_overlay("plane_coords")

    channel_idns = [
        idn
        for idn in selected_idns
        if int(coords_overlay[idn]['C']) == channel
    ]

    z_idn = [(int(coords_overlay[idn]['Z']), idn) for idn in channel_idns]
    z_idn.sort()
    sorted_idns = [idn for x, idn in z_idn]

    images = []
    for idn in sorted_idns:
        im = imread(dataset.item_content_abspath(idn))
        if len(im.shape) == 2:
            images.append(im)
        else:
            images.append(im[:,:,channel])

    stack = np.dstack(images)

    return stack


def get_scale(dataset, series_name):

    def extract_name(idn):
        relpath = dataset.item_properties(idn)['relpath']
        _, name, _ = relpath.split('/')
        return name

    metadata_overlay = dataset.get_overlay('microscope_metadata')
    for idn in dataset.identifiers:
        if extract_name(idn) == series_name:
            metadata = metadata_overlay[idn]
            keys = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ']
            return tuple(float(metadata[k]) for k in keys)


def spike_imageds(dataset):

    series_names = get_series_names(dataset)

    sx, sy, sz = get_scale(dataset, 'Series001')

    cell_stack = get_stack(dataset, 'Series001', 1)
    im3d_to_tiff('cell.tiff', cell_stack)

    find_objects(cell_stack)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    dataset = DataSet.from_uri(dataset_uri)

    spike_imageds(dataset)


if __name__ == '__main__':
    main()
