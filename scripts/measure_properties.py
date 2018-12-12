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


def apply_mask_to_image(mask, image):

    image[np.where(np.logical_not(mask))] = 0

    return image


def get_cell_mask(cell_stack):

    pz = float(cell_stack.metadata.PhysicalSizeZ)
    px = float(cell_stack.metadata.PhysicalSizeX)
    vr = pz / px

    # Find cell
    blurred = gaussian_filter(cell_stack, sigma=[2*vr, 2*vr, 2]).view(Image3D)
    thresholded = (blurred > threshold_otsu(blurred)).view(Image3D)
    connected_components = label(thresholded)

    # We assume (and assert) that we only find one object
    assert len(np.unique(connected_components)) == 2
    mask = np.zeros(cell_stack.shape, dtype=np.uint8)
    mask[np.where(connected_components == 1)] = 255
    binary_dilated_mask = binary_dilation(mask, iterations=10)

    return binary_dilated_mask.view(Image3D)


def remove_nuclei_noise(nuclei_stack):

    return denoise_tv_chambolle(nuclei_stack, weight=0.1).view(Image3D)    


def clipped_subtraction(im1, im2):

    diff = im1.astype(np.int16) - im2.astype(np.int16)

    diff.clip(min=0, max=255)

    return diff.astype(np.uint8)


def dump_properties(cell_objects):

    labels = list(np.unique(cell_objects))
    labels.remove(0)

    for l in labels:
        coords = np.where(cell_objects == l)
        print(len(coords[0]))


def measure_all_items(imageds):

    # image_name = '050218_WT_tapetum_protoplasts_Hoechst_GFP'
    # series_name = 'Series001'
    # # image_name = '240819_amiRILP1_1.5hwrolling_Hoechst'
    # # series_name = 'Series036'

    for image_name in imageds.get_image_names():
        series_names = [sn for sn in imageds.get_series_names(image_name) if 'Series' in sn]
        for series_name in sorted(series_names):
            measure_single_series_nuclei_volume(imageds, image_name, series_name)


def measure_single_series_nuclei_volume(imageds, image_name, series_name):
    cell_stack = imageds.get_stack(image_name, series_name, 1)
    nuclei_stack = imageds.get_stack(image_name, series_name, 0)

    output_dirpath = Path('scratch/working')

    (output_dirpath/image_name/series_name).mkdir(exist_ok=True, parents=True)
    def pathsave(stack, filename):
        full_path = output_dirpath/image_name/series_name/filename

        stack.save(full_path)

    pathsave(cell_stack, 'cell.tif')
    pathsave(nuclei_stack, 'nuclei.tif')

    cell_mask = get_cell_mask(cell_stack)

    pathsave(cell_mask, 'cell_mask.tif')

    masked_nuclei_stack = apply_mask_to_image(cell_mask, nuclei_stack)

    pathsave(masked_nuclei_stack, 'masked_stack.tif')

    denoised = remove_nuclei_noise(masked_nuclei_stack)

    pathsave(denoised, 'denoised.tif')

    planes = []
    for z in range(denoised.shape[2]):
        denoised_z = denoised[:,:,z]
        cell_mask_z = cell_mask[:,:,z]
        median_z = np.median(denoised_z[np.where(cell_mask_z != 0)])
        sub = denoised_z - median_z
        clipped = sub.clip(min=0)
        planes.append(clipped)

    nobg = np.dstack(planes).view(Image3D)

    pathsave(nobg, 'background_removal.tif')

    pathsave(denoised, 'denoised.tif')

    thresholded = nobg > threshold_otsu(nobg)

    pathsave(thresholded, 'thresholded.tif')

    no_small = remove_small_objects(thresholded, min_size=10000).view(Image3D)

    pathsave(no_small, 'no_small.tif')

    px = float(nuclei_stack.metadata.PhysicalSizeX)
    py = float(nuclei_stack.metadata.PhysicalSizeY)
    pz = float(nuclei_stack.metadata.PhysicalSizeX)

    voxel_volume = px * py * pz

    nuclear_volume = np.sum(no_small) * voxel_volume

    print('{},{},{:02f}'.format(image_name, series_name,nuclear_volume))


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
