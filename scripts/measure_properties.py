import click

from dtoolbioimage import ImageDataSet

from find_masks import find_cell_mask, find_nuclei_mask
from utils import measure_object_volume, estimate_cell_volume_from_mask


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

    for i in list(selected_ins_sns)[:5]:
        measure_cell_and_nuclear_volumes(imageds, *i)


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
