import numpy as np

def measure_object_volume(stack):

    px = float(stack.metadata.PhysicalSizeX)
    py = float(stack.metadata.PhysicalSizeY)
    pz = float(stack.metadata.PhysicalSizeZ)

    voxel_volume = px * py * pz

    return np.sum(stack) * voxel_volume


def estimate_cell_volume_from_mask(cell_mask):
    px = float(cell_mask.metadata.PhysicalSizeX)
    largest_disc = np.sum(cell_mask, axis=(0, 1)).max()

    # Estimate volume from radius
    radius = np.sqrt(largest_disc/np.pi)
    radius_microns = radius * px
    estimated_volume = (4 / 3) * np.pi * (radius_microns ** 3)

    return estimated_volume
