
from transforms import (
    blur_3D,
    threshold_otsu_3D,
    find_largest_object,
    convex_hull_per_plane,
    apply_mask_3D,
    denoise_tv_3D,
    threshold_otsu_3D,
    remove_small_objects_3D,
    subtract_bg_clipped_masked_median,
    transformation_3D
)


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


def find_nuclei_mask(nuclear_stack, cell_mask):

    masked_nuclear_stack = apply_mask_3D(nuclear_stack, cell_mask)
    denoised = denoise_tv_3D(masked_nuclear_stack, weight=0.1)
    nobg = subtract_bg_clipped_masked_median(denoised, cell_mask)
    thresholded = threshold_otsu_3D(nobg)
    nosmall = remove_small_objects_3D(thresholded)

    return nosmall
