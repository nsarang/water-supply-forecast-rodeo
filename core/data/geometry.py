import geopandas as gpd
import numpy as np
from shapely.geometry import box


def bbox_filter(gdf, bbox):
    """
    Similar to:  `gdf10.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]`
    `.cx` selects rows that intersect the bounding box but leaves the geometries intact, whereas
    the `gpd.overlay` solution will only return the parts of the geometries in the bounding box.
    """
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)])
    return gpd.overlay(gdf, bbox_gdf, how="intersection")


def center2bbox(cx, cy, cr):
    return [cx - cr, cy - cr, cx + cr, cy + cr]


def haversine_distance(row, suffix1="_site", suffix2="_station", radius=6367.3):
    return haversine_distance_coords(
        row["latitude" + suffix1],
        row["longitude" + suffix1],
        row["latitude" + suffix2],
        row["longitude" + suffix2],
        radius=radius,
        is_degree=True,
    )


def haversine_distance_coords(*args, radius=6367.3, is_degree=True):
    assert len(args) in [2, 4]

    if is_degree:
        args = list(map(np.deg2rad, args))

    if len(args) == 2:
        (lat1, lon1), (lat2, lon2) = args
    else:
        (lat1, lon1, lat2, lon2) = args

    archaversine = 2 * (
        np.arcsin(
            np.sqrt(
                np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            )
        )
    )
    distance = archaversine * radius
    return distance


def meters_to_pixels(meter_coords, center_coords, coverage_area, image_size):
    """
    Convert meter coordinates to pixel coordinates.

    Args:
    - meter_coords (np.array): Array of meter coordinates.
    - center_coords (tuple): Center coordinates in meters.
    - coverage_area (float): Coverage area in meters.
    - image_size (tuple): Size of the image (height, width) in pixels.

    Returns:
    - Array of pixel coordinates.
    """
    # Calculate the scale factor (pixels per meter)
    scale_factor = np.array(image_size) / coverage_area

    # Translate meter coordinates based on the center and convert to pixel coordinates
    translated_coords = meter_coords - np.array(center_coords)
    pixel_coords = translated_coords * scale_factor + np.array(image_size) / 2

    return pixel_coords


def add_subpixels_vectorized(image, coords, colors):
    """
    Add multiple pixels of given colors at non-integer coordinates to an image.

    Args:
    - image (np.array): The image array.
    - coords (np.array): Array of non-integer coordinates of the pixels to add.
    - colors (np.array): Array of colors of the pixels to add.
    """
    # Separate X and Y coordinates
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Integer coordinates and their next values
    x_lows, y_lows = np.floor(x_coords).astype(int), np.floor(y_coords).astype(int)
    x_highs, y_highs = np.ceil(x_coords).astype(int), np.ceil(y_coords).astype(int)

    # Calculate proximity weights
    weights_x_low = (x_highs - x_coords).reshape(-1, 1)
    weights_x_high = (x_coords - x_lows).reshape(-1, 1)
    weights_y_low = (y_highs - y_coords).reshape(-1, 1)
    weights_y_high = (y_coords - y_lows).reshape(-1, 1)

    # Calculate combined weights for each neighbor
    weights = np.array(
        [
            weights_x_low * weights_y_low,
            weights_x_high * weights_y_low,
            weights_x_low * weights_y_high,
            weights_x_high * weights_y_high,
        ]
    ).transpose(
        2, 0, 1
    )  # Shape (num_coords, 4, 1)

    # Calculate the indices for each neighboring pixel
    indices = np.array(
        [[y_lows, x_lows], [y_lows, x_highs], [y_highs, x_lows], [y_highs, x_highs]]
    ).transpose(
        2, 0, 1
    )  # Shape (num_coords, 4, 2)

    # Clip indices to ensure they are within image boundaries
    y_indices = np.clip(indices[..., 0], 0, image.shape[0] - 1)
    x_indices = np.clip(indices[..., 1], 0, image.shape[1] - 1)

    # Calculate flat indices for vectorized assignment
    flat_indices = (y_indices * image.shape[1] + x_indices).ravel()

    # Reshape and duplicate color and weight values for vectorized assignment
    replicated_colors = np.repeat(colors[:, np.newaxis, :], 4, axis=1).reshape(-1, 3)
    replicated_weights = weights.reshape(-1, 1)

    # Vectorized addition of weighted colors to the image
    np.add.at(
        image.reshape(-1, 3),
        flat_indices,
        (replicated_colors * replicated_weights).astype(image.dtype),
    )

    # Ensure pixel values are within the valid range
    np.clip(image, 0, 255, out=image)

    return image


def add_circles_vectorized(image, coords, values, radius, anti_aliasing_margin=1.0):
    # Create a grid of coordinates corresponding to the image
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]  # xx_ij = j, yy_ij = i

    # Reshape coords and values for broadcasting
    coords = coords[:, np.newaxis, np.newaxis, :]
    values = values[:, np.newaxis, np.newaxis, :]

    # Calculate squared distances from each pixel to each circle center
    distances = np.sum((np.stack((xx, yy), axis=-1) - coords) ** 2, axis=-1) ** 0.5

    # Calculate masks with anti-aliasing for each circle
    weights = np.clip((radius - distances) / (anti_aliasing_margin + 1e-18), 0, 1)

    # Apply anti-aliasing weights
    weighted_values = values * weights[..., np.newaxis]

    # Average the contributions from all circles, clipped to valid range
    combined_values = np.max(weighted_values, axis=0)

    # Add the combined values to the original image
    image += combined_values.astype(image.dtype)

    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example blank image
    coords = np.array([[30.4, 40.6], [50.5, 60.5]])  # Non-integer coordinates for circle centers
    colors = np.array([[0, 255, 0], [255, 0, 0]])  # Colors for each circle
    radius = 10  # Radius for circles

    # Add circles with fully vectorized function
    image = add_circles_vectorized(image, coords, colors, radius, anti_aliasing_margin=1)
    plt.imshow(255 - image)

    # Example usage
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # Example blank image
    coords = np.array([[10.4, 24.6], [20.5, 30.5]])  # Multiple non-integer coordinates
    colors = np.array([[0, 255, 0], [255, 0, 0]])  # Multiple colors (e.g., green and red)

    # Add subpixels
    image = add_subpixels_vectorized(image, coords, colors)
    plt.imshow(255 - image)
