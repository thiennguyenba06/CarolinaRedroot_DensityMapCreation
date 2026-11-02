from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyexiv2
from shapely import Polygon

import georef2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".mpo"}


@dataclass(frozen=True)
class GeoreferencedImage:
    image_path: Path
    detections: np.ndarray
    footprint: Polygon
    drone_gps: tuple[float, float]


def _read_xmp(image_path):
    image = pyexiv2.Image(str(image_path))
    try:
        return image.read_xmp()
    finally:
        image.close()


def georeference_images(origin_path, geotagged_dir, label_dir):
    """Map label detections and image footprints into the origin image's frame."""
    origin_path = Path(origin_path)
    geotagged_dir = Path(geotagged_dir)
    label_dir = Path(label_dir)
    if not origin_path.is_file():
        raise FileNotFoundError(f"Origin image does not exist: {origin_path}")
    if not geotagged_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {geotagged_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(f"Label directory does not exist: {label_dir}")

    image_paths = sorted(
        path
        for path in geotagged_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels_by_stem = {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }
    if not image_paths:
        raise FileNotFoundError(f"No images found in {geotagged_dir}")

    missing_labels = [path.name for path in image_paths if path.stem not in labels_by_stem]
    if missing_labels:
        raise FileNotFoundError(
            f"Missing prediction labels for images: {', '.join(missing_labels)}"
        )

    georeferenced = []
    for image_path in image_paths:
        label_path = labels_by_stem[image_path.stem]
        detections = np.asarray(
            georef2.georef(
                str(origin_path), str(image_path), str(label_path)
            ),
            dtype=float,
        ).reshape(-1, 2)
        footprint = Polygon(
            georef2.get_image_corners(str(origin_path), str(image_path))
        )

        xmp = _read_xmp(image_path)
        drone_gps = (
            float(xmp["Xmp.drone-dji.GpsLatitude"]),
            float(xmp["Xmp.drone-dji.GpsLongitude"]),
        )
        georeferenced.append(
            GeoreferencedImage(
                image_path=image_path,
                detections=detections,
                footprint=footprint,
                drone_gps=drone_gps,
            )
        )

    return georeferenced
