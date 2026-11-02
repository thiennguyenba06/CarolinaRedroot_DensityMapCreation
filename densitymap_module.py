import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyexiv2
from shapely import Polygon, STRtree, box, distance

import georef_module


R_EARTH = 6378137.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".mpo"}


def _read_xmp(image_path):
    image = pyexiv2.Image(str(image_path))
    try:
        return image.read_xmp()
    finally:
        image.close()


def meters_to_gps(lat_origin, lon_origin, dx, dy, yaw_angle):
    east_meter = dx * np.sin(yaw_angle) + dy * np.cos(yaw_angle)
    north_meter = -dx * np.cos(yaw_angle) + dy * np.sin(yaw_angle)

    d_lat = (north_meter / R_EARTH) * (180 / np.pi)
    d_lon = (east_meter / (R_EARTH * np.cos(np.radians(lat_origin)))) * (180 / np.pi)
    return lat_origin + d_lat, lon_origin + d_lon


def find_displacement(drone_gps, point_gps, yaw):
    drone_lat, drone_lon = drone_gps
    point_lat, point_lon = point_gps
    d_lat = point_lat - drone_lat
    d_lon = point_lon - drone_lon

    east_meter = (d_lon / (180 / np.pi)) * R_EARTH * np.cos(np.radians(drone_lat))
    north_meter = (d_lat / (180 / np.pi)) * R_EARTH

    dx = east_meter * np.sin(yaw) - north_meter * np.cos(yaw)
    dy = east_meter * np.cos(yaw) + north_meter * np.sin(yaw)
    return dx, dy


def _tree_indices(tree, geometries, cell):
    matches = tree.query(cell)
    if not len(matches):
        return []
    if isinstance(matches[0], (int, np.integer)):
        return [int(index) for index in matches]

    geometry_indices = {id(geometry): index for index, geometry in enumerate(geometries)}
    return [
        geometry_indices[id(geometry)]
        if id(geometry) in geometry_indices
        else next(
            index
            for index, candidate in enumerate(geometries)
            if candidate.equals(geometry)
        )
        for geometry in matches
    ]


def create_map(
    bog_name,
    root,
    density_thresh,
    geotagged_dir,
    label_dir,
    side_length_meters=1.0,
    origin_path=None,
    output_dir=None,
):
    """Create density CSVs and a heat map from extracted frames and prediction labels."""
    if not np.isfinite(side_length_meters) or side_length_meters <= 0:
        raise ValueError("side_length_meters must be a finite positive number.")

    root = Path(root)
    geotagged_dir = Path(geotagged_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir) if output_dir is not None else root

    image_paths = sorted(
        path
        for path in geotagged_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {geotagged_dir}")

    origin_path = Path(origin_path) if origin_path is not None else image_paths[0]
    if not origin_path.is_file():
        raise FileNotFoundError(f"Origin image does not exist: {origin_path}")

    georeferenced_images = georef_module.georeference_images(
        origin_path=origin_path,
        geotagged_dir=geotagged_dir,
        label_dir=label_dir,
    )

    origin_xmp = _read_xmp(origin_path)
    origin_gps = (
        float(origin_xmp["Xmp.drone-dji.GpsLatitude"]),
        float(origin_xmp["Xmp.drone-dji.GpsLongitude"]),
    )
    yaw = np.radians(90 - float(origin_xmp["Xmp.drone-dji.FlightYawDegree"]))

    detections = {}
    img_fname_map = {}
    image_bounds = {}
    lower_half_image_bounds = {}
    all_detections_coor = []
    image_by_id = {}

    for image in georeferenced_images:
        image_id = int(image.image_path.stem.split("_")[1])
        detections[image_id] = image.detections
        img_fname_map[image_id] = image.image_path.name
        image_bounds[image_id] = image.footprint
        vertices = list(image.footprint.exterior.coords)[:4]
        top_left, top_right, bottom_right, bottom_left = vertices
        new_top_left = [
            top_left[0] + (bottom_left[0] - top_left[0]) * 0.5,
            top_left[1] + (bottom_left[1] - top_left[1]) * 0.5,
        ]
        new_top_right = [
            top_right[0] + (bottom_right[0] - top_right[0]) * 0.5,
            top_right[1] + (bottom_right[1] - top_right[1]) * 0.5,
        ]
        lower_half_image_bounds[image_id] = Polygon(
            [new_top_left, new_top_right, bottom_right, bottom_left]
        )
        image_by_id[image_id] = image
        all_detections_coor.extend(image.detections)

    detections = {
        image_id: np.asarray(points) for image_id, points in detections.items()
    }
    all_detections_coor = np.asarray(all_detections_coor)
    if all_detections_coor.size == 0:
        raise ValueError("No detections were found; cannot determine density-grid bounds.")

    id_list = np.sort(np.asarray(list(image_bounds.keys())))
    img_bounds_ordered = np.asarray([image_bounds[image_id] for image_id in id_list])
    tree = STRtree(img_bounds_ordered)
    if tree is None:
        raise RuntimeError("Spatial index construction failed.")

    print("Finished processing images and mapping detections to relative coordinates.")

    # Keep the original grid bounds based on detection coordinates, not image footprints.
    x_min, x_max = np.min(all_detections_coor[:, 0]), np.max(all_detections_coor[:, 0])
    y_min, y_max = np.min(all_detections_coor[:, 1]), np.max(all_detections_coor[:, 1])
    num_x_cells = int(np.ceil((x_max - x_min) / side_length_meters))
    num_y_cells = int(np.ceil((y_max - y_min) / side_length_meters))
    if num_x_cells == 0 or num_y_cells == 0:
        raise ValueError("Detection coordinates do not span a nonzero density grid.")

    print(f"Number of cells in x direction: {num_x_cells}")
    print(f"Number of cells in y direction: {num_y_cells}")

    density_grid = np.zeros((num_y_cells, num_x_cells), dtype=int)
    y_lines = np.linspace(start=y_min, stop=y_max, num=num_y_cells + 1)
    x_lines = np.linspace(start=x_min, stop=x_max, num=num_x_cells + 1)
    gps_map = {}

    print("Density calculation started...")
    for y_idx in range(num_y_cells):
        for x_idx in range(num_x_cells):
            up, down = y_lines[y_idx + 1], y_lines[y_idx]
            left, right = x_lines[x_idx], x_lines[x_idx + 1]
            cell_bounds = box(left, down, right, up)
            cell_center_x = (left + right) / 2
            cell_center_y = (down + up) / 2

            possible_bounds = _tree_indices(tree, img_bounds_ordered, cell_bounds)
            chosen_img = None
            candidates = []
            if possible_bounds:
                for index in possible_bounds:
                    img_id = id_list[index]
                    polygon = image_bounds[img_id]
                    intersection = polygon.intersection(cell_bounds).area
                    union = polygon.area + cell_bounds.area - intersection
                    iou = intersection / union
                    candidates.append([img_id, iou])

                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                best_candidate = candidates[0]
                chosen_img = best_candidate[0]
                for candidate in candidates[1:]:
                    if candidate[1] / (best_candidate[1] + 0.00001) > 0.8:
                        break
                    best_polygon = image_bounds[chosen_img]
                    best_dist = distance(cell_bounds.centroid, best_polygon.centroid)
                    candidate_polygon = lower_half_image_bounds[candidate[0]]
                    candidate_polygon = image_bounds[candidate[0]]
                    candidate_dist = distance(
                        cell_bounds.centroid, candidate_polygon.centroid
                    )
                    if best_dist > candidate_dist:
                        best_candidate = candidate
                        chosen_img = best_candidate[0]

                if chosen_img is None:
                    continue

                points = detections[chosen_img]
                if points.ndim != 2:
                    density = 0
                else:
                    is_in_x = (points[:, 0] >= left) & (points[:, 0] <= right)
                    is_in_y = (points[:, 1] >= down) & (points[:, 1] <= up)
                    count = np.count_nonzero(is_in_x & is_in_y)
                    density = count / side_length_meters**2
                    density_grid[y_idx, x_idx] = density

                selected = image_by_id[chosen_img]
                gps = meters_to_gps(
                    origin_gps[0], origin_gps[1], cell_center_x, cell_center_y, yaw
                )
                displacement = find_displacement(selected.drone_gps, gps, yaw)
                gps_map[gps] = (
                    density,
                    img_fname_map[chosen_img],
                    displacement,
                )
    output_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / f"{bog_name}_density_by_gps.csv"
    spray_output = output_dir / f"{bog_name}_spray_location.csv"
    heatmap_output = output_dir / f"{bog_name}_density_heatmap.png"

    with csv_output.open("w", newline="") as density_file, spray_output.open(
        "w", newline=""
    ) as spray_file:
        density_writer = csv.writer(density_file)
        spray_writer = csv.writer(spray_file)
        density_writer.writerow(
            ["latitude", "longitude", "density", "image_id", "center_x", "center_y"]
        )
        spray_writer.writerow(["latitude", "longitude", "density", "image_id"])

        for (lat, lon), (density, image_name, center_xy) in gps_map.items():
            density_writer.writerow(
                [lat, lon, density, image_name, center_xy[0], center_xy[1]]
            )
            if density > density_thresh:
                spray_writer.writerow([lat, lon, density, image_name])

    print("Run summary:")
    print("Total images with detections:", len(image_bounds))
    print("Nonzero grid cells:", np.count_nonzero(density_grid))
    print("Max density in a cell:", np.max(density_grid))
    print(f"Origin GPS: lat {origin_gps[0]}, lon {origin_gps[1]}")

    heatmap = plt.figure(figsize=(12, 10))
    image = plt.imshow(
        density_grid,
        origin="lower",
        extent=[x_lines[0], x_lines[-1], y_lines[0], y_lines[-1]],
        cmap="OrRd",
        interpolation="bilinear",
    )
    heatmap.colorbar(image, label="Detections per m²")
    plt.xlabel("Relative X (Meters)")
    plt.ylabel("Relative Y (Meters)")
    plt.title("CR Density Heat Map")
    plt.savefig(heatmap_output, dpi=300, bbox_inches="tight")
    plt.close(heatmap)

    print(f"Density data saved to {csv_output}")
    print(f"Spray locations saved to {spray_output}")
    print(f"Heat map saved to {heatmap_output}")
    return {
        "density_csv": csv_output,
        "spray_csv": spray_output,
        "heatmap": heatmap_output,
    }
