import georef2, shift_vector_module
import numpy as np, pyexiv2, os, csv
import matplotlib.pyplot as plt
from shapely import Polygon, box, Point, STRtree, distance
from concurrent.futures import ProcessPoolExecutor

# setting up paths
ORIGIN_PATH = "/Users/thiennguyenba/Documents/School/Research/Density_Video/geotagged_frames/frame_1.jpg" # to be changed depending on dataset
IMG_DIR = "geotagged_frames"
LABEL_DIR = "output"
CORNER_GPS_PATH = "/Users/thiennguyenba/Documents/School/Research/Density_Video/bog_calibration/bog9_calibrations.txt"

actual_corners_coors = [
    [39.741628,-74.526122], 
    [39.742633,-74.525784], 
    [39.742924,-74.527282], 
    [39.741753,-74.527658]
    ] # obtain this from google earth
actual_corners_coors = np.array(actual_corners_coors)
SHIFT_VECTOR = shift_vector_module.calculate_shift_vector(
    corner_coors_path=CORNER_GPS_PATH, 
    actual_corners_vector=actual_corners_coors)

THRESHOLD = 20 # subject to change
CSV_OUTPUT = "density_by_gps.csv"
SPRAY_COORS_CSV = "spray_location.csv"




# setting up constants
SIDE_LENGTH_METERS = 1 # grid square side length in meters
R_EARTH = 6378137.0  # Earth radius
origin_gps = (float(pyexiv2.Image(ORIGIN_PATH).read_xmp()['Xmp.drone-dji.GpsLatitude']), float(pyexiv2.Image(ORIGIN_PATH).read_xmp()['Xmp.drone-dji.GpsLongitude']))


img_list = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])
label_list = sorted([f for f in os.listdir(LABEL_DIR) if f.lower().endswith(".txt")])


# mapping detections to relative coordinate with drone's first image as basis
# y direction is drone's forward direction
# x direction is always orthogonal to y direction to the right
detections = {} # image_id -> list of (x,y) detections in relative coordinate system
img_fname_map = {} # image_id -> image file name
image_bounds = {}
lower_half_image_bounds = {}
all_detections_coor = [] # list of all (x,y) detections in relative coordinate system for all images, used to determine grid size and bounds

gps_map = {} # (lat, lon) -> (density, image_fname) mapping for each grid cell center

# helper methods
def process_img(img, label):
    img_path = os.path.join(IMG_DIR, img)
    img_id = int(img.split("_")[1].split(".")[0])
    label_path = os.path.join(LABEL_DIR, label)
    mapped_list = georef2.georef(ORIGIN_PATH, img_path, label_path)
    vertices = georef2.get_image_corners(ORIGIN_PATH, img_path)
    polygon = Polygon(vertices)
    lower_TL = [v*0.5 for v in vertices[0]]
    lower_TR = [v*0.5 for v in vertices[1]] 
    half_poly = Polygon([lower_TL, lower_TR, vertices[2], vertices[3]])


    return {
        "img": img,
        "img_id": img_id,
        "img_path": img_path,
        "mapped_list": mapped_list,
        "polygon": polygon,
        "lower_half": half_poly
    }

def meters_to_gps(lat_origin, lon_origin, dx, dy, yaw_angle):
    """
    dx: offset to the drone's right (meters)
    dy: offset to the drone's forward (meters)
    yaw_angle: mathematical angle (radians, 0=East, CCW)
    """ 
    # Standard rotation to align Body Frame (x=right, y=forward) 
    # with Inertial Frame (East, North)
    # Using the yaw_angle where 90 deg is North
    east_meter  = dx * np.sin(yaw_angle) + dy * np.cos(yaw_angle)
    north_meter = -dx * np.cos(yaw_angle) + dy * np.sin(yaw_angle)

    # Calculate change in GPS coordinates
    d_lat = (north_meter / R_EARTH) * (180 / np.pi)
    d_lon = (east_meter / (R_EARTH * np.cos(np.radians(lat_origin)))) * (180 / np.pi)
    
    return (lat_origin + d_lat, lon_origin + d_lon)

def find_displacement(drone_gps, point_gps, yaw):
    """
    Calculate the displacement in meters from the drone to a point given their GPS coordinates
    drone_gps: (lat, lon) of the drone
    point_gps: (lat, lon) of the point
    return: (x_dist, y_dist) where x is right and y is forward relative to drone's orientation
    """
    drone_lat, drone_lon = drone_gps
    point_lat, point_lon = point_gps
    # Calculate delta gps
    d_lat = point_lat - drone_lat
    d_lon = point_lon - drone_lon

    # convert to displacement in meters
    east_meter = (d_lon / (180 / np.pi)) * R_EARTH * np.cos(np.radians(drone_lat))
    north_meter = (d_lat / (180 / np.pi)) * R_EARTH

    # solve dx, dy by inversed of transformation
    dx = east_meter * np.sin(yaw) - north_meter * np.cos(yaw)
    dy = east_meter * np.cos(yaw) + north_meter * np.sin(yaw)

    return (dx, dy)

def get_lower_half_centroid(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    lower_half = box(minx, miny, maxx, (miny + maxy) / 2)
    lower_half_polygon = polygon.intersection(lower_half)
    if lower_half_polygon.is_empty:
        return None
    return lower_half_polygon.centroid



# main
if __name__ == "__main__":
    # multithreading for image processing
    print("Processing annotated images...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_img, img_list, label_list))

    for result in results:
        all_detections_coor.extend(result["mapped_list"])
        detections[result["img_id"]] = result["mapped_list"]
        image_bounds[result["img_id"]] = result["polygon"]
        lower_half_image_bounds[result["polygon"]] = result["lower_half"]
        img_fname_map[result["img_id"]] = result["img"]
        
    all_detections_coor = np.array(all_detections_coor) # convert list to numpy array for easier processing later
    detections = {img_id: np.array(weed) for img_id, weed in detections.items()} # convert lists to numpy arrays for easier processing later

    id_list = np.sort(np.array(list(image_bounds.keys())))
    img_bounds_ordered = np.array([image_bounds[img_id] for img_id in id_list])
    tree = STRtree(img_bounds_ordered)
    if tree is not None:
        print("Successfully built spatial index for image bounds.")
    else: 
        print("Failed to build spatial index for image bounds.")
        raise Exception("Spatial index construction failed.")
    

    print("Finished processing images and mapping detections to relative coordinates with origin of drone's first image. \n")

    # create grids
    x_min, x_max = np.min(all_detections_coor[:,0]), np.max(all_detections_coor[:,0])
    y_min, y_max = np.min(all_detections_coor[:,1]), np.max(all_detections_coor[:,1])

    # number of cells in x and y direction
    # y is the drone's forward direction, x is the right direction orthogonal to y
    num_x_cells = int(np.ceil((x_max - x_min) / SIDE_LENGTH_METERS))
    print(f"Number of cells in x direction: {num_x_cells}")
    num_y_cells = int(np.ceil((y_max - y_min) / SIDE_LENGTH_METERS))
    print(f"Number of cells in y direction: {num_y_cells} \n")

    # create cells
    density_grid = np.zeros((num_y_cells, num_x_cells), dtype=int) # a matrix of dimension num_y_cells x num_x_cells initialized to 0 

    # create grid lines
    y_lines = np.linspace(start=y_min, stop=y_max, num=num_y_cells+1)
    x_lines = np.linspace(start=x_min, stop=x_max, num=num_x_cells+1)


    print("Density calculation started...")
    # density calculation
    for y_idx in range(num_y_cells):
        for x_idx in range(num_x_cells):
            up, down = y_lines[y_idx+1], y_lines[y_idx]
            left, right = x_lines[x_idx], x_lines[x_idx+1]
            cell_bounds = box(left, down, right, up)
            cell_center_x, cell_center_y = (left + right) / 2, (down + up) / 2
            cell_centroid = np.array([cell_center_x, cell_center_y])
 
            possible_bounds = tree.query(cell_bounds) # 1d array of possible intersecting polygons (in any region)

            chosen_img = None
            if len(possible_bounds) != 0:
                # choosing best image
                # distances =  []
                # for idx in possible_bounds:
                #     img_id = id_list[idx]
                #     polygon = image_bounds[img_id]
                #     lower_half = lower_half_image_bounds[polygon]
                #     poly_centroid = np.array([lower_half.centroid.x, lower_half.centroid.y])
                #     dist = np.linalg.norm(cell_centroid - poly_centroid)
                #     contestant = [img_id, dist]
                #     distances.append(contestant)

                # distances.sort(key=lambda x:x[1])

                # for (id, dist) in distances:
                #     lower_half = lower_half_image_bounds[image_bounds[id]]
                #     if lower_half.contains(cell_bounds):
                #         chosen_img = id
                #         break
                # if chosen_img is None:
                #     for (id, dist) in distances:
                #         polygon = image_bounds[id]
                #         if polygon.contains(cell_bounds):
                #             chosen_img = id
                #             break

                # choosing best image based on iou/dist
                max_score = 0
                for idx in possible_bounds:
                    img_id = id_list[idx]
                    polygon = image_bounds[img_id]
                    intersection = polygon.intersection(cell_bounds).area
                    union = polygon.area + cell_bounds.area - intersection
                    iou = intersection / union
                    # lower_half = lower_half_image_bounds[polygon]
                    # poly_centroid = np.array([lower_half.centroid.x, lower_half.centroid.y])
                    # dist = np.linalg.norm(cell_centroid - poly_centroid)
                    score = iou

                    if score > max_score:
                        chosen_img = img_id
                        max_score = score
                    
                if chosen_img is None:
                    continue

                points = detections[chosen_img]
                # print(f"{img_id}: {np.ndim(points)}")
                if points.ndim != 2:
                    print(f"Warning: No detections found for image {img_id}. Skipping this cell.")
                    continue
                is_in_x = (points[:,0] >= left) & (points[:,0] <= right)
                is_in_y = (points[:,1] >= down) & (points[:,1] <= up)
                count = np.count_nonzero(is_in_x & is_in_y)
                density = count / SIDE_LENGTH_METERS**2  # density per square meter
                density_grid[y_idx, x_idx] = density

                # map cell center to GPS 
                xmp_data = pyexiv2.Image(os.path.join(IMG_DIR, img_fname_map[chosen_img])).read_xmp()
                yaw = np.radians(90 - float(xmp_data.get('Xmp.drone-dji.FlightYawDegree')))
                gps = meters_to_gps(origin_gps[0], origin_gps[1], cell_center_x, cell_center_y, yaw)
                drone_gps = (float(xmp_data['Xmp.drone-dji.GpsLatitude']), float(xmp_data['Xmp.drone-dji.GpsLongitude']))
                displacement = find_displacement(drone_gps=drone_gps, point_gps=gps, yaw=yaw)
                gps_map[gps] = (density, img_fname_map[chosen_img], displacement)

    print("Finished density map calculation\n")

    # output
    print("Run summary:")
    print("Total images with detections:", len(image_bounds))
    print("Total detection files loaded:", len(detections))
    print("Nonzero grid cells:", np.count_nonzero(density_grid))
    print("Max density in a cell: ", np.max(density_grid))
    print(f"Origin GPS: lat {origin_gps[0]}, lon {origin_gps[1]}")
    print(f"Shift vector: lat {SHIFT_VECTOR[0]}, lon {SHIFT_VECTOR[1]}\n")



    with open(CSV_OUTPUT, "w", newline='') as f1, open(SPRAY_COORS_CSV, "w", newline='') as f2:
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        writer1.writerow(["latitude", "longitude", "density", "image_id", "center_x", "center_y"])
        writer2.writerow(["latitude", "longitude", "density", "image_id"])

        for (lat, lon), (density, image_fname, center_xy) in gps_map.items(): 
            writer1.writerow([lat + SHIFT_VECTOR[0], lon + SHIFT_VECTOR[1], density, image_fname, center_xy[0], center_xy[1]])
            if density > THRESHOLD:
                writer2.writerow([lat + SHIFT_VECTOR[0], lon + SHIFT_VECTOR[1], density, image_fname])

    print(f"Data saved for QGIS in {CSV_OUTPUT}")

    print("Generating visual heat map...")
    plt.figure(figsize=(12, 10))
    img = plt.imshow(
        density_grid, 
        origin='lower', 
        extent=[x_min, x_max, y_min, y_max], 
        cmap='OrRd',
        interpolation='bilinear'
    )
    
    plt.colorbar(img, label='Detections per m²')
    plt.xlabel('Relative X (Meters)')
    plt.ylabel('Relative Y (Meters)')
    plt.title(f'CR Density Heat Map')

    # Save the visual plot
    heatmap_output = "density_heatmap.png"
    plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heat map saved as {heatmap_output}")

    print("Done.")