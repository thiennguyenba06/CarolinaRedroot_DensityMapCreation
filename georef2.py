import numpy as np
import os
import pyexiv2

SENSOR_FOV_VERTICAL = np.radians(55.072)
SENSOR_FOV_HORIZONTAL = np.radians(69.72)

def find_center(points, width, height):
    """
    Conpute the center of a detection box given its corner points
    @param points: a tuple of corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    @return: x, y
    """
    x_center = sum(point[0] for point in points) / len(points)
    y_center = sum(point[1] for point in points) / len(points)
    # denormalize from [0,1] to pixel coordinates
    x_center = x_center * width
    y_center = y_center * height
    # change of basis: origin at center of image
    x = x_center - width / 2
    y = -y_center + height / 2
    return x, y

def find_angle_y(y, img_height):
    """
    Compute the angle in radians form by the vector from camera to (0,0) and to (0,y)
    param y: y coordinate in pixels
    param img_width: width of the image in pixels
    return: angle in radians
    """
    SO =  img_height / (2*np.tan(SENSOR_FOV_VERTICAL/2))
    angle_y = np.arctan(y/SO)
    return angle_y

def find_angle_x(x, y, img_width):
    """
    Compute the angle in radians form by the vector from camera to (0, y) and to (x, y) 
    param x: x coordinate in pixels
    param y: y coordinate in pixels
    param img_width: width of the image in pixels
    """
    SO = img_width / (2*np.tan(SENSOR_FOV_HORIZONTAL/2))
    Oy = np.sqrt(np.square(SO) + np.square(y))
    angle_x = np.arctan(x/Oy)
    return angle_x

def find_point_projection(point, img_width, img_height, drone_height, pitch):
    """
    Find the projection of a point in image coordinates to relative Cartesian coordinates to the drone 
    Mapping: pixels coordinates (origin at center of image) -> cartesian coordinates in meters (origin at drone position)
    param points: center of a detection box
    param img_width: width of the image in pixels
    return : x_distance, y_distance in meters relative to drone position
    """ 
    angle_y = find_angle_y(point[1], img_height)
    angle_x = find_angle_x(point[0], point[1], img_width)
    y_distance = drone_height * np.tan(np.pi/2 + pitch + angle_y)
    x_distance = np.sqrt(np.square(drone_height) + np.square(y_distance)) * np.tan(angle_x)
    return x_distance, y_distance

def get_drone_coor(lat1, lon1, lat2, lon2, yaw_angle):
    """
    Get the relative Cartesian coordinates of the drone given its GPS coordinates
    param lat1: latitude of the origin point in degrees
    param lon1: longitude of the origin point in degrees
    param lat2: latitude of the drone point in degrees
    param lon2: longitude of the drone point in degrees
    param yaw_angle: yaw angle of the drone in radians
    return: x, y coordinates in meters relative to origin point
    refer to notes for rotation matrix derivation
    """
    R = 6378137.0
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    x = np.radians(delta_lon) * R * np.cos(np.radians(lat1))
    y = np.radians(delta_lat) * R
    
    drone_x = x*np.sin(yaw_angle) - y*np.cos(yaw_angle)
    drone_y = x*np.cos(yaw_angle) + y*np.sin(yaw_angle)
    return drone_x, drone_y


def get_detections_coor(img_path, detections_path):
    """
    Get the relative Cartesian coordinates of all detections in an image
    param img_path: path to the image
    param detections_path: path to the detection txt file
    return: list of (x,y) coordinates in meters relative to drone position
    """
    img = pyexiv2.Image(img_path)
    yaw = np.radians(90 - float(img.read_xmp()['Xmp.drone-dji.FlightYawDegree']))
    pitch = np.radians(float(img.read_xmp()['Xmp.drone-dji.GimbalPitchDegree']))
    # print("pitch:", pitch)
    altitude = float(img.read_xmp()['Xmp.drone-dji.RelativeAltitude']) + 1
    # print("altitude:", altitude)
    # print("yaw:", yaw) 
    img_width = float(img.read_exif()['Exif.Photo.PixelXDimension'])
    # print("img width: ", img_width)
    img_height = float(img.read_exif()['Exif.Photo.PixelYDimension'])
    # print("img height: ", img_height)
    coor_list = []
    with open(detections_path) as file:
        lines = file.read().splitlines()
    for line in lines:
        if (line.strip() != ""):
            boxlist = list(line.split(" "))[1:]
            box = [(float(boxlist[0]), float(boxlist[1])), (float(boxlist[2]), float(boxlist[3])),
                   (float(boxlist[4]), float(boxlist[5])), (float(boxlist[6]), float(boxlist[7]))]
            coor = find_center(box, img_width, img_height)
            x, y = find_point_projection(coor, img_width, img_height, altitude, pitch)
            coor_list.append([x, y])
    return np.array(coor_list)

def map_to_drone(detections_coor, drone_coor):
    """
    Map the relative Cartesian coordinates of detections to the drone's relative coordinate system
    param detections_coor: list of (x,y) coordinates in meters relative to drone position
    param drone_coor: (x,y) coordinate of the drone in relative coordinate system
    return: list of (x,y) coordinates in meters in relative coordinate system
    """
    mapped_list = []
    for detection in detections_coor:
        x = detection[0] + drone_coor[0]
        y = detection[1] + drone_coor[1]
        mapped_list.append([x, y])
    return mapped_list


def get_image_corners(origin_path, img_path):
    """
    Computes the projected ground coordinates of the four image corners

    The returned coordinates follow the ordering:
    Top-Left → Top-Right → Bottom-Right → Bottom-Left.

    Parameters
    ----------
    origin_path : str
        File path to the origin image whose GPS coordinates form the reference
        point for relative positioning.
    img_path : str
        File path to the image whose corner coordinates will be projected and
        mapped.

    Returns
    -------
    mapped_corners : list of [float, float]
        A list of four (x, y) ground-projected corner coordinates in the order:
        Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        Units are in meters relative to the origin image's position.
    """
    origin = pyexiv2.Image(origin_path)
    img = pyexiv2.Image(img_path)
    corners = [] # stored in (x, y) from TL, TR, BR, BL order
    LAT1 = float(origin.read_xmp()['Xmp.drone-dji.GpsLatitude'])
    LON1 = float(origin.read_xmp()['Xmp.drone-dji.GpsLongitude'])
    lat2 = float(img.read_xmp()['Xmp.drone-dji.GpsLatitude'])
    lon2 = float(img.read_xmp()['Xmp.drone-dji.GpsLongitude'])   
    yaw = np.radians(90 - float(img.read_xmp()['Xmp.drone-dji.FlightYawDegree']))
    altitude = float(img.read_xmp()['Xmp.drone-dji.RelativeAltitude']) + 1  
    pitch = np.radians(float(img.read_xmp()['Xmp.drone-dji.GimbalPitchDegree']))    
    img_width = float(img.read_exif()['Exif.Photo.PixelXDimension'])
    img_height = float(img.read_exif()['Exif.Photo.PixelYDimension'])

    # Top-left
    x, y = find_point_projection((-img_width/2,img_height/2), img_width, img_height, altitude, pitch)
    corners.append([x, y])
    # Top-right
    x, y = find_point_projection((img_width/2,img_height/2), img_width, img_height, altitude, pitch)
    corners.append([x, y])
    # Bottom-right
    x, y = find_point_projection((img_width/2,-img_height/2), img_width, img_height, altitude, pitch)
    corners.append([x, y])
    # Bottom-left
    x, y = find_point_projection((-img_width/2,-img_height/2), img_width, img_height, altitude, pitch)
    corners.append([x, y])
 
    drone_coor = get_drone_coor(LAT1, LON1, lat2, lon2, yaw)
    mapped_corners = map_to_drone(corners, drone_coor)
    return mapped_corners

def georef(origin_path, img_path, label_path):
    detections_coor = get_detections_coor(img_path, label_path)
    LAT1 = float(pyexiv2.Image(origin_path).read_xmp()['Xmp.drone-dji.GpsLatitude'])
    LON1 = float(pyexiv2.Image(origin_path).read_xmp()['Xmp.drone-dji.GpsLongitude'])
    lat2 = float(pyexiv2.Image(img_path).read_xmp()['Xmp.drone-dji.GpsLatitude'])
    lon2 = float(pyexiv2.Image(img_path).read_xmp()['Xmp.drone-dji.GpsLongitude'])
    yaw = np.radians(90 - float(pyexiv2.Image(origin_path).read_xmp()['Xmp.drone-dji.FlightYawDegree']))
    drone_coor = get_drone_coor(LAT1, LON1, lat2, lon2, yaw)
    mapped_coor = map_to_drone(detections_coor, drone_coor)
    pyexiv2.Image(img_path).close()
    pyexiv2.Image(origin_path).close()
    return mapped_coor



