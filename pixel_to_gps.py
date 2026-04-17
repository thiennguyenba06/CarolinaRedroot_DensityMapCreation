import georef2
import numpy as np
import pyexiv2

# constants for calculation
SENSOR_FOV_VERTICAL = np.radians(55.072)
SENSOR_FOV_HORIZONTAL = np.radians(69.72)
R_EARTH = 6378137.0  # Earth radius in meters

def meters_to_gps(lat_origin, lon_origin, dx, dy, yaw_angle):
    """
    dx: offset to the drone's right (meters)
    dy: offset to the drone's forward (meters)
    yaw_angle: mathematical angle (radians, 0=East, CCW)
    """
    R_EARTH = 6378137.0  # Earth radius
    
    # Standard rotation to align Body Frame (x=right, y=forward) 
    # with Inertial Frame (East, North)
    # Using the yaw_angle where 90 deg is North
    east_meter  = dx * np.sin(yaw_angle) + dy * np.cos(yaw_angle)
    north_meter = -dx * np.cos(yaw_angle) + dy * np.sin(yaw_angle)

    # Calculate change in GPS coordinates
    d_lat = (north_meter / R_EARTH) * (180 / np.pi)
    d_lon = (east_meter / (R_EARTH * np.cos(np.radians(lat_origin)))) * (180 / np.pi)
    
    return (lat_origin + d_lat, lon_origin + d_lon)



def get_gps(origin_path, img_path, pixel_coor):
    origin_img = pyexiv2.Image(origin_path)
    origin = float(origin_img.read_xmp()['Xmp.drone-dji.GpsLatitude']), float(origin_img.read_xmp()['Xmp.drone-dji.GpsLongitude'])
    current_img = pyexiv2.Image(img_path)
    width, height = float(current_img.read_exif()['Exif.Photo.PixelXDimension']), float(current_img.read_exif()['Exif.Photo.PixelYDimension'])
    current = float(current_img.read_xmp()['Xmp.drone-dji.GpsLatitude']), float(current_img.read_xmp()['Xmp.drone-dji.GpsLongitude'])
    yaw = np.radians(90 - float(current_img.read_xmp()['Xmp.drone-dji.FlightYawDegree']))
    altitude = float(current_img.read_xmp()['Xmp.drone-dji.RelativeAltitude']) + 1
    pitch = np.radians(float(current_img.read_xmp()['Xmp.drone-dji.GimbalPitchDegree']))
    temp_x, temp_y = pixel_coor[0] - width/2, -pixel_coor[1] + height/2
    pixel_coor[0], pixel_coor[1] = temp_x, temp_y

    # Convert pixel coordinates to Cartesian coordinates relative to drone
    pixel_cartesian = georef2.find_point_projection(pixel_coor, width, height, altitude, pitch)

    # get current drone x, y from origin in meters
    drone_from_origin_cartesian = georef2.get_drone_coor(origin[0], origin[1], current[0], current[1], yaw)
    

    # map the pixel_coor to drone with origin as basis, unit meter
    x_meter, y_meter = pixel_cartesian[0] + drone_from_origin_cartesian[0], pixel_cartesian[1] + drone_from_origin_cartesian[1]

    # convert to gps
    gps = meters_to_gps(origin[0], origin[1], x_meter, y_meter, yaw)
    return gps
