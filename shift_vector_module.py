import numpy as np, os, pyexiv2
import pixel_to_gps



def calculate_shift_vector(PARENT_DIR, corner_folder_dir):     
    corners_file_list = sorted(os.listdir(os.path.join(PARENT_DIR, corner_folder_dir)))
    corners_file_list = [f for f in corners_file_list if f.lower().endswith(('jpg', '.jpeg', '.png'))]
    origin_path = os.path.join(PARENT_DIR, corner_folder_dir, corners_file_list[0])
    # actual gps of corners
    corner_dict = {} # corner_1: (lat, lon), corner_2: (lat, lon), ... corner_4: (lat, lon)
    name = "corner"
    for i, img in enumerate(corners_file_list):
        name = f"corner{i+1}"
        img_path = os.path.join(PARENT_DIR, corner_folder_dir, img)
        xmp = pyexiv2.Image(img_path).read_xmp()
        # print(f"Pitch degree for {name}: {xmp['Xmp.drone-dji.GimbalPitchDegree']}")
        corner_dict[name] = (float(xmp['Xmp.drone-dji.GpsLatitude']), float(xmp['Xmp.drone-dji.GpsLongitude']))

    delta_gps_vector = []
    for i, img in enumerate(sorted(corners_file_list)):
        name = f"corner{i+1}"
        img_path = os.path.join(PARENT_DIR, corner_folder_dir, img)
        exif = pyexiv2.Image(img_path).read_exif()
        center_cor = [float(exif['Exif.Photo.PixelXDimension'])/2, float(exif['Exif.Photo.PixelYDimension'])/2 - 1450]
        gps = pixel_to_gps.get_gps(origin_path, img_path, center_cor)
        delta_gps = [gps[0] - corner_dict[name][0], gps[1] - corner_dict[name][1]]
        delta_gps_vector.append(delta_gps)

    return np.mean(np.array(delta_gps_vector), axis=0) * -1



