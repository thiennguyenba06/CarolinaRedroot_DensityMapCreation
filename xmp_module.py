import pyexiv2, os, shutil

pyexiv2.registerNs("http://www.dji.com/drone-dji/", "drone-dji")

xmp_keys_map = {
    "LATITUDE": "Xmp.drone-dji.GpsLatitude", 
    "LONGITUDE": "Xmp.drone-dji.GpsLongitude", 
    "REL_ALT": "Xmp.drone-dji.RelativeAltitude",
    "GB_PITCH": "Xmp.drone-dji.GimbalPitchDegree",
    "GB_YAW": "Xmp.drone-dji.FlightYawDegree"
}


def write_xmp_exif(img_path, xmp_data, exif_data):
    """Writes XMP metadata to an image file.
    Params:
        img_path (str): The path to the image file absolutepath.
        xmp_data (dict): A dictionary containing the XMP metadata to be written.
    """
    try:
        img = pyexiv2.Image(img_path)
        metadata = {}
    except:
        raise Exception("Image path is invalid")
    for key, value in xmp_data.items():
        if key in xmp_keys_map.keys():
            xmp_key = xmp_keys_map[key]
            metadata[xmp_key] = str(value) 
    img.modify_xmp(metadata)
    img.modify_exif(exif_data)
    img.close()


