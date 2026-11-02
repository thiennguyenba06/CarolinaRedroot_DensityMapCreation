from pathlib import Path
import densitymap_module
import extract_module
import split_predict_module
# Paths set up
BOG_NAME = input("What bog is being processed: ").strip().lower()
proc_opt = input("Process picture by picture? [y/n]").strip().lower()
SEQUENTIAL = True if proc_opt == "y" else False

ROOT_DIR = Path(__file__).resolve().parent
# update these configurations for the corresponding video dataset
VIDEO_DIR = ROOT_DIR / "DJI_202606291517_096_PineIslandbog9newH3m4x0video" 
WEIGHTS_PATH = ROOT_DIR / "best.pt"

FRAMES_INTERVAL = 15
DENSITY_THRESHOLD = 20
DENSITY_CELL_SIDE_LENGTH_METERS = 1

if __name__ == "__main__":
    # extract images from video stream
    geotagged_dir = extract_module.extract(bog_name=BOG_NAME,
                                       parent_dir=ROOT_DIR,
                                       data_dir=VIDEO_DIR,
                                       frames_interval=FRAMES_INTERVAL)

    # do inference on extracted frames
    label_dir = split_predict_module.infer(bog_name=BOG_NAME,
                    sequential=SEQUENTIAL,
                    parent_dir=ROOT_DIR,
                    weights_path=WEIGHTS_PATH,
                    geotagged_dir=geotagged_dir)

    # create density map and geospatial output files
    densitymap_module.create_map(
        bog_name=BOG_NAME,
        root=ROOT_DIR,
        density_thresh=DENSITY_THRESHOLD,
        geotagged_dir=geotagged_dir,
        label_dir=label_dir,
        side_length_meters=DENSITY_CELL_SIDE_LENGTH_METERS,
    )