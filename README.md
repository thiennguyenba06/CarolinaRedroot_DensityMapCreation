# CarolinaRedroot_DensityMapCreation

# Initial set up
Create a python virtual environment\
```bash
python3 virutalenv <your_env_name>
```

Activate the environment\
```bash
source <your_env_name>/bin/activate
```

Download libraries and dependencies\
```bash
pip install -r requirements.txt
```

# Code execution pipeline

## Extract frames from video and do inference
Run `extract.py`
### path configurations for extract.py
```bash
parent_dir = "./" # set to this directory, can be changed if needed
data_dir = "YOUR VIDEO FOLDER DIRECTORY"
```
Run `split_predict.py` to do inference on extracted images\

# Create density map
Run `densitymap.py` to create density map
### setting up paths and constants
```bash
ORIGIN_PATH = "Absolute Path to the first frame as we are choosing the first frame to be our coordinate system origin"
IMG_DIR = "Relative Path to directory of extracted images"
LABEL_DIR = "Relative Path to directory of the results of inference stage"
CORNER_GPS_PATH = "Absolute path to .txt file containing the 4 corners of the bog"
THRESHOLD = <density threshold for choosing spraying location>
```

### shift vector calibration
```bash
actual_corners_coors = [
    [Corner1 GPS in decimal], 
    [Corner2 GPS in decimal], 
    [Corner3 GPS in decimal], 
    [Corner4 GPS in decimal]
    ] # obtain this from google earth
```

