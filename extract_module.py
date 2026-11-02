import cv2, re, numpy as np, math, os, PIL.Image as Image, shutil, ultralytics
import xmp_module
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

exif_data = {}

def get_frames(path, perframe):
    idxes = []
    try:
        vid = cv2.VideoCapture(path)
        width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        exif_data["Exif.Photo.PixelXDimension"] = width
        exif_data["Exif.Photo.PixelYDimension"] = height
    except:
        raise Exception("Video path is invalid")
    count, success = 0, True
    frames_list = []
    while success:
        success = vid.grab() # Read frame
        if success:
            if count % perframe == 0:
                print(f"image frame {count}")
                idxes.append(count)
                success, image = vid.retrieve()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image)
                frames_list.append(pil_img)
            count += 1
    vid.release() 
    return frames_list, idxes 

# dict keys = LONGITUDE, LATITUDE, REL_ALT, ABS_ALT
def srt_list(srt_path):
    with open(srt_path, "r") as srt_file:
        frames_list = [frames.split("\n") for frames in srt_file.read().strip().split("\n\n")]
        for frame in frames_list:
            # print(frame[4])
            new_dict = {}
            # for gpsinfo in re.findall(r'\w+:\s-?\d.?\d*', frame[4]):
            for gpsinfo in re.findall(r'\w+:\s-?\d+\.?\d*', frame[4]):
                key, value = gpsinfo.split(":")
                # print(f"key: {key}, value: {value}")
                if "/" in value:
                    num, denom = value.split("/")
                    new_dict[key.upper()] = float(num.strip()) / float(denom.strip())
                elif "]" in value:
                    value = value.split("]")
                    # print(f"value: {value}")
                    new_dict[key.upper()] = float(value[0].strip())
                else:
                    new_dict[key.upper()] = float(value.strip())
            frame[4] = new_dict
        return frames_list
    

def process_video(video_path, srt_path, per_frame, output_dir, counter_start):
    exif_data = {}
    srts = srt_list(srt_path=srt_path)
    try:
        vid = cv2.VideoCapture(video_path)
        width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        exif_data["Exif.Photo.PixelXDimension"] = width
        exif_data["Exif.Photo.PixelYDimension"] = height
    except:
        raise Exception("Video path is invalid")
    count = 0
    frame_counter = counter_start
    all_alts = []

    while True:
        success = vid.grab()
        if not success: 
            break
        if count % per_frame == 0:
            success, image = vid.retrieve()
            srt_obj = srts[count]
            pitch = float(srt_obj[4].get("GB_PITCH", 0)) 
            if -89 < pitch < -50:
                img_path = os.path.join(output_dir, f"frame_{frame_counter}.jpg")
                cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                xmp_meta = srt_obj[4]
                curr_alt = float(xmp_meta.get("REL_ALT"))
                if frame_counter > 20:
                    if valid_alt(all_alts, curr_alt):
                        all_alts.append(curr_alt)
                    else:
                        pass
                else: 
                    all_alts.append(curr_alt)
                xmp_module.write_xmp_exif(img_path, xmp_meta, exif_data)
                print(f"done writng frame {frame_counter}")
                frame_counter += 1
        count += 1
    return frame_counter

def valid_alt(alt_list, curr_alt):
    mean, std = np.mean(alt_list), np.std(alt_list)
    return mean - std <= curr_alt <= mean + std


def extract(bog_name, parent_dir, data_dir, frames_interval):
    print("Starting to extract frames from video stream...")
    video_file_list = sorted(data_dir.glob('*.MP4'))
    srt_file_list = sorted(data_dir.glob("*.SRT"))
    output_dir = parent_dir / f"{bog_name}_geotagged_frames"
    Path.mkdir(output_dir, exist_ok=True)

    current_frame_counter = 1
    for i, (video_file, srt_file) in enumerate(zip(video_file_list, srt_file_list), start=1):
        print(f"processing batch {i}")
        video_path = os.path.join(data_dir, video_file)
        srt_path = os.path.join(data_dir, srt_file)

        current_frame_counter = process_video(
            video_path=video_path,
            srt_path=srt_path,
            per_frame=frames_interval,
            output_dir=output_dir,
            counter_start=current_frame_counter
        )
    print(f"done extracting from video streams")
    return output_dir


# for testing purpose
if __name__ == "__main__":
    print("starting")
    bog_name = "Bog9"
    parent_dir = Path(__file__).resolve().parent # set to this directory, can be changed if needed
    data_dir = "DJI_202606291517_096_PineIslandbog9newH3m4x0video"
    data_dir = parent_dir / data_dir
    video_file_list = sorted(data_dir.glob('*.MP4'))
    srt_file_list = sorted(data_dir.glob("*.SRT"))
    for video, srt in zip(video_file_list, srt_file_list):
        print(video, srt)

    output_dir = parent_dir / f"{bog_name}_geotagged_frames"
    Path.mkdir(output_dir, exist_ok=True)

    current_frame_counter = 1
    for i, (video_file, srt_file) in enumerate(zip(video_file_list, srt_file_list), start=1):
        print(f"processing batch {i}")
        video_path = os.path.join(data_dir, video_file)
        srt_path = os.path.join(data_dir, srt_file)

        current_frame_counter = process_video(
            video_path=video_path,
            srt_path=srt_path,
            per_frame=50,
            output_dir=output_dir,
            counter_start=current_frame_counter
        )

    print(f"done extracting from video streams")