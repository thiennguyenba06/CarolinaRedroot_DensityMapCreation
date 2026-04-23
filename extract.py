import cv2, re, numpy as np, math, os, PIL.Image as Image, shutil, ultralytics
import xmp_module
from concurrent.futures import ProcessPoolExecutor

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
                xmp_module.write_xmp_exif(img_path, xmp_meta, exif_data)
                print(f"done writng frame {frame_counter}")
                frame_counter += 1
        count += 1
    return frame_counter


            


if __name__ == "__main__":
    parent_dir = "./" # set to this directory, can be changed if needed
    data_dir = "YOUR VIDEO FOLDER DIRECTORY"
    data_dir = os.path.join(parent_dir, "DJI_202507011226_138_PineIslandbog9H3m3x0video")
    video_file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".MP4")])
    srt_file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".SRT")])
    for video, srt in zip(video_file_list, srt_file_list):
        print(video, srt)


    output_dir = os.path.join(parent_dir, "geotagged_frames")
    os.makedirs(output_dir, exist_ok=True)

    current_frame_counter = 1
    for i, (video_file, srt_file) in enumerate(zip(video_file_list, srt_file_list), start=1):
        print(f"processing batch {i}")
        video_path = os.path.join(data_dir, video_file)
        srt_path = os.path.join(data_dir, srt_file)

        current_frame_counter = process_video(
            video_path=video_path,
            srt_path=srt_path,
            per_frame=15,
            output_dir=output_dir,
            counter_start=current_frame_counter
        )

    print(f"done extracting from video streams")


    
