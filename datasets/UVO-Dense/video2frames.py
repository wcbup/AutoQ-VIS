# ------------------------------------------------------------------------------------------------
# Modified by Kaixuan Lu from https://sites.google.com/view/unidentified-video-object/dataset

import cv2
import os
from tqdm import tqdm


def split_single_video(video_path, frames_dir=""):
	cap = cv2.VideoCapture(video_path)
	cnt = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			success, buffer = cv2.imencode(".png", frame)
			if success:
				with open(f"{frames_dir}{cnt}.png", "wb") as f:
					f.write(buffer.tobytes())
					f.flush()
				cnt += 1
		else:
			break


# rename with the directory where you stored videos
video_dir = "/tmp/uvo/uvo_videos_dense/"
# rename with the directory where you would like to store frames
frames_dir = "/tmp/uvo/uvo_videos_dense_frames/"
all_video_paths = os.listdir(video_dir)
TOTAL = 0
for video_path in tqdm(all_video_paths):
	v_frame_dir = f"{frames_dir}{video_path[:-4]}/"
	os.mkdir(v_frame_dir)
	split_single_video(f"{video_dir}{video_path}", frames_dir=v_frame_dir)
	TOTAL += 1
	# if TOTAL > 5:
	# 	break