#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import subprocess


if __name__ == '__main__':
    if len(sys.argv != 2):
        print("Usage: python3 convert_video_to_images.py <path-to-SURREAL-dataset>")
    DATA_PATH = sys.argv[1]
    dataset_dir = os.path.realpath(DATA_PATH)
    basenames = sorted(os.listdir(dataset_dir))
    maps_files = []
    info_files = []
    frames_paths = []
    for basename in basenames:
        one_data_dir = os.path.join(dataset_dir, basename)
        one_dir_videos = glob.glob(os.path.join(one_data_dir,
                                                basename + '_c*.mp4'))
        for video_filename in one_dir_videos:
            frames_dir_name = video_filename[:-4] + '_frames'
            frames_dir_path = os.path.join(one_data_dir, frames_dir_name)
            os.mkdir(frames_dir_path)
            frames_name = os.path.join(frames_dir_path, 'f%04d.jpg')
            subprocess.run(['ffmpeg', '-i', video_filename, 
                            '-qscale:v', '2', frames_name])
