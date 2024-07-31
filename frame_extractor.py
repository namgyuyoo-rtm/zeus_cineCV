# frame_extractor.py

import time
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math
import gc
import os
import psutil
import ctypes

from PySide6.QtCore import QThread, Signal

import cv2
import numpy as np
from pycine.raw import read_frames
from pycine.file import read_header

from cine_settings import color_pipeline

def to_3ch_gray(frame: np.ndarray):
    return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None].repeat(3, axis=-1)

def get_frames(video_path: str, start_frame: int, count: int, stride: int = 1, cfa: int = 3):
    chunk_size = 100  # 한 번에 처리할 프레임 수
    for i in range(0, count, chunk_size):
        chunk_count = min(chunk_size, count - i)
        raw_images, setup, bpp = read_frames(video_path, start_frame=start_frame + i * stride, count=chunk_count)
        setup.CFA = cfa
        
        for j, raw_image in enumerate(raw_images):
            if j % stride == 0:
                yield to_3ch_gray(color_pipeline(raw_image, setup=setup, bpp=bpp))
        
        del raw_images
        gc.collect()

# Windows에서 디스크 캐시를 플러시하는 함수
def flush_disk_cache():
    try:
        ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
        return True
    except AttributeError:
        return False

class FrameExtractorThread(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(float)

    def __init__(self, video_path, save_dir, start_frame, frame_count, stride):
        super().__init__()
        self.video_path = video_path
        self.save_dir = save_dir
        self.start_frame = start_frame
        self.frame_count = frame_count
        self.stride = stride
        self.is_paused = False
        self.is_stopped = False

    def run(self):
        start_time = time.time()
        
        header = read_header(self.video_path)
        trigger_time = header["cinefileheader"].TriggerTime
        total_frames = header["cinefileheader"].ImageCount
        frame_rate = header["setup"].FrameRate

        self.log.emit(f"Total Frames: {total_frames}, Frame Rate: {frame_rate}")

        start_time_frame = datetime.fromtimestamp(trigger_time.seconds + trigger_time.fractions / 1e6)
        end_time = start_time_frame + timedelta(seconds=total_frames / frame_rate)
        time_per_frame = timedelta(seconds=1 / frame_rate)

        self.log.emit(f"Start Time: {start_time_frame}, End Time: {end_time}, Time per Frame: {time_per_frame.total_seconds():.6f} seconds")

        total_duration = end_time - start_time_frame
        group_interval = timedelta(milliseconds=300)
        total_groups = math.ceil(total_duration / group_interval)

        total_frames_to_process = total_frames if self.frame_count is None else min(self.frame_count, total_frames - self.start_frame)
        total_frames_to_process = math.ceil(total_frames_to_process / self.stride)

        max_digits = len(str(total_frames))
        time_format = "%y%m%d_%H%M%S.%f"

        csv_file = self.save_dir / f"{Path(self.video_path).stem}_processing_log.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame Number", "Timestamp", "Group", "Filename"])

            current_group = 1
            group_start_time = start_time_frame
            frames_processed = 0

            while frames_processed < total_frames_to_process and not self.is_stopped:
                while self.is_paused:
                    time.sleep(0.1)
                    if self.is_stopped:
                        break

                if self.is_stopped:
                    break

                group_frames = get_frames(str(self.video_path), self.start_frame + frames_processed * self.stride, 
                                          min(100, total_frames_to_process - frames_processed), self.stride)
                
                for i, frame in enumerate(group_frames):
                    frame_number = self.start_frame + (frames_processed + i) * self.stride
                    frame_time = start_time_frame + frame_number * time_per_frame
                    
                    if frame_time - group_start_time >= group_interval:
                        current_group += 1
                        group_start_time = frame_time
                    
                    group_dir = Path(self.save_dir) / Path(self.video_path).stem / f"frame_group_{current_group:04d}of{total_groups:04d}"
                    group_dir.mkdir(parents=True, exist_ok=True)

                    filename = f"{frame_number:0{max_digits}d}_{Path(self.video_path).stem}_{frame_time.strftime(time_format)}.png"
                    save_path = group_dir / filename

                    try:
                        # 디렉토리 존재 여부 다시 확인
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        cv2.imwrite(str(save_path), frame)
                        
                        # 파일 존재 확인
                        if save_path.exists():
                            self.log.emit(f"Saved image: {save_path}")
                        else:
                            self.log.emit(f"Failed to save image: {save_path}")
                    except Exception as e:
                        self.log.emit(f"Error saving image {save_path}: {str(e)}")

                    csv_writer.writerow([frame_number, frame_time.strftime(time_format), current_group, filename])

                    frames_processed += 1
                    progress = int(frames_processed / total_frames_to_process * 100)
                    self.progress.emit(progress)
                    self.log.emit(f"Processed frame {frame_number} of {total_frames}")

                    # 메모리 사용량 로깅 (매 10 프레임마다)
                    if frames_processed % 10 == 0:
                        mem = psutil.virtual_memory()
                        self.log.emit(f"Memory usage: {mem.percent}% (Used: {mem.used / 1024 / 1024:.2f} MB, Available: {mem.available / 1024 / 1024:.2f} MB)")

                    # 디스크 캐시 플러시 (매 100 프레임마다)
                    if frames_processed % 100 == 0:
                        if flush_disk_cache():
                            self.log.emit("Disk cache flushed")
                        else:
                            self.log.emit("Failed to flush disk cache")

                # 그룹 처리 후 메모리 정리
                del group_frames
                gc.collect()

        end_time = time.time()
        total_time = end_time - start_time
        self.finished.emit(total_time)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_stopped = True