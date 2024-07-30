import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math
import concurrent.futures

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QProgressBar, QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from PySide6.QtCore import Qt, QThread, Signal

import cv2
import numpy as np
from tqdm import tqdm
from pycine.raw import read_frames
from pycine.file import read_header

def apply_gamma(rgb_image, setup):
    # FIXME: using 2.2 for now because 8.0 from the sample image seems way out of place
    # --> this is not at all how vri is doing it!
    rgb_image = np.abs(rgb_image) ** (1.0 / 2.2)
    # rgb_image[:, :, 0] **= (1.0 / (setup.fGammaR + setup.fGamma))
    # rgb_image[:, :, 1] **= (1.0 / setup.fGamma)
    # rgb_image[:, :, 2] **= (1.0 / (setup.fGammaB + setup.fGamma))

    return rgb_image


def whitebalance_raw(raw, setup, pattern):
    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    whitebalance = np.diag(cmCalib)
    whitebalance = [1.193739671606806, 1.0, 1.7885392465247287]

    # FIXME: maybe use .copy()
    wb_raw = np.ma.MaskedArray(raw)

    wb_raw.mask = gen_mask(pattern, "r", wb_raw)
    wb_raw *= whitebalance[0]
    wb_raw.mask = gen_mask(pattern, "g", wb_raw)
    wb_raw *= whitebalance[1]
    wb_raw.mask = gen_mask(pattern, "b", wb_raw)
    wb_raw *= whitebalance[2]

    wb_raw.mask = np.ma.nomask

    return wb_raw


def gen_mask(pattern, c, image):
    def color_kern(pattern, c):
        return np.array(
            [[pattern[0] != c, pattern[1] != c], [pattern[2] != c, pattern[3] != c]]
        )

    (h, w) = image.shape[:2]
    cells = np.ones((h // 2, w // 2))

    return np.kron(cells, color_kern(pattern, c))

def color_pipeline(raw, setup, bpp=12):
    BayerPatterns = {3: "gbrg", 4: "rggb"}
    pattern = BayerPatterns[setup.CFA]

    raw = whitebalance_raw(raw.astype(np.float32), setup, pattern).astype(np.uint16)
    rgb_image = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)
    rgb_image = rgb_image.astype(np.float32) / (2**bpp - 1)

    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    m = np.asarray([
        1.4956012040024347, -0.5162879962189262, 0.020686792216491584,
        -0.09884672458400766, 0.757682383759598, 0.34116434082440983,
        -0.04121405804689133, -0.5527871476076358, 1.5940012056545272
    ]).reshape(3, 3)

    rgb_image = np.dot(rgb_image, m.T)
    rgb_image = apply_gamma(rgb_image, setup)

    return rgb_image

def to_3ch_gray(frame: np.ndarray):
    return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None].repeat(3, axis=-1)

def process_batch(args):
    video_path, start_frame, count, stride, setup, bpp = args
    raw_images, _, _ = read_frames(video_path, start_frame=start_frame, count=count)
    processed_frames = []
    for i, raw_image in enumerate(raw_images):
        if i % stride == 0:
            processed = to_3ch_gray(color_pipeline(raw_image, setup=setup, bpp=bpp))
            processed_frames.append(processed)
    return processed_frames

def get_frames(video_path: str, start_frame: int, count: int = None, stride: int = 1, cfa: int = 3):
    header = read_header(video_path)
    total_frames = header["cinefileheader"].ImageCount
    
    if count is None:
        count = total_frames - start_frame
    else:
        count = min(count, total_frames - start_frame)

    # 멀티프로세싱을 위한 배치 크기 설정
    batch_size = 100
    num_batches = math.ceil(count / batch_size)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for batch in range(num_batches):
            batch_start = start_frame + batch * batch_size
            batch_count = min(batch_size, count - batch * batch_size)
            
            args = (video_path, batch_start, batch_count, stride, cfa)
            futures.append(executor.submit(process_batch, args))
        
        for future in concurrent.futures.as_completed(futures):
            yield from future.result()

def process_batch(args):
    video_path, batch_start, batch_count, stride, cfa = args
    raw_images, setup, bpp = read_frames(video_path, start_frame=batch_start, count=batch_count)
    setup.CFA = cfa
    
    processed_frames = []
    for i, raw_image in enumerate(raw_images):
        if i % stride == 0:
            processed = to_3ch_gray(color_pipeline(raw_image, setup=setup, bpp=bpp))
            processed_frames.append(processed)
    return processed_frames
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

        self.log.emit(f"Start Time: {start_time_frame}, End Time: {end_time}, Time per Frame: {time_per_frame}")

        total_duration = end_time - start_time_frame
        group_interval = timedelta(milliseconds=300)
        total_groups = math.ceil(total_duration / group_interval)

        frames = get_frames(str(self.video_path), self.start_frame, self.frame_count, self.stride)
   
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

            for i, frame in enumerate(frames):
                frame_time = start_time_frame + i * self.stride * time_per_frame
                
                if frame_time - group_start_time >= group_interval:
                    current_group += 1
                    group_start_time = frame_time
                
                group_dir = self.save_dir / Path(self.video_path).stem / f"frame_group_{current_group:04d}of{total_groups:04d}"
                group_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{(self.start_frame + i*self.stride):0{max_digits}d}_{Path(self.video_path).stem}_{frame_time.strftime(time_format)}.png"
                save_path = group_dir / filename

                cv2.imwrite(save_path.as_posix(), frame)

                csv_writer.writerow([self.start_frame + i*self.stride, frame_time.strftime(time_format), current_group, filename])

                progress = int((i + 1) / total_frames_to_process * 100)
                self.progress.emit(progress)
                self.log.emit(f"Processed frame {self.start_frame + i*self.stride} of {total_frames}")

        end_time = time.time()
        total_time = end_time - start_time
        self.finished.emit(total_time)
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CINE Frame Extractor")
        self.setGeometry(100, 100, 600, 400)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # Input file selection
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        input_button = QPushButton("Select CINE File")
        input_button.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        output_button = QPushButton("Select Output Directory")
        output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Parameters
        param_layout = QHBoxLayout()
        self.start_frame = QLineEdit("0")
        self.frame_count = QLineEdit("None")
        self.stride = QLineEdit("10")
        param_layout.addWidget(QLabel("Start Frame:"))
        param_layout.addWidget(self.start_frame)
        param_layout.addWidget(QLabel("Frame Count:"))
        param_layout.addWidget(self.frame_count)
        param_layout.addWidget(QLabel("Stride:"))
        param_layout.addWidget(self.stride)
        layout.addLayout(param_layout)

        # Extract button
        self.extract_button = QPushButton("Extract Frames")
        self.extract_button.clicked.connect(self.start_extraction)
        layout.addWidget(self.extract_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        main_widget.setLayout(layout)

    def select_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CINE File", "", "CINE Files (*.cine)")
        if file_name:
            self.input_path.setText(file_name)

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)

    def start_extraction(self):
        input_path = Path(self.input_path.text())
        output_path = Path(self.output_path.text())
        start_frame = int(self.start_frame.text())
        frame_count = None if self.frame_count.text() == "None" else int(self.frame_count.text())
        stride = int(self.stride.text())

        self.extractor_thread = FrameExtractorThread(input_path, output_path, start_frame, frame_count, stride)
        self.extractor_thread.progress.connect(self.update_progress)
        self.extractor_thread.log.connect(self.update_log)
        self.extractor_thread.finished.connect(self.extraction_finished)
        self.extractor_thread.start()

        self.extract_button.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

    def extraction_finished(self, total_time):
        self.log_output.append(f"Extraction completed in {total_time:.2f} seconds")
        self.extract_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())