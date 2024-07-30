import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math
import multiprocessing
import zipfile
import os

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
    """Order from:
    http://www.visionresearch.com/phantomzone/viewtopic.php?f=20&t=572#p3884
    """
    # 1. Offset the raw image by the amount in flare

    # 2. White balance the raw picture using the white balance component of cmatrix
    BayerPatterns = {3: "gbrg", 4: "rggb"}
    pattern = BayerPatterns[setup.CFA]

    raw = whitebalance_raw(raw.astype(np.float32), setup, pattern).astype(np.uint16)

    # 3. Debayer the image
    rgb_image = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)

    # convert to float
    rgb_image = rgb_image.astype(np.float32) / (2**bpp - 1)

    # return rgb_image

    # 4. Apply the color correction matrix component of cmatrix
    #
    # From the documentation:
    # ...should decompose this
    # matrix in two components: a diagonal one with the white balance to be
    # applied before interpolation and a normalized one to be applied after
    # interpolation.

    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)

    # normalize matrix
    ccm = cmCalib / cmCalib.sum(axis=1)[:, np.newaxis]

    # or should it be normalized this way?
    ccm2 = cmCalib.copy()
    ccm2[0][0] = 1 - ccm2[0][1] - ccm2[0][2]
    ccm2[1][1] = 1 - ccm2[1][0] - ccm2[1][2]
    ccm2[2][2] = 1 - ccm2[2][0] - ccm2[2][1]

    m = np.asarray(
        [
            1.4956012040024347,
            -0.5162879962189262,
            0.020686792216491584,
            -0.09884672458400766,
            0.757682383759598,
            0.34116434082440983,
            -0.04121405804689133,
            -0.5527871476076358,
            1.5940012056545272,
        ]
    ).reshape(3, 3)

    rgb_image = np.dot(rgb_image, m.T)
    # rgb_reshaped = rgb_image.reshape((rgb_image.shape[0] * rgb_image.shape[1], rgb_image.shape[2]))
    # rgb_image = np.dot(m, rgb_reshaped.T).T.reshape(rgb_image.shape)

    # 5. Apply the user RGB matrix umatrix
    # cmUser = np.asarray(setup.cmUser).reshape(3, 3)
    # rgb_image = np.dot(rgb_image, cmUser.T)

    # 6. Offset the image by the amount in offset

    # 7. Apply the global gain

    # 8. Apply the per-component gains red, green, blue

    # 9. Apply the gamma curves; the green channel uses gamma, red uses gamma + rgamma and blue uses gamma + bgamma
    rgb_image = apply_gamma(rgb_image, setup)

    # 10. Apply the tone curve to each of the red, green, blue channels
    fTone = np.asarray(setup.fTone)

    # 11. Add the pedestals to each color channel, and linearly rescale to keep the white point the same.

    # 12. Convert to YCrCb using REC709 coefficients

    # 13. Scale the Cr and Cb components by chroma.

    # 14. Rotate the Cr and Cb components around the origin in the CrCb plane by hue degrees.

    return rgb_image


def to_3ch_gray(frame: np.ndarray):
    return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)[
        ..., None
    ].repeat(3, axis=-1)

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math
import multiprocessing
import zipfile
import os

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QProgressBar, QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from PySide6.QtCore import Qt, QThread, Signal

import cv2
import numpy as np
from tqdm import tqdm
from pycine.raw import read_frames
from pycine.file import read_header

# 기존 함수들 (apply_gamma, whitebalance_raw, gen_mask, color_pipeline, to_3ch_gray)은 그대로 유지

def process_frame(args):
    frame, setup, bpp, save_path = args
    processed_frame = to_3ch_gray(color_pipeline(frame, setup=setup, bpp=bpp))
    cv2.imwrite(str(save_path), processed_frame)
    return save_path

class FrameExtractorThread(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(float)

    def __init__(self, video_path, save_dir, stride):
        super().__init__()
        self.video_path = video_path
        self.save_dir = save_dir
        self.stride = stride
        self.stop_flag = False

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

        raw_images_gen = read_frames(self.video_path)
        first_frame_data = next(raw_images_gen)
        
        # 첫 번째 프레임에서 setup과 bpp 정보 추출
        if isinstance(first_frame_data, tuple) and len(first_frame_data) == 3:
            first_frame, setup, bpp = first_frame_data
        else:
            self.log.emit("Unexpected data structure from read_frames")
            return

        setup.CFA = 3  # Assuming CFA is always 3, modify if needed

        total_frames_to_process = total_frames // self.stride

        max_digits = len(str(total_frames))
        time_format = "%y%m%d_%H%M%S.%f"

        csv_file = self.save_dir / f"{Path(self.video_path).stem}_processing_log.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame Number", "Timestamp", "Group", "Filename"])

            current_group = 1
            group_start_time = start_time_frame

            # 멀티프로세싱 설정
            pool = multiprocessing.Pool()
            frame_data = []

            # 첫 번째 프레임 처리
            frame_time = start_time_frame
            group_dir = self.save_dir / Path(self.video_path).stem / f"frame_group_{current_group:04d}of{total_groups:04d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            filename = f"0_{Path(self.video_path).stem}_{frame_time.strftime(time_format)}.png"
            save_path = group_dir / filename
            frame_data.append((first_frame, setup, bpp, save_path))
            csv_writer.writerow([0, frame_time.strftime(time_format), current_group, filename])

            # 나머지 프레임 처리
            for i, (frame, _, _) in enumerate(raw_images_gen, start=1):
                if self.stop_flag:
                    pool.terminate()
                    self.log.emit("Extraction stopped by user")
                    return

                if i % self.stride != 0:
                    continue

                frame_time = start_time_frame + i * time_per_frame
                
                if frame_time - group_start_time >= group_interval:
                    current_group += 1
                    group_start_time = frame_time
                
                group_dir = self.save_dir / Path(self.video_path).stem / f"frame_group_{current_group:04d}of{total_groups:04d}"
                group_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{i:0{max_digits}d}_{Path(self.video_path).stem}_{frame_time.strftime(time_format)}.png"
                save_path = group_dir / filename

                frame_data.append((frame, setup, bpp, save_path))
                csv_writer.writerow([i, frame_time.strftime(time_format), current_group, filename])

                if len(frame_data) >= 100:  # 프레임을 100개씩 처리
                    self.process_frames(pool, frame_data, total_frames)
                    frame_data = []

            # 남은 프레임 처리
            if frame_data:
                self.process_frames(pool, frame_data, total_frames)

            pool.close()
            pool.join()
        # 압축
        self.log.emit("Compressing extracted frames...")
        zip_file = self.save_dir / f"{Path(self.video_path).stem}_extracted_frames.zip"
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.save_dir / Path(self.video_path).stem):
                for file in files:
                    zipf.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                                               self.save_dir / Path(self.video_path).stem))

        end_time = time.time()
        total_time = end_time - start_time
        self.finished.emit(total_time)
        
    def process_frames(self, pool, frame_data, total_frames):
        for i, _ in enumerate(pool.imap_unordered(process_frame, frame_data)):
            progress = int((i + 1) / total_frames * 100)
            self.progress.emit(progress)
            self.log.emit(f"Processed frame {i} of {total_frames}")

    def stop(self):
        self.stop_flag = True
        
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
        self.stride = QLineEdit("10")
        param_layout.addWidget(QLabel("Stride:"))
        param_layout.addWidget(self.stride)
        layout.addLayout(param_layout)

        # Extract and Stop buttons
        button_layout = QHBoxLayout()
        self.extract_button = QPushButton("Extract Frames")
        self.extract_button.clicked.connect(self.start_extraction)
        self.stop_button = QPushButton("Stop Extraction")
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.extract_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

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
        stride = int(self.stride.text())

        self.extractor_thread = FrameExtractorThread(input_path, output_path, stride)
        self.extractor_thread.progress.connect(self.update_progress)
        self.extractor_thread.log.connect(self.update_log)
        self.extractor_thread.finished.connect(self.extraction_finished)
        self.extractor_thread.start()

        self.extract_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_extraction(self):
        if hasattr(self, 'extractor_thread'):
            self.extractor_thread.stop()
        self.stop_button.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

    def extraction_finished(self, total_time):
        self.log_output.append(f"Extraction completed in {total_time:.2f} seconds")
        self.extract_button.setEnabled(True)
        self.stop_button.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())