# mainclass.py

from PySide6.QtWidgets import QMainWindow, QFileDialog, QProgressBar, QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path

from frame_extractor import FrameExtractorThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CINE Frame Extractor")
        self.setGeometry(100, 100, 600, 400)
        self.setAcceptDrops(True)  # 드래그 앤 드롭 활성화

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

        # Control buttons
        control_layout = QHBoxLayout()
        self.extract_button = QPushButton("Extract Frames")
        self.extract_button.clicked.connect(self.start_extraction)
        self.pause_resume_button = QPushButton("Pause")
        self.pause_resume_button.clicked.connect(self.toggle_pause_resume)
        self.pause_resume_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.extract_button)
        control_layout.addWidget(self.pause_resume_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        main_widget.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.cine'):
                self.input_path.setText(file_path)
                self.update_output_path(file_path)
                break

    def select_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CINE File", "", "CINE Files (*.cine)")
        if file_name:
            self.input_path.setText(file_name)
            self.update_output_path(file_name)

    def update_output_path(self, input_path):
        input_dir = Path(input_path).parent
        output_dir = input_dir / 'cineextract'
        self.output_path.setText(str(output_dir))

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)

    def start_extraction(self):
        input_path = Path(self.input_path.text())
        output_path = Path(self.output_path.text())
        start_frame = int(self.start_frame.text())
        frame_count = None if self.frame_count.text().lower() == "none" else int(self.frame_count.text())
        stride = int(self.stride.text())

        self.extractor_thread = FrameExtractorThread(input_path, output_path, start_frame, frame_count, stride)
        self.extractor_thread.progress.connect(self.update_progress)
        self.extractor_thread.log.connect(self.update_log)
        self.extractor_thread.finished.connect(self.extraction_finished)
        self.extractor_thread.start()

        self.extract_button.setEnabled(False)
        self.pause_resume_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def toggle_pause_resume(self):
        if self.extractor_thread.is_paused:
            self.extractor_thread.resume()
            self.pause_resume_button.setText("Pause")
        else:
            self.extractor_thread.pause()
            self.pause_resume_button.setText("Resume")

    def stop_extraction(self):
        self.extractor_thread.stop()
        self.extract_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

    def extraction_finished(self, total_time):
        self.log_output.append(f"Extraction completed in {total_time:.2f} seconds")
        self.extract_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)