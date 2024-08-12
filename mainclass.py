# mainclass.py

from PySide6.QtWidgets import QMainWindow, QFileDialog, QProgressBar, QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QTreeView, QHeaderView, QCheckBox
from PySide6.QtCore import Qt, QThread, Signal, QDir, QModelIndex
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QStandardItemModel, QStandardItem
from pathlib import Path
import os

from frame_extractor import FrameExtractorThread

class CineFileModel(QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(['File'])

    def data(self, index, role):
        if role == Qt.CheckStateRole and index.column() == 0:
            return self.item(index.row()).checkState()
        return super().data(index, role)

    def setData(self, index, value, role):
        if role == Qt.CheckStateRole and index.column() == 0:
            self.item(index.row()).setCheckState(Qt.CheckState(value))
            return True
        return super().setData(index, value, role)

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsUserCheckable

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CINE Frame Extractor")
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_path = QLineEdit()
        folder_button = QPushButton("Select Input Folder")
        folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(folder_button)
        layout.addLayout(folder_layout)

        # Save path selection
        save_path_layout = QHBoxLayout()
        self.save_path = QLineEdit()
        save_path_button = QPushButton("Select Save Folder")
        save_path_button.clicked.connect(self.select_save_folder)
        save_path_layout.addWidget(self.save_path)
        save_path_layout.addWidget(save_path_button)
        layout.addLayout(save_path_layout)

        # Select All checkbox
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        layout.addWidget(self.select_all_checkbox)

        # File tree view
        self.file_model = CineFileModel()
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setHeaderHidden(False)
        self.file_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.file_tree)

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

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_path.setText(folder)
            self.populate_file_tree(folder)

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.save_path.setText(folder)

    def populate_file_tree(self, folder):
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels(['File'])
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.cine'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, folder)
                    item = QStandardItem(relative_path)
                    item.setCheckable(True)
                    item.setCheckState(Qt.Unchecked)
                    self.file_model.appendRow(item)
        self.select_all_checkbox.setChecked(False)

    def toggle_select_all(self, state):
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        for row in range(self.file_model.rowCount()):
            self.file_model.item(row).setCheckState(check_state)

    def start_extraction(self):
        if not self.save_path.text():
            QMessageBox.warning(self, "No Save Path", "Please select a save folder.")
            return

        selected_files = []
        for row in range(self.file_model.rowCount()):
            item = self.file_model.item(row)
            if item.checkState() == Qt.Checked:
                selected_files.append(os.path.join(self.folder_path.text(), item.text()))

        if not selected_files:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one CINE file for extraction.")
            return

        try:
            start_frame = int(self.start_frame.text())
            frame_count = None if self.frame_count.text().lower() == "none" else int(self.frame_count.text())
            stride = int(self.stride.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for Start Frame, Frame Count, and Stride.")
            return

        self.current_file_index = 0
        self.selected_files = selected_files
        self.process_next_file(start_frame, frame_count, stride)


    def process_next_file(self, start_frame, frame_count, stride):
        if self.current_file_index < len(self.selected_files):
            input_path = Path(self.selected_files[self.current_file_index])
            relative_path = input_path.relative_to(self.folder_path.text())
            
            # Create the output path
            output_base = Path(self.save_path.text()) / 'convert'
            output_path = output_base / relative_path.parent
            output_path.mkdir(parents=True, exist_ok=True)

            # Use the original filename (without extension) as the base for frame filenames
            base_filename = input_path.stem

            self.extractor_thread = FrameExtractorThread(
                input_path, 
                output_path, 
                start_frame, 
                frame_count, 
                stride, 
                base_filename
            )
            self.extractor_thread.progress.connect(self.update_progress)
            self.extractor_thread.log.connect(self.update_log)
            self.extractor_thread.finished.connect(self.extraction_finished)
            self.extractor_thread.error.connect(self.extraction_error)
            self.extractor_thread.start()

            self.extract_button.setEnabled(False)
            self.pause_resume_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        else:
            self.log_output.append("All selected files have been processed.")
            self.extract_button.setEnabled(True)
            self.pause_resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
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
        self.log_output.append(f"Extraction of file {self.selected_files[self.current_file_index]} completed in {total_time:.2f} seconds")
        self.current_file_index += 1
        self.process_next_file(int(self.start_frame.text()),
                               None if self.frame_count.text().lower() == "none" else int(self.frame_count.text()),
                               int(self.stride.text()))

    def extraction_error(self, error_message):
        self.log_output.append(f"Error: {error_message}")
        self.extract_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during extraction:\n\n{error_message}")

    def closeEvent(self, event):
        if hasattr(self, 'extractor_thread') and self.extractor_thread.isRunning():
            reply = QMessageBox.question(self, 'Exit', 'Extraction is still in progress. Are you sure you want to quit?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.extractor_thread.stop()
                self.extractor_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

