"""
OmniFusion-X V2: PyQt6 GUI Frontend
================================================================================
Complete GUI for training and inference with:
- Training tab with data selection, hyperparameters, progress monitoring
- Inference tab with chat interface, image generation, sampling controls
- Settings management and presets
- Real-time progress visualization

Requirements: pip install PyQt6

Author: OmniFusion Team
License: Apache 2.0
"""

import os
import subprocess
import numpy as np
from PIL import Image

# Fix for torch.compile/Triton on Windows
def setup_windows_compiler():
    if os.name != 'nt': return
    
    # Common paths for vcvars64.bat (VS 2022, 2019, Community, BuildTools, etc.)
    potential_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    
    vcvars_path = next((p for p in potential_paths if os.path.exists(p)), None)
    
    if vcvars_path:
        print(f"DEBUG: Found vcvars64.bat at {vcvars_path}. Capturing environment...", flush=True)
        try:
            # Captures environment variables from the .bat file
            command = f'"{vcvars_path}" && set'
            shell_env = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode(errors='ignore')
            
            for line in shell_env.splitlines():
                if "=" in line and not line.startswith("["):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
            
            import shutil
            cl_path = shutil.which("cl.exe")
            if cl_path:
                print(f"DEBUG: VS Environment Loaded. cl.exe found: {cl_path}", flush=True)
                os.environ["TRITON_CL_PATH"] = cl_path
                os.environ["CC"] = cl_path
                os.environ["CXX"] = cl_path
                
                # DLL Discovery for Triton on Windows
                try:
                    import triton
                    triton_dir = os.path.dirname(triton.__file__)
                    if os.path.exists(triton_dir):
                        print(f"DEBUG: Adding Triton DLL directory: {triton_dir}", flush=True)
                        if hasattr(os, "add_dll_directory"):
                            # Store handle to prevent GC
                            if not hasattr(setup_windows_compiler, "_dll_handles"):
                                setup_windows_compiler._dll_handles = []
                            setup_windows_compiler._dll_handles.append(os.add_dll_directory(triton_dir))
                        
                        # Also add to PATH at the front
                        os.environ["PATH"] = triton_dir + os.pathsep + os.environ.get("PATH", "")
                except Exception as e:
                    print(f"Warning: Triton check failed: {e}", flush=True)
            else:
                print("Warning: vcvars64.bat ran but cl.exe still not in PATH.", flush=True)
        except Exception as e:
            print(f"Error: Failed to load VS environment: {e}", flush=True)
    else:
        print("Warning: Visual Studio vcvars64.bat not found in standard paths.", flush=True)
        print("Note: If torch.compile fails, please run the GUI from a 'Developer Command Prompt for VS'.", flush=True)

setup_windows_compiler()

import sys
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict

# PyQt6 imports
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
        QDoubleSpinBox, QCheckBox, QComboBox, QSlider, QProgressBar,
        QFileDialog, QGroupBox, QFormLayout, QScrollArea, QSplitter,
        QListWidget, QListWidgetItem, QMessageBox, QStatusBar, QFrame,
        QGridLayout, QSizePolicy
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QPixmap, QImage, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not installed. Run: pip install PyQt6")

import torch

# Import backends
from backend import TrainingConfig, train_model, TrainingCallback
from inference_backend import (
    InferenceEngine, SamplingConfig, InstructConfig, ImageGenConfig,
    load_inference_engine
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GUI")


# =============================================================================
# Worker Threads
# =============================================================================

# Global stop flag for training (shared between GUI and trainer)
_training_stop_flag = {"stop": False}

class TrainingWorker(QThread):
    """Background thread for training."""
    
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, config: TrainingConfig, data_paths: List[str]):
        super().__init__()
        self.config = config
        self.data_paths = data_paths
        _training_stop_flag["stop"] = False
    
    def run(self):
        try:
            # Validate paths first
            valid_paths = [p for p in self.data_paths if p and os.path.exists(p)]
            if not valid_paths:
                self.error.emit("No valid data paths found. Please check that files/directories exist.")
                return
            
            def progress_callback(info):
                # Check for stop request
                if _training_stop_flag["stop"]:
                    raise KeyboardInterrupt("Training stopped by user")
                self.progress.emit(info)
            
            state = train_model(
                valid_paths,
                asdict(self.config), # Pass config as dictionary
                callback=progress_callback,
                stop_flag=_training_stop_flag,
            )
            
            if _training_stop_flag["stop"]:
                self.finished.emit({"stopped": True})
            else:
                self.finished.emit(state.to_dict() if hasattr(state, 'to_dict') else {})
        
        except KeyboardInterrupt:
            self.finished.emit({"stopped": True, "message": "Training stopped by user"})
        except Exception as e:
            import traceback
            # Print to terminal for visibility
            print(f"\n{'='*60}")
            print(f"TRAINING ERROR: {e}")
            print(f"{'='*60}")
            traceback.print_exc()
            self.error.emit(str(e))
    
    def stop(self):
        _training_stop_flag["stop"] = True
        self.wait(2000)  # Wait up to 2 seconds for thread to finish
        if self.isRunning():
            self.terminate()  # Force terminate if still running



class InferenceWorker(QThread):
    """Background thread for inference."""
    
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, engine: InferenceEngine, task: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.task = task
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.task == "load":
                # Handle model loading as a task
                from inference_backend import load_inference_engine
                result = load_inference_engine(**self.kwargs)
            elif self.task == "text":
                result = self.engine.generate_text(**self.kwargs)
            elif self.task == "chat":
                result = self.engine.chat(**self.kwargs)
            elif self.task == "image":
                result = self.engine.generate_image(**self.kwargs)
            elif self.task == "multimodal":
                result = self.engine.generate_multimodal(**self.kwargs)
            else:
                result = None
            
            self.result.emit(result)
        
        except Exception as e:
            import traceback
            print(f"\n--- GPU THREAD ERROR ---\n", flush=True)
            traceback.print_exc()
            print(f"--- END ERROR ---\n", flush=True)
            self.error.emit(str(e))


# =============================================================================
# Custom Widgets
# =============================================================================

class LabeledSpinBox(QWidget):
    """SpinBox with label."""
    
    def __init__(self, label: str, min_val: int, max_val: int, default: int):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.spinbox = QSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default)
        
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
    
    def value(self) -> int:
        return self.spinbox.value()
    
    def setValue(self, v: int):
        self.spinbox.setValue(v)


class LabeledDoubleSpinBox(QWidget):
    """DoubleSpinBox with label."""
    
    def __init__(self, label: str, min_val: float, max_val: float, default: float, decimals: int = 4):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setSingleStep(0.01)
        
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
    
    def value(self) -> float:
        return self.spinbox.value()
    
    def setValue(self, v: float):
        self.spinbox.setValue(v)


class LabeledSlider(QWidget):
    """Slider with label and value display."""
    
    def __init__(self, label: str, min_val: float, max_val: float, default: float, decimals: int = 2):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QHBoxLayout()
        self.label = QLabel(label)
        self.value_label = QLabel(f"{default:.{decimals}f}")
        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.value_label)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        
        self.setValue(default)
        self.slider.valueChanged.connect(self._update_label)
        
        layout.addLayout(header)
        layout.addWidget(self.slider)
    
    def _update_label(self):
        self.value_label.setText(f"{self.value():.{self.decimals}f}")
    
    def value(self) -> float:
        ratio = self.slider.value() / 1000.0
        return self.min_val + ratio * (self.max_val - self.min_val)
    
    def setValue(self, v: float):
        ratio = (v - self.min_val) / (self.max_val - self.min_val)
        self.slider.setValue(int(ratio * 1000))


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that displays values in scientific notation (e.g., 4e-4)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(10)
        self.setRange(1e-10, 1.0)
        # Default step, will be overridden by stepBy
        self.setSingleStep(1e-5)

    def stepBy(self, steps: int):
        """Adjusts step size based on current value magnitude (e.g., 10% of value)."""
        current_val = self.value()
        # Set step to 1/10th of current value to maintain precision across scales
        self.setSingleStep(current_val * 0.1 if current_val > 0 else 1e-6)
        super().stepBy(steps)

    def textFromValue(self, value: float) -> str:
        if value == 0: return "0"
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")

    def valueFromText(self, text: str) -> float:
        try: return float(text)
        except ValueError: return self.value()
    
    def textFromValue(self, value: float) -> str:
        """Convert internal value to display text in scientific notation."""
        if value == 0:
            return "0"
        # Format as scientific notation like "4e-4"
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")
    
    def valueFromText(self, text: str) -> float:
        """Parse user input - accepts both decimal and scientific notation."""
        try:
            return float(text)
        except ValueError:
            return self.value()
    
    def validate(self, text: str, pos: int):
        """Allow typing scientific notation."""
        from PyQt6.QtGui import QValidator
        # Allow partial entries like "4e" or "4e-"
        if text in ["", "-", ".", "e", "E"] or text.endswith(("e", "E", "e-", "E-", "e+", "E+")):
            return (QValidator.State.Intermediate, text, pos)
        try:
            val = float(text)
            if self.minimum() <= val <= self.maximum():
                return (QValidator.State.Acceptable, text, pos)
            return (QValidator.State.Intermediate, text, pos)
        except ValueError:
            return (QValidator.State.Invalid, text, pos)


# =============================================================================
# Training Tab
# =============================================================================

class TrainingTab(QWidget):
    """Training configuration and monitoring tab."""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE: Settings Scroll Area ---
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(480)

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # 1. Data & Processing (Maximized)
        data_group = QGroupBox("Data & Preprocessing")
        data_layout = QFormLayout(data_group)
        self.data_sources_list = QListWidget()
        self.data_sources_list.setMaximumHeight(100)
        
        btn_box = QHBoxLayout()
        add_dir = QPushButton("+ Dir"); add_dir.clicked.connect(self.add_data_directory)
        add_file = QPushButton("+ File"); add_file.clicked.connect(self.add_data_files)
        clear_src = QPushButton("Clear"); clear_src.clicked.connect(lambda: self.data_sources_list.clear())
        btn_box.addWidget(add_dir); btn_box.addWidget(add_file); btn_box.addWidget(clear_src)
        
        self.max_image_size = QSpinBox(); self.max_image_size.setRange(1, 1000000); self.max_image_size.setValue(512)
        self.max_text_length = QSpinBox(); self.max_text_length.setRange(1, 1000000); self.max_text_length.setValue(2048)
        self.patch_size = QSpinBox(); self.patch_size.setRange(1, 1000000); self.patch_size.setValue(8) # New: Patch size control
        
        data_layout.addRow("Sources:", self.data_sources_list)
        data_layout.addRow("", btn_box)
        data_layout.addRow("Max Image Res:", self.max_image_size)
        data_layout.addRow("Max Seq Length:", self.max_text_length)
        data_layout.addRow("Vision Patch Size:", self.patch_size)
        settings_layout.addWidget(data_group)
        
        # 1b. Text Training Directories (ChatML, Alpaca, etc.)
        text_group = QGroupBox("Text Training Data")
        text_layout = QFormLayout(text_group)
        self.text_dirs_list = QListWidget()
        self.text_dirs_list.setMaximumHeight(80)
        
        text_btn_box = QHBoxLayout()
        add_text_dir = QPushButton("+ Text Dir"); add_text_dir.clicked.connect(self.add_text_directory)
        clear_text = QPushButton("Clear"); clear_text.clicked.connect(lambda: self.text_dirs_list.clear())
        text_btn_box.addWidget(add_text_dir); text_btn_box.addWidget(clear_text)
        
        self.alpha_ntp_text = LabeledDoubleSpinBox("Text Loss (α):", 0.0, 10.0, 1.0, 2)
        
        text_layout.addRow("Text Dirs:", self.text_dirs_list)
        text_layout.addRow("", text_btn_box)
        text_layout.addRow(self.alpha_ntp_text)
        settings_layout.addWidget(text_group)

        # 2. Training & Scheduling (Maximized)
        train_group = QGroupBox("Training Engine")
        train_layout = QFormLayout(train_group)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10000000); self.epochs.setValue(900)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 1000000); self.batch_size.setValue(4)
        self.grad_accum = QSpinBox(); self.grad_accum.setRange(1, 1000000); self.grad_accum.setValue(1)
        self.learning_rate = ScientificDoubleSpinBox(); self.learning_rate.setRange(0.0, 1000.0); self.learning_rate.setValue(2e-4) # Matches run_training_v2.bat
        self.warmup_steps = QSpinBox(); self.warmup_steps.setRange(0, 10000000); self.warmup_steps.setValue(100)
        self.scheduler_type = QComboBox(); self.scheduler_type.addItems(["cosine", "linear", "constant", "one_cycle"])
        self.weight_decay = QDoubleSpinBox(); self.weight_decay.setRange(0.0, 1000.0); self.weight_decay.setValue(0.05)
        
        train_layout.addRow("Epochs:", self.epochs)
        train_layout.addRow("Batch Size:", self.batch_size)
        train_layout.addRow("Grad Accum:", self.grad_accum)
        train_layout.addRow("Learning Rate:", self.learning_rate)
        train_layout.addRow("Warmup Steps:", self.warmup_steps)
        train_layout.addRow("Scheduler:", self.scheduler_type)
        train_layout.addRow("Weight Decay:", self.weight_decay)
        settings_layout.addWidget(train_group)

        # 3. Optimizations & Z-Turbo (Maximized)
        opt_group = QGroupBox("Optimizations & Z-Turbo")
        opt_layout = QGridLayout(opt_group)
        self.use_8bit = QCheckBox("8-bit AdamW"); self.use_8bit.setChecked(True)
        self.use_ema = QCheckBox("EMA Weights"); self.use_ema.setChecked(True)
        self.compile_model = QCheckBox("torch.compile"); self.compile_model.setChecked(False)  # Disabled by default to prevent freeze
        self.use_grad_ckpt = QCheckBox("Grad Checkpointing"); self.use_grad_ckpt.setChecked(False) # New: Save VRAM
        self.use_min_snr = QCheckBox("Min-SNR Weighting"); self.use_min_snr.setChecked(True)
        self.use_noise_bank = QCheckBox("Noise Bank"); self.use_noise_bank.setChecked(True)
        
        opt_layout.addWidget(self.use_8bit, 0, 0); opt_layout.addWidget(self.use_ema, 0, 1)
        opt_layout.addWidget(self.compile_model, 1, 0); opt_layout.addWidget(self.use_grad_ckpt, 1, 1)
        opt_layout.addWidget(self.use_min_snr, 2, 0); opt_layout.addWidget(self.use_noise_bank, 2, 1)
        
        self.neftune_alpha = LabeledDoubleSpinBox("NeFTune Alpha:", 0.0, 1000.0, 0.0)
        self.lambda_img = LabeledDoubleSpinBox("Img Loss Scale (λ):", 0.0, 100.0, 5.0)  # Default matches run_training_v2.bat
        settings_layout.addWidget(opt_group)
        settings_layout.addWidget(self.neftune_alpha)
        settings_layout.addWidget(self.lambda_img)

        # 4. Checkpointing
        ckpt_group = QGroupBox("Output & Checkpointing")
        ckpt_layout = QFormLayout(ckpt_group)
        self.save_every = QSpinBox(); self.save_every.setRange(1, 1000000); self.save_every.setValue(10)
        self.checkpoint_dir = QLineEdit("checkpoints")
        ckpt_layout.addRow("Save Every N Epochs:", self.save_every)
        ckpt_layout.addRow("Export Path:", self.checkpoint_dir)
        settings_layout.addWidget(ckpt_group)

        settings_layout.addStretch()
        settings_scroll.setWidget(settings_widget)
        splitter.addWidget(settings_scroll)

        # --- RIGHT SIDE: Progress & Output ---
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        
        mon_group = QGroupBox("Monitor")
        mon_layout = QGridLayout(mon_group)
        self.loss_label = QLabel("Loss: -"); self.step_label = QLabel("Step: 0")
        self.lr_display = QLabel("LR: -"); self.epoch_label = QLabel("Epoch: 0")
        self.progress_bar = QProgressBar()

        # Wire up labels for on_progress compatibility (renaming to match class variables if needed)
        self.lr_label = self.lr_display 
        
        mon_layout.addWidget(self.epoch_label, 0, 0); mon_layout.addWidget(self.step_label, 0, 1)
        mon_layout.addWidget(self.loss_label, 1, 0); mon_layout.addWidget(self.lr_display, 1, 1)
        mon_layout.addWidget(self.progress_bar, 2, 0, 1, 2)
        right_layout.addWidget(mon_group)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        right_layout.addWidget(QLabel("Real-time Logs:"))
        right_layout.addWidget(self.log_text)
        
        btns = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Training"); self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("🛑 Stop"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_training)
        btns.addWidget(self.start_btn); btns.addWidget(self.stop_btn)
        
        # Add Save/Load buttons as they are useful
        save_btn = QPushButton("Save"); save_btn.clicked.connect(self.save_config)
        load_btn = QPushButton("Load"); load_btn.clicked.connect(self.load_config)
        btns.addStretch()
        btns.addWidget(save_btn); btns.addWidget(load_btn)

        right_layout.addLayout(btns)

        splitter.addWidget(right_widget)
        main_layout.addWidget(splitter)
    
    def add_data_directory(self):
        """Add a directory to data sources."""
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            # Check if already in list
            for i in range(self.data_sources_list.count()):
                if self.data_sources_list.item(i).text().startswith(f"📁 {path}"):
                    return
            
            item = QListWidgetItem(f"📁 {path}")
            item.setData(Qt.ItemDataRole.UserRole, {"path": path, "type": "directory"})
            self.data_sources_list.addItem(item)
    
    def add_data_files(self):
        """Add files to data sources."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Data Files", "",
            "Data Files (*.json *.jsonl *.txt *.parquet *.xml);;All Files (*)"
        )
        if paths:
            for path in paths:
                item = QListWidgetItem(f"📄 {os.path.basename(path)}")
                item.setData(Qt.ItemDataRole.UserRole, {"path": path, "type": "file"})
                item.setToolTip(path)
                self.data_sources_list.addItem(item)
    
    def remove_data_source(self):
        """Remove selected data source."""
        current = self.data_sources_list.currentRow()
        if current >= 0:
            self.data_sources_list.takeItem(current)
    
    def get_data_paths(self) -> List[str]:
        """Get all configured data paths."""
        paths = []
        for i in range(self.data_sources_list.count()):
            item = self.data_sources_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and "path" in data:
                paths.append(data["path"])
        return paths
    
    def add_text_directory(self):
        """Add a text data directory (ChatML, Alpaca, etc.)."""
        path = QFileDialog.getExistingDirectory(self, "Select Text Data Directory")
        if path:
            # Check if already in list
            for i in range(self.text_dirs_list.count()):
                if self.text_dirs_list.item(i).text() == path:
                    return
            self.text_dirs_list.addItem(path)
    
    def get_text_data_dirs(self) -> str:
        """Get comma-separated list of text directories."""
        dirs = []
        for i in range(self.text_dirs_list.count()):
            dirs.append(self.text_dirs_list.item(i).text())
        return ",".join(dirs)

    
    def browse_checkpoint_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if path:
            self.checkpoint_dir.setText(path)
    
    def get_config(self) -> TrainingConfig:
        """Get current config from UI."""
        return TrainingConfig(
            # Data
            max_image_size=self.max_image_size.value(),
            max_text_length=self.max_text_length.value(),
            patch_size=self.patch_size.value(),
            
            # Training
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            gradient_accumulation_steps=self.grad_accum.value(),
            learning_rate=self.learning_rate.value(),
            warmup_steps=self.warmup_steps.value(),
            scheduler_type=self.scheduler_type.currentText(),
            weight_decay=self.weight_decay.value(),
            
            # Optimizations
            use_8bit_adam=self.use_8bit.isChecked(),
            use_ema=self.use_ema.isChecked(),
            compile_model=self.compile_model.isChecked(),
            gradient_checkpointing=self.use_grad_ckpt.isChecked(),
            use_min_snr_weighting=self.use_min_snr.isChecked(),
            use_noise_bank=self.use_noise_bank.isChecked(),
            neftune_alpha=self.neftune_alpha.value(),
            
            # Checkpointing
            save_every_n_epochs=self.save_every.value(),
            checkpoint_dir=self.checkpoint_dir.text(),

            # Model architecture defaults (matching run_training_v2.bat)
            d_model=512, n_layers=8, n_heads=8,
            ema_decay=0.9999,
            ema_update_every=50,  # Matches batch file
            use_logit_normal_sampling=True,
            lambda_img=self.lambda_img.value(),
            
            # Text Training
            text_data_dirs=self.get_text_data_dirs(),
            alpha_ntp_text_only=self.alpha_ntp_text.value(),
        )
    
    def set_config(self, config: TrainingConfig):
        """Set UI from config."""
        self.max_image_size.setValue(getattr(config, 'max_image_size', 512))
        self.max_text_length.setValue(getattr(config, 'max_text_length', 2048))
        self.patch_size.setValue(getattr(config, 'patch_size', 8))
        
        self.epochs.setValue(getattr(config, 'epochs', 100))
        self.batch_size.setValue(getattr(config, 'batch_size', 4))
        self.grad_accum.setValue(getattr(config, 'gradient_accumulation_steps', 4))
        self.learning_rate.setValue(getattr(config, 'learning_rate', 4e-4))
        self.warmup_steps.setValue(getattr(config, 'warmup_steps', 100))
        self.scheduler_type.setCurrentText(getattr(config, 'scheduler_type', 'cosine'))
        self.weight_decay.setValue(getattr(config, 'weight_decay', 0.05))
        
        self.use_8bit.setChecked(getattr(config, 'use_8bit_adam', True))
        self.use_ema.setChecked(getattr(config, 'use_ema', True))
        self.compile_model.setChecked(getattr(config, 'compile_model', False))
        self.use_grad_ckpt.setChecked(getattr(config, 'gradient_checkpointing', False))
        self.use_min_snr.setChecked(getattr(config, 'use_min_snr_weighting', True))
        self.use_noise_bank.setChecked(getattr(config, 'use_noise_bank', True))
        self.neftune_alpha.setValue(getattr(config, 'neftune_alpha', 0.0))
        self.lambda_img.setValue(getattr(config, 'lambda_img', 5.0))
        self.alpha_ntp_text.setValue(getattr(config, 'alpha_ntp_text_only', 1.0))
        
        # Text directories
        text_dirs = getattr(config, 'text_data_dirs', '')
        self.text_dirs_list.clear()
        if text_dirs:
            for d in text_dirs.split(','):
                if d.strip():
                    self.text_dirs_list.addItem(d.strip())
        
        self.save_every.setValue(getattr(config, 'save_every_n_epochs', 10))
        self.checkpoint_dir.setText(getattr(config, 'checkpoint_dir', 'checkpoints'))
    
    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "training_config.json", "JSON Files (*.json)"
        )
        if path:
            config = self.get_config()
            # Manual JSON serialization since TrainingConfig may not have save()
            config_dict = asdict(config)
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.log("Saved config to " + path)
    
    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "JSON Files (*.json)"
        )
        if path:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            # Filter to valid TrainingConfig fields
            valid_keys = set(TrainingConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
            config = TrainingConfig(**filtered)
            self.set_config(config)
            self.log("Loaded config from " + path)
    
    def start_training(self):
        paths = [p for p in self.get_data_paths() if p.strip() and os.path.exists(p)]
        if not paths: # Checks if paths actually exist on disk
            QMessageBox.warning(self, "Error", "No valid existing data paths found.")
            return
        
        config = self.get_config()
        
        self.progress_bar.setValue(0)
        self.loss_label.setText("Loss: -")
        self.step_label.setText("Step: 0")
        self.lr_label.setText("LR: -")
        
        config = self.get_config()
        self.log("Starting training...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.worker = TrainingWorker(config, paths)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def stop_training(self):
        if self.worker:
            self.stop_btn.setEnabled(False) # Prevent double-clicking
            self.log("Requesting stop...")
            self.worker.stop()
            
            # Clear CUDA cache and garbage collect to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.log("CUDA cache cleared.")
            
            # Ensure start button re-enables even if thread takes time to exit
            QTimer.singleShot(3000, lambda: self.start_btn.setEnabled(True))
    
    def on_progress(self, info: dict):
        if info.get("type") == "step":
            self.step_label.setText(f"Step: {info['step']}")
            # Update progress bar (assuming info contains total_steps)
            if "total_steps" in info:
                progress = int((info['step'] / info['total_steps']) * 100)
                self.progress_bar.setValue(progress)
            self.loss_label.setText(f"Loss: {info['loss']:.6f}")
            self.lr_label.setText(f"LR: {info['lr']:.2e}")
            self.epoch_label.setText(f"Epoch: {info['epoch']}")
        
        elif info.get("type") == "epoch":
            epoch = info["epoch"]
            metrics = info.get("metrics", {})
            self.log(f"Epoch {epoch}: {metrics}")
    
    def on_finished(self, state: dict):
        if state.get("stopped"):
            self.log("Training stopped by user.")
        else:
            self.log("Training finished!")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def on_error(self, error: str):
        self.log(f"ERROR: {error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Training Error", error)
    
    def log(self, message: str):
        self.log_text.append(message)
        # Fix: Auto-scroll to the bottom
        self.log_text.ensureCursorVisible()

# =============================================================================
# Inference Tab
# =============================================================================

class InferenceTab(QWidget):
    """Inference and generation tab."""
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.worker = None
        self.messages = []
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # --- LEFT SIDE: Chat/Generation ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        mode_tabs = QTabWidget()

        # Quick image setting (visible for both Chat and Image tabs).
        # This avoids "CFG missing" confusion when the right settings panel is scrolled/ignored.
        quick_img_settings = QHBoxLayout()
        quick_img_settings.addWidget(QLabel("CFG:"))
        self.cfg_scale_quick = QDoubleSpinBox()
        self.cfg_scale_quick.setRange(1.0, 20.0)
        self.cfg_scale_quick.setSingleStep(0.5)
        self.cfg_scale_quick.setValue(7.5)
        quick_img_settings.addWidget(self.cfg_scale_quick)
        quick_img_settings.addStretch()
        
        # Chat mode
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.returnPressed.connect(self.send_message)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_btn)
        
        chat_layout.addWidget(self.chat_display)
        chat_layout.addLayout(input_layout)
        mode_tabs.addTab(chat_widget, "Chat")
        
        # Image generation mode
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        self.image_prompt = QTextEdit()
        self.image_prompt.setPlaceholderText("Enter image prompt...")
        self.image_prompt.setMaximumHeight(100)
        
        img_btn_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate Image")
        generate_btn.clicked.connect(self.generate_image)
        img_btn_layout.addWidget(generate_btn)
        img_btn_layout.addStretch()
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMinimumSize(256, 256)
        self.image_display.setStyleSheet("border: 1px solid gray;")
        
        image_layout.addWidget(self.image_prompt)
        image_layout.addLayout(img_btn_layout)
        image_layout.addWidget(self.image_display)
        mode_tabs.addTab(image_widget, "Image Generation")
        
        left_layout.addLayout(quick_img_settings)
        left_layout.addWidget(mode_tabs)

        # --- RIGHT SIDE: Settings ---
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(380)

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # 1. Model Loading Group (Fixed variables here)
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout(model_group)
        
        load_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to checkpoint...")
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_model)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_model)
        
        load_layout.addWidget(self.model_path)
        load_layout.addWidget(browse_btn)
        load_layout.addWidget(self.load_btn)
        
        self.model_status = QLabel("No model loaded")
        model_layout.addLayout(load_layout)
        model_layout.addWidget(self.model_status)
        settings_layout.addWidget(model_group)
        
        # 2. Sampling Settings
        sampling_group = QGroupBox("Sampling")
        sampling_layout = QVBoxLayout(sampling_group)
        self.temperature = LabeledSlider("Temperature:", 0.0, 2.0, 0.85)
        self.top_p = LabeledSlider("Top P:", 0.0, 1.0, 1.0)
        self.top_k = LabeledSpinBox("Top K:", 0, 200, 0)
        self.min_p = LabeledSlider("Min P:", 0.0, 0.5, 0.1)
        self.rep_penalty = LabeledSlider("Repetition Penalty:", 1.0, 2.0, 1.0)
        self.max_tokens = LabeledSpinBox("Max Tokens:", 1, 4096, 512)
        
        for w in [self.temperature, self.top_p, self.top_k, self.min_p, self.rep_penalty, self.max_tokens]:
            sampling_layout.addWidget(w)
        settings_layout.addWidget(sampling_group)
        
        # 3. Image Generation Settings
        img_settings_group = QGroupBox("Image Config")
        img_settings_layout = QFormLayout(img_settings_group)
        self.img_width = QSpinBox(); self.img_width.setRange(64, 2048); self.img_width.setValue(512)
        self.img_height = QSpinBox(); self.img_height.setRange(64, 2048); self.img_height.setValue(512)
        self.img_steps = QSpinBox(); self.img_steps.setRange(1, 100); self.img_steps.setValue(20)
        self.cfg_scale = QDoubleSpinBox(); self.cfg_scale.setRange(1.0, 20.0); self.cfg_scale.setValue(7.5)
        self.solver = QComboBox(); self.solver.addItems(["euler", "midpoint"])

        # Keep quick CFG control (image tab) and settings-panel CFG control in sync.
        if hasattr(self, "cfg_scale_quick") and self.cfg_scale_quick is not None:
            self.cfg_scale_quick.blockSignals(True)
            self.cfg_scale_quick.setValue(self.cfg_scale.value())
            self.cfg_scale_quick.blockSignals(False)

            def _sync_cfg_to_panel(v: float):
                self.cfg_scale.blockSignals(True)
                self.cfg_scale.setValue(v)
                self.cfg_scale.blockSignals(False)

            def _sync_cfg_to_quick(v: float):
                self.cfg_scale_quick.blockSignals(True)
                self.cfg_scale_quick.setValue(v)
                self.cfg_scale_quick.blockSignals(False)

            self.cfg_scale_quick.valueChanged.connect(_sync_cfg_to_panel)
            self.cfg_scale.valueChanged.connect(_sync_cfg_to_quick)
        
        img_settings_layout.addRow("Width:", self.img_width)
        img_settings_layout.addRow("Height:", self.img_height)
        img_settings_layout.addRow("Steps:", self.img_steps)
        img_settings_layout.addRow("CFG Scale:", self.cfg_scale)
        img_settings_layout.addRow("Solver:", self.solver)
        settings_layout.addWidget(img_settings_group)
        
        # 4. System Prompt
        system_group = QGroupBox("System Prompt")
        system_layout = QVBoxLayout(system_group)
        self.system_prompt = QTextEdit()
        self.system_prompt.setMaximumHeight(100)
        system_layout.addWidget(self.system_prompt)
        settings_layout.addWidget(system_group)
        
        # 5. Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        load_preset_btn = QPushButton("Load Settings")
        load_preset_btn.clicked.connect(self.load_preset)
        save_preset_btn = QPushButton("Save Settings")
        save_preset_btn.clicked.connect(self.save_settings)
        preset_layout.addWidget(load_preset_btn)
        preset_layout.addWidget(save_preset_btn)
        settings_layout.addWidget(preset_group)
        
        settings_layout.addStretch()
        settings_scroll.setWidget(settings_widget)
        
        layout.addWidget(left_widget, 2)
        layout.addWidget(settings_scroll, 1)
        
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "Checkpoint Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_path.setText(path)
    
    def load_model(self):
        path = self.model_path.text()
        self.model_status.setText("⌛ Loading Model...")
        self.load_btn.setEnabled(False) # Need to make sure load_btn is available
        
        self.worker = InferenceWorker(None, "load", checkpoint_path=path if path else None)
        self.worker.result.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_model_error)
        self.worker.start()

    def on_model_loaded(self, engine):
        self.engine = engine
        self.model_status.setText("✅ Model loaded")
        self.load_btn.setEnabled(True)
        # Auto-update UI from engine config
        if hasattr(self.engine, 'omni_config'):
             self.chat_display.append(f"<i>Model loaded: {self.engine.omni_config.d_model}d, {self.engine.omni_config.n_layers}L</i>")

    def on_model_error(self, error):
        self.model_status.setText("❌ Load Error")
        self.load_btn.setEnabled(True)
        print(f"MODEL LOAD ERROR: {error}", flush=True)
        QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{error}")
    
    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ST Settings", "", "JSON Files (*.json)"
        )
        if path and self.engine:
            try:
                self.engine.load_st_settings(path)
                
                # Update UI
                self.temperature.setValue(self.engine.sampling_config.temperature)
                self.top_p.setValue(self.engine.sampling_config.top_p)
                self.top_k.setValue(self.engine.sampling_config.top_k)
                self.min_p.setValue(self.engine.sampling_config.min_p)
                self.rep_penalty.setValue(self.engine.sampling_config.repetition_penalty)
                self.max_tokens.setValue(self.engine.sampling_config.max_new_tokens)
                
                QMessageBox.information(self, "Success", "Settings loaded!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "settings.json", "JSON Files (*.json)"
        )
        if path:
            settings = {
                "sampling": self.get_sampling_config().to_dict(),
                "image": self.get_image_config().to_dict(),
            }
            with open(path, "w") as f:
                json.dump(settings, f, indent=2)
    
    def get_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            temperature=self.temperature.value(),
            top_p=self.top_p.value(),
            top_k=self.top_k.value(),
            min_p=self.min_p.value(),
            repetition_penalty=self.rep_penalty.value(),
            max_new_tokens=self.max_tokens.value(),
        )
    
    def get_image_config(self) -> ImageGenConfig:
        return ImageGenConfig(
            width=self.img_width.value(),
            height=self.img_height.value(),
            num_steps=self.img_steps.value(),
            guidance_scale=self.cfg_scale.value(),
        )
    
    def send_message(self):
        if not self.engine:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return
        
        text = self.chat_input.text().strip()
        if not text:
            return
        
        self.chat_input.clear()
        
        # Add user message
        self.messages.append({"role": "user", "content": text})
        self.chat_display.append(f"\n<b>User:</b> {text}")
        
        # Generate response
        try:
            self.chat_display.append("\n<b>Assistant:</b> <i>Generating multimodal response...</i>")
            QApplication.processEvents()
            
            # Use multimodal task for autonomy support
            self.worker = InferenceWorker(
                self.engine, 
                "multimodal", 
                prompt=text,
                max_new_tokens=self.max_tokens.value()
            )
            # Connect signals
            self.worker.result.connect(self.on_inference_finished) 
            self.worker.error.connect(self.on_inference_error)
            self.worker.start()
            
        except Exception as e:
            self.chat_display.append(f"\n<b>Launch Error:</b> {e}")
    
    def generate_image(self):
            if not self.engine:
                QMessageBox.warning(self, "Error", "Please load a model first")
                return
            
            prompt = self.image_prompt.toPlainText().strip()
            if not prompt: return

            self.image_display.setText("Generating...")
            
            # Start background worker
            self.worker = InferenceWorker(
                self.engine, 
                "image", 
                prompt=prompt, 
                config=self.get_image_config()
            )
            self.worker.result.connect(self.on_inference_finished)
            self.worker.error.connect(lambda e: self.image_display.setText(f"Error: {e}"))
            self.worker.start()

    def on_inference_finished(self, result):
        """Unified slot to handle both Text and Image results from the worker."""
        if isinstance(result, dict):
            # Multimodal Result (text + list of images)
            text = result.get("text", "")
            images = result.get("images", [])
            
            self.chat_display.append(f"<b>Assistant:</b> {text}")
            self.messages.append({"role": "assistant", "content": text})
            
            if images:
                for i, img in enumerate(images):
                    self.display_chat_image(img, f"Generated Image {i+1}")
        
        elif isinstance(result, str):
            # Text Only Result
            self.chat_display.append(f"<b>Assistant:</b> {result}")
            self.messages.append({"role": "assistant", "content": result})
        
        elif isinstance(result, Image.Image):
            # Image Only Result (from Generate Image button or direct task)
            self.image_display.setPixmap(self.pil_to_pixmap(result))
            self.image_display.setText("") # Clear "Generating..."
        
        elif isinstance(result, torch.Tensor):
            # Raw Tensor Result (backward compatibility)
            img_pil = self.tensor_to_pil(result)
            self.image_display.setPixmap(self.pil_to_pixmap(img_pil))
        
        self.worker = None

    def on_inference_error(self, error: str):
        """Handle inference errors."""
        print(f"INFERENCE ERROR: {error}", flush=True)
        self.chat_display.append(f"\n<b>Error:</b> {error}")
        self.image_display.setText("Error")
        self.worker = None

    def display_chat_image(self, pil_img, caption=""):
        # Update the main image display with the latest generated image
        pixmap = self.pil_to_pixmap(pil_img)
        self.image_display.setPixmap(pixmap)
        self.chat_display.append(f"<i>[{caption} displayed in Image Generation tab]</i>")

    def pil_to_pixmap(self, pil_img):
        import io
        from PyQt6.QtGui import QPixmap
        byte_array = io.BytesIO()
        pil_img.save(byte_array, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(byte_array.getvalue())
        return pixmap.scaled(self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)

    def tensor_to_pil(self, result):
        import numpy as np # Local import for safety
        img_np = ((result.cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        if img_np.shape[0] == 3: img_np = img_np.transpose(1, 2, 0)
        return Image.fromarray(img_np)

# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniFusion-X V2")
        self.setMinimumSize(1200, 800)
        
        # Central widget with tabs
        tabs = QTabWidget()
        
        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTab()
        
        tabs.addTab(self.training_tab, "🎯 Training")
        tabs.addTab(self.inference_tab, "💬 Inference")
        
        self.setCentralWidget(tabs)
        
        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.status.showMessage(f"Ready | Device: {device}")
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #252526;
                color: #cccccc;
            }
            QGroupBox {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666666;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #4c4c4c;
                border-radius: 4px;
                padding: 4px;
                color: #cccccc;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #0e639c;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 8px 20px;
                border: 1px solid #3c3c3c;
            }
            QTabBar::tab:selected {
                background-color: #252526;
                border-bottom-color: #252526;
            }
            QScrollBar:vertical {
                background-color: #252526;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #4c4c4c;
                border-radius: 6px;
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3c3c3c;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0e639c;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    if not PYQT_AVAILABLE:
        print("PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
