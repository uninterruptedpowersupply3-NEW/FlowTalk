
import tkinter as tk
from tkinter import ttk, messagebox, Label, Entry, Button, Text, Scrollbar, Checkbutton, IntVar, StringVar, filedialog
import subprocess
import threading
import sys
import os
from PIL import Image, ImageTk

# Configuration
PYTHON_EXE = sys.executable
INFERENCE_SCRIPT = "inference_backend.py"
TRAINING_SCRIPT = "test_dataset_generalization.py"
OUTPUT_IMAGE = "gui_gen_output.png"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("OmniFusion Pro GUI")
        self.root.geometry("1200x900")
        
        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.init_inference_tab()
        self.init_training_tab()
        
    def init_inference_tab(self):
        self.inf_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.inf_frame, text="Chat & Inference")
        
        # Grid Layout
        self.inf_frame.columnconfigure(0, weight=1) # Chat area
        self.inf_frame.columnconfigure(1, weight=0) # Controls area
        self.inf_frame.rowconfigure(0, weight=1)

        # === LEFT COLUMN: Chat Interface ===
        chat_pane = ttk.Frame(self.inf_frame)
        chat_pane.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Chat History Display
        self.chat_display = tk.Text(chat_pane, wrap='word', state='disabled', font=("Segoe UI", 10))
        self.chat_display.pack(fill='both', expand=True, pady=(0, 10))
        
        # System Prompt (Collapsible/Expandable)
        self.sys_prompt_expanded = tk.BooleanVar(value=False)
        sys_header = ttk.Frame(chat_pane)
        sys_header.pack(fill='x', pady=(5, 0))
        
        self.sys_toggle_btn = tk.Button(sys_header, text="▶ System Prompt", command=self._toggle_system_prompt,
                                        font=("Segoe UI", 9), relief='flat', anchor='w')
        self.sys_toggle_btn.pack(side='left')
        
        self.sys_prompt_frame = ttk.Frame(chat_pane)
        # Initially collapsed - don't pack
        
        self.sys_prompt_text = tk.Text(self.sys_prompt_frame, height=4, width=50, font=("Segoe UI", 9), wrap='word')
        self.sys_prompt_text.pack(fill='x', padx=5, pady=5)
        self.sys_prompt_text.insert('1.0', "You are a helpful AI assistant.")


        # User Input Area
        input_frame = ttk.Frame(chat_pane)
        input_frame.pack(fill='x')
        
        self.prompt_entry = tk.Text(input_frame, height=4, font=("Segoe UI", 10))
        self.prompt_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Send Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side='right', fill='y')
        
        self.text_btn = tk.Button(btn_frame, text="SEND TEXT", command=self.run_text_generation, bg="#007bff", fg="white", font=('Arial', 9, 'bold'), width=12)
        self.text_btn.pack(pady=2)
        
        self.gen_btn = tk.Button(btn_frame, text="GEN IMAGE", command=self.run_generation, bg="#28a745", fg="white", font=('Arial', 9, 'bold'), width=12)
        self.gen_btn.pack(pady=2)
        
        tk.Button(btn_frame, text="CLEAR CHAT", command=self.clear_chat, bg="#dc3545", fg="white", font=('Arial', 8)).pack(pady=2)

        # === Multi-Image Selection ===
        img_select_frame = ttk.LabelFrame(chat_pane, text="Attached Images (use <image> in prompt)")
        img_select_frame.pack(fill='x', pady=5)
        
        img_btn_row = ttk.Frame(img_select_frame)
        img_btn_row.pack(fill='x', padx=5, pady=5)
        
        tk.Button(img_btn_row, text="Add Images", command=self._add_images, bg="#6c757d", fg="white").pack(side='left', padx=2)
        tk.Button(img_btn_row, text="Clear Images", command=self._clear_images, bg="#dc3545", fg="white").pack(side='left', padx=2)
        self.img_count_label = tk.Label(img_btn_row, text="0 images selected")
        self.img_count_label.pack(side='left', padx=10)
        
        # Gallery frame for thumbnails
        self.gallery_frame = ttk.Frame(img_select_frame)
        self.gallery_frame.pack(fill='x', padx=5, pady=5)
        
        # Storage for selected images
        self.selected_images = []  # List of file paths
        self.gallery_thumbnails = []  # Keep refs to prevent GC

        # === RIGHT COLUMN: Controls ===
        ctrl_pane = ttk.LabelFrame(self.inf_frame, text="Settings")
        ctrl_pane.grid(row=0, column=1, sticky='ns', padx=10, pady=10)
        
        def add_entry(parent, label, var, r):
            tk.Label(parent, text=label).grid(row=r, column=0, sticky='e', pady=5)
            tk.Entry(parent, textvariable=var, width=10).grid(row=r, column=1, sticky='w', padx=5, pady=5)

        # Basic
        self.seed_var = tk.StringVar(value="42")
        add_entry(ctrl_pane, "Seed:", self.seed_var, 0)
        
        self.max_tokens_var = tk.StringVar(value="512")
        add_entry(ctrl_pane, "Max Tokens:", self.max_tokens_var, 1)

        # Sampling
        sep = ttk.Separator(ctrl_pane, orient='horizontal')
        sep.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)
        tk.Label(ctrl_pane, text="Sampling Parameters", font=("Arial", 9, "bold")).grid(row=3, column=0, columnspan=2)

        self.temperature_var = tk.StringVar(value="0.7")
        add_entry(ctrl_pane, "Temperature:", self.temperature_var, 4)
        
        self.top_k_var = tk.StringVar(value="40")
        add_entry(ctrl_pane, "Top-K:", self.top_k_var, 5)
        
        self.top_p_var = tk.StringVar(value="0.9")
        add_entry(ctrl_pane, "Top-P:", self.top_p_var, 6)
        
        self.min_p_var = tk.StringVar(value="0.05")
        add_entry(ctrl_pane, "Min-P:", self.min_p_var, 7)
        
        self.rep_penalty_var = tk.StringVar(value="1.2")
        add_entry(ctrl_pane, "Rep. Penalty:", self.rep_penalty_var, 8)

        # Image Settings
        sep2 = ttk.Separator(ctrl_pane, orient='horizontal')
        sep2.grid(row=9, column=0, columnspan=2, sticky='ew', pady=10)
        tk.Label(ctrl_pane, text="Image Settings", font=("Arial", 9, "bold")).grid(row=10, column=0, columnspan=2)

        self.width_var = tk.StringVar(value="256")
        add_entry(ctrl_pane, "Width:", self.width_var, 11)
        
        self.height_var = tk.StringVar(value="256")
        add_entry(ctrl_pane, "Height:", self.height_var, 12)
        
        self.steps_var = tk.StringVar(value="50")
        add_entry(ctrl_pane, "Steps:", self.steps_var, 13)

        # CFG guidance scale (used by inference_backend.InferenceModel.generate_image as `cfg`)
        self.cfg_var = tk.StringVar(value="7.5")
        add_entry(ctrl_pane, "CFG Scale:", self.cfg_var, 14)

        # Model Path & Persistent Engine
        sep3 = ttk.Separator(ctrl_pane, orient='horizontal')
        sep3.grid(row=15, column=0, columnspan=2, sticky='ew', pady=10)
        
        tk.Label(ctrl_pane, text="Checkpoint Path:").grid(row=16, column=0, columnspan=2, sticky='w')
        self.inf_model_var = tk.StringVar(value="")
        self.input_model_var = tk.StringVar(value="") # Fallback from training tab
        tk.Entry(ctrl_pane, textvariable=self.inf_model_var, width=25).grid(row=17, column=0, columnspan=2, padx=5)
        tk.Button(ctrl_pane, text="Browse Checkpoint", command=lambda: self._browse_inf_model()).grid(row=18, column=0, columnspan=2, pady=5)
        
        # [NEW] Persistent Model Controls
        self.engine = None
        self.engine_lock = threading.Lock()
        
        btn_row = ttk.Frame(ctrl_pane)
        btn_row.grid(row=19, column=0, columnspan=2, pady=5)
        
        self.load_btn = tk.Button(btn_row, text="Load Model", command=self._load_model_persistent, bg="#17a2b8", fg="white", width=11)
        self.load_btn.pack(side='left', padx=2)
        
        self.unload_btn = tk.Button(btn_row, text="Unload (VRAM)", command=self._unload_model_persistent, state='disabled', width=11)
        self.unload_btn.pack(side='left', padx=2)
        
        self.engine_status_label = tk.Label(ctrl_pane, text="Status: Unloaded", fg="gray")
        self.engine_status_label.grid(row=20, column=0, columnspan=2)
        
        # Image Output Preview (Small)
        self.img_label = tk.Label(ctrl_pane, text="[Img Preview]", bg="#ddd", width=25, height=10)
        self.img_label.grid(row=21, column=0, columnspan=2, pady=10, sticky='ew')
        
        # State
        self.chat_history = [] # List of {"role": "user"|"assistant", "content": "..."}

    def clear_chat(self):
        self.chat_history = []
        self.chat_display.config(state='normal')
        self.chat_display.delete('1.0', 'end')
        self.chat_display.config(state='disabled')
        messagebox.showinfo("Chat Cleared", "Conversation history has been reset.")
    
    def _toggle_system_prompt(self):
        """Toggle system prompt visibility."""
        if self.sys_prompt_expanded.get():
            # Collapse
            self.sys_prompt_frame.pack_forget()
            self.sys_toggle_btn.config(text="▶ System Prompt")
            self.sys_prompt_expanded.set(False)
        else:
            # Expand
            self.sys_prompt_frame.pack(fill='x', pady=(0, 5), before=self.prompt_entry.master)
            self.sys_toggle_btn.config(text="▼ System Prompt")
            self.sys_prompt_expanded.set(True)
    
    def _update_ratio_label(self, val):
        """Update ratio label when slider moves."""
        ratio = float(val)
        img_pct = int(ratio * 100)
        txt_pct = 100 - img_pct
        self.ratio_label.config(text=f"{img_pct}% img / {txt_pct}% text")

    
    def _add_images(self):
        """Open file dialog to select multiple images."""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.selected_images.extend(files)
            self._update_gallery()
    
    def _clear_images(self):
        """Clear all selected images."""
        self.selected_images = []
        self._update_gallery()
    
    def _update_gallery(self):
        """Update the gallery display with thumbnails."""
        # Clear existing thumbnails
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        self.gallery_thumbnails = []
        
        # Update count label
        count = len(self.selected_images)
        self.img_count_label.config(text=f"{count} image{'s' if count != 1 else ''} selected")
        
        # Create thumbnails (max 6 shown)
        for i, img_path in enumerate(self.selected_images[:6]):
            try:
                pil_img = Image.open(img_path)
                pil_img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                self.gallery_thumbnails.append(tk_img)  # Keep ref
                
                label = tk.Label(self.gallery_frame, image=tk_img, bd=1, relief='solid')
                label.grid(row=0, column=i, padx=2, pady=2)
                
                # Tooltip with filename
                filename = os.path.basename(img_path)
                label.bind("<Enter>", lambda e, f=filename: self.root.title(f"OmniFusion Pro - {f}"))
                label.bind("<Leave>", lambda e: self.root.title("OmniFusion Pro GUI"))
                
            except Exception as e:
                # Show placeholder for failed loads
                label = tk.Label(self.gallery_frame, text="?", width=8, height=4, bg="#fcc", bd=1, relief='solid')
                label.grid(row=0, column=i, padx=2, pady=2)
        
        # Show "more" indicator if needed
        if count > 6:
            more_label = tk.Label(self.gallery_frame, text=f"+{count-6} more", font=("Arial", 8))
            more_label.grid(row=0, column=6, padx=5)
        
    def init_training_tab(self):
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Training (Full Control)")
        
        # Scrollable Frame for Config
        canvas = tk.Canvas(self.train_frame)
        scrollbar = ttk.Scrollbar(self.train_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- SECT 1: Data Paths ---
        data_grp = ttk.LabelFrame(scroll_frame, text="Data Configuration")
        data_grp.pack(fill='x', padx=5, pady=5)
        
        tk.Label(data_grp, text="Image Data Dir:").grid(row=0, column=0, sticky='e')
        self.data_dir_var = tk.StringVar(value=os.path.abspath("Train_Img"))
        tk.Entry(data_grp, textvariable=self.data_dir_var, width=60).grid(row=0, column=1, padx=5)
        tk.Button(data_grp, text="Browse", command=lambda: self.browse_dir(self.data_dir_var)).grid(row=0, column=2)
        
        tk.Label(data_grp, text="Text Data Dirs:").grid(row=1, column=0, sticky='e')
        self.text_dirs_var = tk.StringVar(value=os.path.abspath("text_data/maths") if os.path.exists("text_data/maths") else "")
        tk.Entry(data_grp, textvariable=self.text_dirs_var, width=60).grid(row=1, column=1, padx=5)
        tk.Label(data_grp, text="(comma separated)").grid(row=1, column=2, sticky='w')

        tk.Label(data_grp, text="Latent Cache Dir:").grid(row=2, column=0, sticky='e')
        self.cache_dir_var = tk.StringVar(value="")
        tk.Entry(data_grp, textvariable=self.cache_dir_var, width=60).grid(row=2, column=1, padx=5)
        tk.Button(data_grp, text="Browse", command=lambda: self.browse_dir(self.cache_dir_var)).grid(row=2, column=2)

        # --- SECT 1.25: Caption Verification / Selection (JSON datasets only) ---
        cap_grp = ttk.LabelFrame(scroll_frame, text="Caption Verification (JSON Multi-Caption Datasets)")
        cap_grp.pack(fill='x', padx=5, pady=5)

        tk.Label(
            cap_grp,
            text="These options only affect JSON datasets that contain multiple caption sources (WD / Florence / BLIP / SmolVLM).",
            fg="gray",
        ).grid(row=0, column=0, columnspan=6, sticky="w", padx=5)

        tk.Label(cap_grp, text="Caption Key Override:").grid(row=1, column=0, sticky="e", padx=5)
        self.caption_key_var = tk.StringVar(value="")
        tk.Entry(cap_grp, textvariable=self.caption_key_var, width=40).grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(cap_grp, text="(leave blank to mix sources)", fg="gray").grid(row=1, column=2, sticky="w")

        tk.Label(cap_grp, text="Caption Sampling:").grid(row=2, column=0, sticky="e", padx=5)
        self.caption_sampling_var = tk.StringVar(value="")
        ttk.Combobox(
            cap_grp,
            textvariable=self.caption_sampling_var,
            values=["", "random", "deterministic"],
            width=15,
        ).grid(row=2, column=1, sticky="w", padx=5)
        tk.Label(
            cap_grp,
            text="random: can change per-epoch. deterministic: stable per-image to avoid contradictory supervision.",
            fg="gray",
        ).grid(row=2, column=2, columnspan=4, sticky="w")

        # --- SECT 1.5: Checkpoint Config ---
        ckpt_grp = ttk.LabelFrame(scroll_frame, text="Model Checkpoints")
        ckpt_grp.pack(fill='x', padx=5, pady=5)
        
        tk.Label(ckpt_grp, text="Input Model (Optional):").grid(row=0, column=0, sticky='e')
        self.input_model_var = tk.StringVar()
        tk.Entry(ckpt_grp, textvariable=self.input_model_var, width=50).grid(row=0, column=1, padx=5)
        tk.Button(ckpt_grp, text="Browse", command=lambda: self.browse_file(self.input_model_var)).grid(row=0, column=2)
        
        tk.Label(ckpt_grp, text="Output Name:").grid(row=0, column=3, sticky='e')
        self.output_name_var = tk.StringVar(value="my_model")
        tk.Entry(ckpt_grp, textvariable=self.output_name_var, width=20).grid(row=0, column=4, padx=5)
        
        tk.Label(ckpt_grp, text="Save Every N Steps:").grid(row=1, column=0, sticky='e')
        self.save_every_var = tk.StringVar(value="1000")
        tk.Entry(ckpt_grp, textvariable=self.save_every_var, width=10).grid(row=1, column=1, sticky='w', padx=5)
        tk.Label(ckpt_grp, text="(0 = disable autosave)").grid(row=1, column=2, sticky='w')


        # --- SECT 2: Model Architecture ---
        arch_grp = ttk.LabelFrame(scroll_frame, text="Model Architecture")
        arch_grp.pack(fill='x', padx=5, pady=5)
        
        tk.Label(arch_grp, text="D-Model:").grid(row=0, column=0, sticky='e')
        self.d_model_var = tk.StringVar(value="512")
        tk.Entry(arch_grp, textvariable=self.d_model_var, width=8).grid(row=0, column=1, padx=5)
        
        tk.Label(arch_grp, text="Heads:").grid(row=0, column=2, sticky='e')
        self.heads_var = tk.StringVar(value="8")
        tk.Entry(arch_grp, textvariable=self.heads_var, width=8).grid(row=0, column=3, padx=5)
        
        tk.Label(arch_grp, text="Layers:").grid(row=0, column=4, sticky='e')
        self.layers_var = tk.StringVar(value="8")
        tk.Entry(arch_grp, textvariable=self.layers_var, width=8).grid(row=0, column=5, padx=5)

        # --- SECT 2.5: Text Conditioning (Pooled Injection Controls) ---
        cond_grp = ttk.LabelFrame(scroll_frame, text="Text Conditioning (Pooled Injection Controls)")
        cond_grp.pack(fill='x', padx=5, pady=5)

        tk.Label(cond_grp, text="Text Pooling:").grid(row=0, column=0, sticky="e", padx=5)
        # Default to attention pooling: it starts as uniform pooling (score head is zero-init) but can
        # learn to be more discriminative than mean for long tag lists.
        self.text_pooling_var = tk.StringVar(value="attn")
        ttk.Combobox(cond_grp, textvariable=self.text_pooling_var, values=["mean", "attn"], width=12).grid(
            row=0, column=1, sticky="w", padx=5
        )
        tk.Label(cond_grp, text="mean: average tags. attn: learned pooling (more discriminative).", fg="gray").grid(
            row=0, column=2, columnspan=4, sticky="w"
        )

        tk.Label(cond_grp, text="Pooled Text Scale:").grid(row=1, column=0, sticky="e", padx=5)
        # Default to 0.0 to disable the pooled-text AdaLN shortcut and force token-level conditioning.
        # Requires retraining to change.
        self.pooled_text_scale_var = tk.StringVar(value="0.0")
        tk.Entry(cond_grp, textvariable=self.pooled_text_scale_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(cond_grp, text="(0.0 disables pooled shortcut; requires retraining)", fg="gray").grid(
            row=1, column=2, columnspan=4, sticky="w"
        )

        tk.Label(cond_grp, text="Pooled Text Dropout:").grid(row=2, column=0, sticky="e", padx=5)
        # If pooled scale is enabled (>0), a small dropout helps prevent over-reliance on the pooled path.
        self.pooled_text_dropout_var = tk.StringVar(value="0.2")
        tk.Entry(cond_grp, textvariable=self.pooled_text_dropout_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        tk.Label(cond_grp, text="(per-sample, training only; encourages token-level conditioning)", fg="gray").grid(
            row=2, column=2, columnspan=4, sticky="w"
        )

        # --- SECT 3: Training Loop ---
        train_grp = ttk.LabelFrame(scroll_frame, text="Training Hyperparameters")
        train_grp.pack(fill='x', padx=5, pady=5)
        
        
        tk.Label(train_grp, text="Mode:").grid(row=0, column=0, sticky='e')
        self.duration_mode = tk.StringVar(value="epochs")
        frame_mode = tk.Frame(train_grp)
        frame_mode.grid(row=0, column=1, columnspan=2, sticky='w')
        tk.Radiobutton(frame_mode, text="Epochs", variable=self.duration_mode, value="epochs", command=self.update_duration_label).pack(side='left')
        tk.Radiobutton(frame_mode, text="Steps", variable=self.duration_mode, value="steps", command=self.update_duration_label).pack(side='left')
        
        self.duration_label = tk.Label(train_grp, text="Epochs:")
        self.duration_label.grid(row=0, column=3, sticky='e')
        self.duration_var = tk.StringVar(value="1500")
        tk.Entry(train_grp, textvariable=self.duration_var, width=8).grid(row=0, column=4, padx=5)
        
        
        tk.Label(train_grp, text="Batch Size:").grid(row=1, column=0, sticky='e')
        # Default to 1 for 8GB-class GPUs; use grad-accum to scale effective batch size.
        self.batch_var = tk.StringVar(value="1")
        tk.Entry(train_grp, textvariable=self.batch_var, width=8).grid(row=1, column=1, padx=5)
        
        tk.Label(train_grp, text="Grad Accum:").grid(row=1, column=2, sticky='e')
        self.grad_accum_var = tk.StringVar(value="1")
        tk.Entry(train_grp, textvariable=self.grad_accum_var, width=8).grid(row=1, column=3, padx=5)
        
        tk.Label(train_grp, text="LR:").grid(row=1, column=4, sticky='e')
        self.lr_var = tk.StringVar(value="2e-4")
        tk.Entry(train_grp, textvariable=self.lr_var, width=10).grid(row=1, column=5, padx=5)

        tk.Label(train_grp, text="Weight Decay:").grid(row=1, column=6, sticky='e')
        # Align with common AdamW defaults (e.g., 1e-2) and avoid overly aggressive decay.
        self.weight_decay_var = tk.StringVar(value="0.01")
        tk.Entry(train_grp, textvariable=self.weight_decay_var, width=8).grid(row=1, column=7, padx=5)

        tk.Label(train_grp, text="Warmup:").grid(row=2, column=4, sticky='e')
        self.warmup_var = tk.StringVar(value="1000")
        tk.Entry(train_grp, textvariable=self.warmup_var, width=10).grid(row=2, column=5, padx=5)

        tk.Label(train_grp, text="Max Size:").grid(row=2, column=0, sticky='e')
        self.max_size_var = tk.StringVar(value="256")

        size_frame = tk.Frame(train_grp)
        size_frame.grid(row=2, column=1, sticky='w', padx=5)
        tk.Entry(size_frame, textvariable=self.max_size_var, width=8).pack(side='left')
        tk.Button(size_frame, text="Auto Calc", command=self._auto_calc_max_size, font=("Arial", 8)).pack(side='left', padx=(5, 0))
        
        tk.Label(train_grp, text="Adam β1:").grid(row=3, column=0, sticky='e')
        self.adam_beta1_var = tk.StringVar(value="0.9")
        tk.Entry(train_grp, textvariable=self.adam_beta1_var, width=8).grid(row=3, column=1, padx=5)
        
        tk.Label(train_grp, text="Adam β2:").grid(row=3, column=2, sticky='e')
        self.adam_beta2_var = tk.StringVar(value="0.999")
        tk.Entry(train_grp, textvariable=self.adam_beta2_var, width=8).grid(row=3, column=3, padx=5)
        
        tk.Label(train_grp, text="Adam Eps:").grid(row=3, column=4, sticky='e')
        self.adam_eps_var = tk.StringVar(value="1e-8")
        tk.Entry(train_grp, textvariable=self.adam_eps_var, width=10).grid(row=3, column=5, padx=5)


        # --- SECT 4: Loss Balancing ---
        loss_grp = ttk.LabelFrame(scroll_frame, text="Loss Logic")
        loss_grp.pack(fill='x', padx=5, pady=5)
        
        tk.Label(loss_grp, text="Lambda Img:").grid(row=0, column=0, sticky='e')
        self.lambda_img_var = tk.StringVar(value="1.0")
        tk.Entry(loss_grp, textvariable=self.lambda_img_var, width=8).grid(row=0, column=1, padx=5)
        
        tk.Label(loss_grp, text="Alpha NTP:").grid(row=0, column=2, sticky='e')
        # Default low for image training runs: large alpha makes the LM loss dominate and
        # can cause prompt-conditioning collapse (model learns an unconditional image prior).
        self.alpha_ntp_var = tk.StringVar(value="0.01")
        tk.Entry(loss_grp, textvariable=self.alpha_ntp_var, width=8).grid(row=0, column=3, padx=5)
        
        tk.Label(loss_grp, text="Alpha NTP (TextOnly):").grid(row=0, column=4, sticky='e')
        self.alpha_text_var = tk.StringVar(value="1.0")
        tk.Entry(loss_grp, textvariable=self.alpha_text_var, width=8).grid(row=0, column=5, padx=5)

        # --- SECT 5: Advanced & Optimizations ---
        adv_grp = ttk.LabelFrame(scroll_frame, text="Advanced System Config")
        adv_grp.pack(fill='x', padx=5, pady=5)
        
        tk.Label(adv_grp, text="Workers:").grid(row=0, column=0, sticky='e')
        self.workers_var = tk.StringVar(value="8")
        tk.Entry(adv_grp, textvariable=self.workers_var, width=5).grid(row=0, column=1, padx=5)
        
        tk.Label(adv_grp, text="Graph Batch:").grid(row=0, column=2, sticky='e')
        self.graph_batch_var = tk.StringVar(value="1")
        tk.Entry(adv_grp, textvariable=self.graph_batch_var, width=5).grid(row=0, column=3, padx=5)
        
        tk.Label(adv_grp, text="EMA Every:").grid(row=0, column=4, sticky='e')
        self.ema_every_var = tk.StringVar(value="8")
        tk.Entry(adv_grp, textvariable=self.ema_every_var, width=5).grid(row=0, column=5, padx=5)
        
        tk.Label(adv_grp, text="Prefetch:").grid(row=1, column=0, sticky='e')
        self.prefetch_var = tk.StringVar(value="2")
        tk.Entry(adv_grp, textvariable=self.prefetch_var, width=5).grid(row=1, column=1, padx=5)
        tk.Label(adv_grp, text="(batches to preload)").grid(row=1, column=2, sticky='w')
        
        # Checkboxes
        chk_frame = ttk.Frame(adv_grp)
        chk_frame.grid(row=1, column=0, columnspan=6, pady=5)
        
        self.lazy_load_var = IntVar(value=0)
        self.parallel_enc_var = IntVar(value=1)
        self.use_8bit_var = IntVar(value=1)
        self.use_ema_var = IntVar(value=1)
        # Default OFF: combining Min-SNR with logit-normal sampling can over-weight timesteps.
        self.use_minsnr_var = IntVar(value=0)
        # Default to compiled execution for speed. Disable if you see excessive recompiles.
        self.compile_var = IntVar(value=1)
        self.grad_ckpt_var = IntVar(value=1) # [NEW] Default to True to prevent OOM
        
        # Data Loading Section
        load_frame = ttk.LabelFrame(adv_grp, text="Data Loading Strategy")
        load_frame.grid(row=2, column=0, columnspan=6, pady=5, sticky='ew')
        
        tk.Checkbutton(load_frame, text="Lazy Load (Load on-the-fly, saves RAM)", variable=self.lazy_load_var).grid(row=0, column=0, sticky='w', padx=5)
        tk.Checkbutton(load_frame, text="Parallel Encode (Encode in background, faster start)", variable=self.parallel_enc_var).grid(row=0, column=1, sticky='w', padx=5)
        
        self.use_cache_var = IntVar(value=1)
        tk.Checkbutton(load_frame, text="Use Latent Cache (if available)", variable=self.use_cache_var).grid(row=1, column=0, columnspan=2, sticky='w', padx=5)
        
        # Optimization Checkboxes
        opt_frame = ttk.LabelFrame(adv_grp, text="Optimizations")
        opt_frame.grid(row=3, column=0, columnspan=6, pady=5, sticky='ew')
        
        tk.Checkbutton(opt_frame, text="8-bit Adam (Low VRAM)", variable=self.use_8bit_var).grid(row=0, column=0, sticky='w', padx=5)
        tk.Checkbutton(opt_frame, text="EMA (Smooth weights)", variable=self.use_ema_var).grid(row=0, column=1, sticky='w', padx=5)
        tk.Checkbutton(opt_frame, text="Min-SNR (Loss weighting)", variable=self.use_minsnr_var).grid(row=0, column=2, sticky='w', padx=5)
        tk.Checkbutton(opt_frame, text="Compile (Faster run)", variable=self.compile_var).grid(row=0, column=3, sticky='w', padx=5)
        tk.Checkbutton(opt_frame, text="Grad Checkpoint (-50% VRAM)", variable=self.grad_ckpt_var).grid(row=0, column=4, sticky='w', padx=5)
        
        # --- SECT 6: Finetuning & Control ---
        fine_grp = ttk.LabelFrame(scroll_frame, text="Finetuning & Control")
        fine_grp.pack(fill='x', padx=5, pady=5)
        
        self.freeze_img_var = IntVar(value=0)
        self.freeze_text_var = IntVar(value=0)
        
        tk.Checkbutton(fine_grp, text="Freeze Image Adapters (Keep Memories)", variable=self.freeze_img_var).grid(row=0, column=0, sticky='w')
        tk.Checkbutton(fine_grp, text="Freeze Text Adapters (Keep Logic)", variable=self.freeze_text_var).grid(row=0, column=1, sticky='w')
        
        # --- SECT 6.5: Multi-Image & Context Packing ---
        pack_grp = ttk.LabelFrame(scroll_frame, text="Multi-Image & Context Packing")
        pack_grp.pack(fill='x', padx=5, pady=5)
        
        # Multi-Image Dataset
        self.multi_image_var = IntVar(value=0)
        tk.Checkbutton(pack_grp, text="Enable Multi-Image Training", variable=self.multi_image_var).grid(row=0, column=0, sticky='w', padx=5)
        tk.Label(pack_grp, text="(Multiple images per conversation, 16K context)", fg="gray").grid(row=0, column=1, sticky='w')
        
        tk.Label(pack_grp, text="Multi-Image Data:").grid(row=1, column=0, sticky='e', padx=5)
        self.multi_img_data_var = tk.StringVar(value="")
        tk.Entry(pack_grp, textvariable=self.multi_img_data_var, width=50).grid(row=1, column=1, padx=5)
        tk.Button(pack_grp, text="Browse", command=lambda: self.browse_dir(self.multi_img_data_var)).grid(row=1, column=2)
        
        # Context Packing
        ttk.Separator(pack_grp, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=8)
        
        self.context_pack_var = IntVar(value=0)
        tk.Checkbutton(pack_grp, text="Enable Context Packing", variable=self.context_pack_var).grid(row=3, column=0, sticky='w', padx=5)
        tk.Label(pack_grp, text="(Pack multiple samples into context window for GPU efficiency)", fg="gray").grid(row=3, column=1, sticky='w')
        
        # Context Size Dropdown
        tk.Label(pack_grp, text="Context Size:").grid(row=4, column=0, sticky='e', padx=5)
        self.context_size_var = tk.StringVar(value="16384")
        context_combo = ttk.Combobox(pack_grp, textvariable=self.context_size_var, 
                                     values=["4096", "8192", "16384", "32768", "65536"], width=10) # Removed state="readonly"
        context_combo.grid(row=4, column=1, sticky='w', padx=5)
        
        # Max Context Limit (editable)
        limit_frame = ttk.Frame(pack_grp)
        limit_frame.grid(row=4, column=2, sticky='w', padx=5)
        tk.Label(limit_frame, text="Max Limit:").pack(side='left')
        self.max_context_limit_var = tk.StringVar(value="32768")
        tk.Entry(limit_frame, textvariable=self.max_context_limit_var, width=8).pack(side='left', padx=2)
        tk.Label(limit_frame, text="Fallback:").pack(side='left', padx=(10,0))
        self.fallback_context_var = tk.StringVar(value="16384")
        tk.Entry(limit_frame, textvariable=self.fallback_context_var, width=8).pack(side='left', padx=2)
        tk.Label(limit_frame, text="(auto-applies if OOM)", fg="gray").pack(side='left', padx=5)
        
        # Image/Text Ratio Slider
        tk.Label(pack_grp, text="Image/Text Ratio:").grid(row=5, column=0, sticky='e', padx=5)
        ratio_frame = ttk.Frame(pack_grp)
        ratio_frame.grid(row=5, column=1, columnspan=2, sticky='w', padx=5)
        
        self.image_ratio_var = tk.DoubleVar(value=0.5)
        self.ratio_scale = tk.Scale(ratio_frame, from_=0.0, to=1.0, resolution=0.1, orient='horizontal',
                                    variable=self.image_ratio_var, length=200, command=self._update_ratio_label)
        self.ratio_scale.pack(side='left')
        self.ratio_label = tk.Label(ratio_frame, text="50% img / 50% text", fg="blue")
        self.ratio_label.pack(side='left', padx=10)
        tk.Label(ratio_frame, text="(Fallback: auto-fills if data missing)", fg="gray").pack(side='left')
        
        # Text Chunk Size (for packing calculation)
        tk.Label(pack_grp, text="Avg Sample Size:").grid(row=6, column=0, sticky='e', padx=5)
        chunk_frame = ttk.Frame(pack_grp)
        chunk_frame.grid(row=6, column=1, columnspan=2, sticky='w', padx=5)
        self.text_chunk_var = tk.StringVar(value="512")
        ttk.Combobox(chunk_frame, textvariable=self.text_chunk_var, 
                     values=["256", "512", "1024", "2048", "4096"], width=8).pack(side='left')
        tk.Label(chunk_frame, text="tokens (smaller = more fit per context → fewer steps)", fg="gray").pack(side='left', padx=5)
        
        # Cross-Attention (Pretraining only)
        self.allow_cross_attn_var = IntVar(value=0)
        tk.Checkbutton(pack_grp, text="Allow Cross-Document Attention", variable=self.allow_cross_attn_var).grid(row=7, column=0, sticky='w', padx=5)
        tk.Label(pack_grp, text="⚠️ PRETRAINING ONLY - Unsafe for SFT (disables document isolation)", fg="red").grid(row=7, column=1, sticky='w')


        # --- SECT 7: Calculator Info ---
        calc_grp = ttk.LabelFrame(scroll_frame, text="Training Calculator Info")
        calc_grp.pack(fill='x', padx=5, pady=5)
        
        self.calc_res_label = tk.Label(calc_grp, text="Stats: (Click Calculate below)", fg="blue")
        self.calc_res_label.pack(side='left', padx=10)

        # --- SECT 8: Execution ---
        exec_grp = ttk.LabelFrame(scroll_frame, text="Actions")
        exec_grp.pack(fill='x', padx=5, pady=10)
        
        # Grid layout for buttons
        exec_grp.columnconfigure(0, weight=1)
        exec_grp.columnconfigure(1, weight=2)
        exec_grp.columnconfigure(2, weight=1)

        tk.Button(exec_grp, text="Calculate Stats", command=self.run_calculator).grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        self.train_btn = tk.Button(exec_grp, text="START TRAINING", command=self.run_training, bg="blue", fg="white", font=('Arial', 12, 'bold'))
        self.train_btn.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Stop & Save (Red Button)
        tk.Button(exec_grp, text="STOP & SAVE", command=self.stop_and_save, bg="red", fg="white", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky='ew', padx=5, pady=5)

        
        # Log Output (Placed OUTSIDE scroll frame in lower part of window)
        self.log_text = tk.Text(self.train_frame, height=15)
        self.log_text.pack(side="bottom", fill='both', padx=10, pady=10)

    def browse_dir(self, var):
        path = filedialog.askdirectory()
        if path: var.set(path)
        
    def browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if path: var.set(path)
        
    def update_duration_label(self):
        mode = self.duration_mode.get()
        if mode == "epochs":
            self.duration_label.config(text="Epochs:")
        else:
            self.duration_label.config(text="Total Steps:")

    def run_calculator(self):
        """Quickly counts files and estimates steps with model-accurate token costs."""
        data_dir = self.data_dir_var.get()
        text_dirs = self.text_dirs_var.get()
        
        try:
            batch_size = int(self.batch_var.get())
            grad_accum = int(self.grad_accum_var.get())
            duration_val = int(self.duration_var.get())
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Batch Size, Grad Accum, and Duration must be valid integers.")
            return
        
        mode = self.duration_mode.get()
        
        if batch_size * grad_accum == 0:
             messagebox.showerror("Error", "Batch Size * Grad Accum cannot be zero.")
             return
        
        # Validate Avg Sample Size for context packing
        avg_sample_tokens = 512  # default
        if self.context_pack_var.get() == 1:
            try:
                avg_sample_tokens = int(self.text_chunk_var.get())
                if avg_sample_tokens < 32:
                    messagebox.showerror("Invalid Input", "Avg Sample Size must be >= 32 tokens.")
                    return
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Input", "Avg Sample Size must be a valid integer (>= 32).")
                return
             
        self.calc_res_label.config(text="Counting files... (Please wait)")
        self.root.update_idletasks() # Force UI update
        
        def count_task():
            img_count = 0
            txt_count = 0
            
            # Count Images
            try:
                if os.path.exists(data_dir):
                    with os.scandir(data_dir) as it:
                        for entry in it:
                            if entry.is_file():
                                lower = entry.name.lower()
                                if lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                                    img_count += 1
                                elif lower.endswith(('.txt', '.json')):
                                    img_count += 1  # Text captions alongside images
            except Exception as e:
                print(f"Calc Error (Img): {e}")

            # Count Text (with Extrapolation for large files)
            if text_dirs:
                for d in text_dirs.split(','):
                    d = d.strip()
                    if not os.path.exists(d):
                        print(f"Path not found: {d}")
                        continue
                        
                    files_to_process = []
                    if os.path.isfile(d):
                        files_to_process.append(d)
                    else:
                        for root_d, dirs, files in os.walk(d):
                            for f in files:
                                if f.lower().endswith(('.txt', '.json', '.xml', '.md', '.jsonl', '.csv')):
                                    files_to_process.append(os.path.join(root_d, f))
                                    
                    for full_path in files_to_process:
                        try:
                             size = os.path.getsize(full_path)
                             if size < 1024 * 1024:
                                 with open(full_path, 'r', encoding='utf-8', errors='ignore') as f_read:
                                     file_samples = sum(1 for line in f_read if line.strip())
                             else:
                                 sample_size = 1024 * 1024
                                 with open(full_path, 'r', encoding='utf-8', errors='ignore') as f_read:
                                     chunk = f_read.read(sample_size)
                                     lines_in_chunk = chunk.count('\n')
                                     if lines_in_chunk == 0: lines_in_chunk = 1
                                     avg_bytes = sample_size / lines_in_chunk
                                     file_samples = int(size / avg_bytes)
                             
                             if file_samples == 0: file_samples = 1
                             txt_count += file_samples
                        except Exception as e:
                             print(f"Skipping {full_path}: {e}")
                             txt_count += 1
            
            count = img_count + txt_count
            steps_per_epoch = count // (batch_size * grad_accum)
            if steps_per_epoch == 0: steps_per_epoch = 1
            
            total_steps = 0
            coverage_pct = 0.0
            
            if mode == "epochs":
                total_steps = duration_val * steps_per_epoch
                coverage_pct = duration_val * 100.0
                duration_str = f"{duration_val} Epochs"
            else:
                total_steps = duration_val
                coverage_pct = (duration_val / steps_per_epoch) * 100.0
                duration_str = f"{duration_val} Steps"

            # Context packing estimate (model-accurate token costs)
            context_pack_str = ""
            if self.context_pack_var.get() == 1:
                context_size = int(self.context_size_var.get())
                img_ratio = self.image_ratio_var.get()
                
                # Calculate exact image token cost using model constants
                vae_ds, patch_sz = 8, 2
                block = vae_ds * patch_sz  # 16px alignment block
                try:
                    max_img_size = int(self.max_size_var.get())
                except (ValueError, TypeError):
                    max_img_size = 256
                
                # Align max_image_size to block grid (same as data_manager.py)
                aligned_img_size = max(block, ((max_img_size + block - 1) // block) * block)
                img_token_cost = (aligned_img_size // vae_ds // patch_sz) ** 2  # Square image assumption
                
                # Weighted average token cost per sample
                # Image samples cost: image_tokens + text_tokens (caption)
                # Text samples cost: text_tokens only
                img_sample_cost = img_token_cost + avg_sample_tokens  # Image patches + caption text
                txt_sample_cost = avg_sample_tokens                   # Text tokens only
                avg_cost_per_sample = img_ratio * img_sample_cost + (1.0 - img_ratio) * txt_sample_cost
                avg_cost_per_sample = max(32, avg_cost_per_sample)  # Floor
                
                samples_per_context = max(1, int(context_size / avg_cost_per_sample))
                packed_contexts = max(1, count // samples_per_context)
                packed_steps = max(1, packed_contexts // (batch_size * grad_accum))
                
                context_pack_str = (f" | 📦 ~{packed_contexts:,} contexts (~{packed_steps:,} steps) "
                                    f"@ ~{avg_cost_per_sample:.0f}tok/sample "
                                    f"(img:{img_token_cost}tok @{max_img_size}px) "
                                    f"| {int(img_ratio*100)}%img")
                
                # Use packed steps for total calculation
                if mode == "epochs":
                    total_steps = duration_val * packed_steps
                    steps_per_epoch = packed_steps
                else:
                    steps_per_epoch = packed_steps

            res_text = (f"Files: {count:,} (img:{img_count:,} txt:{txt_count:,}) | Steps/Epoch: {steps_per_epoch:,} | "
                        f"Target: {duration_str} = {total_steps:,} Steps | "
                        f"Data Cov: {coverage_pct:.1f}%{context_pack_str}")
                        
            self.root.after(0, lambda: self.calc_res_label.config(text=res_text))
            
        threading.Thread(target=count_task, daemon=True).start()

    def stop_and_save(self):
        """Creates the stop signal file."""
        if messagebox.askyesno("Stop Training", "Are you sure you want to trigger a stop & save?"):
            with open("stop_training.signal", "w") as f:
                f.write("STOP")
            messagebox.showinfo("Signal Sent", "Stop signal sent. Training will save and exit at the next log step.")

    def _browse_inf_model(self):
        """Browse for inference model file."""
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if path:
            self.inf_model_var.set(path)

    def _load_model_persistent(self):
        """Loads inference model into VRAM natively to avoid subprocess startup penalties."""
        inf_model = self.inf_model_var.get().strip() or self.input_model_var.get().strip()
        if not inf_model:
            messagebox.showerror("Error", "Please specify a checkpoint path first.")
            return
            
        self.load_btn.config(state='disabled', text="Loading...")
        self.engine_status_label.config(text="Status: Loading to VRAM...", fg="orange")
        self.root.update_idletasks()
        
        def load_task():
            try:
                import torch

                # Force Hugging Face Hub offline so loading the cached Flux VAE does not make network requests.
                # If the VAE is not already present in the local HF cache, model load will fail (by design).
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                from inference_backend import InferenceModel
                
                with self.engine_lock:
                    if self.engine is not None:
                         # Unload old one first
                         del self.engine
                         torch.cuda.empty_cache()
                         import gc
                         gc.collect()
                         
                    self.engine = InferenceModel(inf_model)
                    
                self.root.after(0, lambda: self.engine_status_label.config(text="Status: Loaded (VRAM Active)", fg="green"))
                self.root.after(0, lambda: self.load_btn.config(state='normal', text="Reload Model"))
                self.root.after(0, lambda: self.unload_btn.config(state='normal'))
                self.root.after(0, lambda: messagebox.showinfo("Loaded", f"Model {os.path.basename(inf_model)} loaded successfully!"))
            except Exception as e:
                self.root.after(0, lambda: self.engine_status_label.config(text="Status: Error loading", fg="red"))
                self.root.after(0, lambda: self.load_btn.config(state='normal', text="Load Model"))
                self.root.after(0, lambda: messagebox.showerror("Load Error", str(e)))

        threading.Thread(target=load_task, daemon=True).start()

    def _unload_model_persistent(self):
        """Frees VRAM."""
        with self.engine_lock:
            if self.engine is not None:
                del self.engine
                self.engine = None
                
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                
        self.engine_status_label.config(text="Status: Unloaded", fg="gray")
        self.unload_btn.config(state='disabled')
        self.load_btn.config(text="Load Model")
        messagebox.showinfo("Unloaded", "Model cleared from VRAM.")

    def run_generation(self):
        prompt = self.prompt_entry.get("1.0", "end-1c").strip()
        seed = int(self.seed_var.get())
        w = int(self.width_var.get())
        h = int(self.height_var.get())
        steps = int(self.steps_var.get())
        cfg = float(self.cfg_var.get())
        
        if not self.engine:
            if not messagebox.askyesno("Load Model?", "Model is not loaded into memory. Load it now? (Takes ~15s)"):
                return
            self._load_model_persistent()
            return # Wait for user to trigger again once loaded
            
        self.gen_btn.config(state='disabled', text="Generating...")
        self.img_label.config(text="Generating... Please wait.", image="")
        
        def thread_target():
            import torch
            try:
                with self.engine_lock:
                    torch.manual_seed(seed)
                    generator = self.engine.generate_image(prompt, OUTPUT_IMAGE, steps=steps, cfg=cfg, width=w, height=h)
                    
                    for update in generator:
                        if update["type"] == "image_preview":
                            # Live Update GUI Preview
                            pil_img = update["image"]
                            step = update["step"]
                            tot = update["total_steps"]
                            
                            def update_ui_preview(img=pil_img, s=step, t=tot):
                                preview_img = img.copy()
                                preview_img.thumbnail((200, 200)) 
                                self.tk_preview = ImageTk.PhotoImage(preview_img)
                                self.img_label.config(image=self.tk_preview, text=f"Step {s}/{t}")
                            self.root.after(0, update_ui_preview)
                            
                        elif update["type"] == "final_image":
                            # Final Save
                            self.root.after(0, self.update_image)
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = str(e) or repr(e)
                self.root.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(state='normal', text="GEN IMAGE"))

        threading.Thread(target=thread_target, daemon=True).start()
        
    def run_text_generation(self):
        user_msg = self.prompt_entry.get("1.0", "end-1c").strip()
        if not user_msg:
             return
             
        # UI Update: Show User Message
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', f"User: {user_msg}\n\n", "user")
        self.chat_display.config(state='disabled')
        self.prompt_entry.delete('1.0', 'end')
        
        # 1. Update History
        self.chat_history.append({"role": "user", "content": user_msg})
        
        # 2. Construct Full Context Prompt
        # Format: <|im_start|>system ... <|im_end|> ... <|im_start|>user ...
        sys_prompt = self.sys_prompt_text.get("1.0", "end-1c").strip()
        full_prompt = f"<|im_start|>system\n{sys_prompt}\n<|im_end|>\n"
        
        for msg in self.chat_history:
             full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}\n<|im_end|>\n"
             
        # Add assistant generation start
        full_prompt += "<|im_start|>assistant\n"
        
        seed = int(self.seed_var.get())
        max_tokens = int(self.max_tokens_var.get())
        temperature = float(self.temperature_var.get())
        top_k = int(self.top_k_var.get())
        top_p = float(self.top_p_var.get())
        min_p = float(self.min_p_var.get())
        rep_pen = float(self.rep_penalty_var.get())
        
        if not self.engine:
            if not messagebox.askyesno("Load Model?", "Model is not loaded into memory. Load it now? (Takes ~15s)"):
                return
            self._load_model_persistent()
            return
        
        self.text_btn.config(state='disabled', text="Generating...")
        
        # Extract images if provided
        img_paths = self.selected_images.copy() if self.selected_images else None
        
        # Prepare UI for incoming text stream
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', "Assistant: ", "assistant")
        
        # We need a tracker to find where the assistant text starts
        # so we can style it or update it dynamically
        start_index = self.chat_display.index("end-1c")
        self.chat_display.see('end')
        self.chat_display.config(state='disabled')
        
        def thread_target():
            import torch
            try:
                with self.engine_lock:
                    torch.manual_seed(seed)
                    
                    if img_paths:
                        generator = self.engine.generate_multimodal_with_images(
                            full_prompt, image_paths=img_paths, max_new_tokens=max_tokens, 
                            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=rep_pen
                        )
                    else:
                        generator = self.engine.generate_text_completion(
                            full_prompt, max_new_tokens=max_tokens, 
                            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=rep_pen
                        )
                        
                    full_response = ""
                    
                    for update in generator:
                        if update["type"] == "token":
                            chunk = update["text"]
                            full_response += chunk
                            
                            def append_chunk(c=chunk):
                                self.chat_display.config(state='normal')
                                self.chat_display.insert('end', c, "assistant")
                                self.chat_display.see('end')
                                self.chat_display.config(state='disabled')
                            self.root.after(0, append_chunk)
                            
                        elif update["type"] == "final_text":
                            # Stream finishes, add newlines
                            def finish_stream(tt=full_response, met=update.get("metrics", "")):
                                self.chat_display.config(state='normal')
                                self.chat_display.insert('end', f"\n\n[Finished {met}]\n\n", "system")
                                self.chat_display.see('end')
                                self.chat_display.config(state='disabled')
                                
                                # Update chat history
                                self.chat_history.append({"role": "assistant", "content": tt})
                            self.root.after(0, finish_stream)
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = str(e) or repr(e)
                self.root.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
            finally:
                self.root.after(0, lambda: self.text_btn.config(state='normal', text="SEND TEXT"))

        threading.Thread(target=thread_target, daemon=True).start()
        
    def _auto_calc_max_size(self):
        text_dir = self.text_dirs_var.get().strip()
        if not text_dir or not os.path.exists(text_dir):
            messagebox.showerror("Error", "Please select a valid Text Data Dir in the Data Loading Strategy section first.")
            return
            
        try:
            from data_manager import TiktokenTokenizer
            import json
            import math
            import os
            tokenizer = TiktokenTokenizer()
            
            jsonl_files = [f for f in os.listdir(text_dir) if f.endswith('.jsonl')]
            if not jsonl_files:
                messagebox.showerror("Error", "No .jsonl files found in the selected directory.")
                return
                
            filepath = os.path.join(text_dir, jsonl_files[0])
            max_tokens = 0
            lines_checked = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if lines_checked >= 20: 
                        break
                    try:
                        data = json.loads(line)
                        texts = []
                        if "messages" in data:
                            texts = [m.get("content", "") for m in data["messages"]]
                        elif "text" in data:
                            texts = [data["text"]]
                            
                        if texts:
                            full_text = " ".join(texts)
                            tokens = tokenizer.encode(full_text)
                            max_tokens = max(max_tokens, len(tokens))
                            lines_checked += 1
                    except json.JSONDecodeError:
                        continue
                        
            if max_tokens > 0:
                # Add a small buffer and round up to the nearest power of 2 (or a standard limit)
                padded_tokens = max_tokens + 16 
                target = max(32, 2 ** math.ceil(math.log2(padded_tokens)))
                
                self.max_size_var.set(str(target))
                messagebox.showinfo("Auto Size", f"Scanned {lines_checked} samples from {jsonl_files[0]}.\n\nMax tokens found: {max_tokens}\nOptimum Max Size set to: {target}")
            else:
                messagebox.showwarning("Warning", "Could not parse texts/messages from the first 20 lines to count tokens.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-calculate sample size: {e}")
            

    def update_image(self):
        if os.path.exists(OUTPUT_IMAGE):
            try:
                # 1. Load Image
                pil_img = Image.open(OUTPUT_IMAGE)
                
                # 2. Resize for Sidebar Preview (Small)
                preview_img = pil_img.copy()
                preview_img.thumbnail((200, 200)) 
                self.tk_preview = ImageTk.PhotoImage(preview_img)
                self.img_label.config(image=self.tk_preview, text="")
                
                # 3. Insert into Chat History (Medium/Large)
                chat_img = pil_img.copy()
                chat_img.thumbnail((500, 500))
                tk_chat_img = ImageTk.PhotoImage(chat_img)
                
                # Robust Reference Keeping
                if not hasattr(self, 'image_refs'):
                    self.image_refs = []
                self.image_refs.append(tk_chat_img) # Keep ref alive!
                
                self.chat_display.config(state='normal')
                self.chat_display.insert('end', "\nAssistant generated an image:\n", "assistant")
                self.chat_display.image_create('end', image=tk_chat_img)
                self.chat_display.insert('end', "\n\n")
                self.chat_display.see('end')
                self.chat_display.config(state='disabled')
                
                # Append to history logic (conceptually)
                self.chat_history.append({"role": "assistant", "content": "[IMAGE GENERATED]"})
                
            except Exception as e:
                 self.img_label.config(text=f"Error loading image: {e}")
        else:
            self.img_label.config(text="Generation Failed (No Output)")

    def run_training(self):
        # VCVARS Detection
        potential_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        ]
        vcvars_path = None
        for p in potential_paths:
            if os.path.exists(p):
                vcvars_path = p
                break
        
        # Build Command
        cmd_args = [PYTHON_EXE, TRAINING_SCRIPT]
        
        # [FIX] Only add --data-dir if value is NOT empty.
        # This prevents passing an empty flag which causes the "expected one argument" error.
        data_dir_val = self.data_dir_var.get().strip()
        if data_dir_val:
            cmd_args.extend(["--data-dir", data_dir_val])
        
        # Add text-data-dirs only if not empty
        text_dirs = self.text_dirs_var.get().strip()
        if text_dirs:
            cmd_args.extend(["--text-data-dirs", text_dirs])

        # Caption verification / selection (JSON multi-caption datasets only)
        if hasattr(self, "caption_key_var"):
            cap_key = self.caption_key_var.get().strip()
            if cap_key:
                cmd_args.extend(["--caption-key", cap_key])

        if hasattr(self, "caption_sampling_var"):
            cap_sampling = self.caption_sampling_var.get().strip()
            if cap_sampling:
                cmd_args.extend(["--caption-sampling", cap_sampling])
            
        # Add latency cache configuration
        cache_dir = self.cache_dir_var.get().strip()
        if cache_dir:
            cmd_args.extend(["--cache-dir", cache_dir])
        if self.use_cache_var.get() == 0:
            cmd_args.append("--no-cache")
            
        cmd_args.extend([
            "--batch-size", self.batch_var.get(),
            "--lr", self.lr_var.get(),
            "--weight-decay", self.weight_decay_var.get(),
            "--adam-beta1", self.adam_beta1_var.get(),
            "--adam-beta2", self.adam_beta2_var.get(),
            "--adam-eps", self.adam_eps_var.get(),
            "--max-size", self.max_size_var.get(),
            "--grad-accum-steps", self.grad_accum_var.get(),
            # Model Arch
            "--d-model", self.d_model_var.get(),
            "--n-heads", self.heads_var.get(),
            "--n-layers", self.layers_var.get(),
            "--lr", self.lr_var.get(),
            "--warmup-steps", self.warmup_var.get(),
        ])

        # Text conditioning controls (pooled injection)
        if hasattr(self, "text_pooling_var"):
            cmd_args.extend(["--text-pooling", self.text_pooling_var.get().strip()])
        if hasattr(self, "pooled_text_scale_var"):
            cmd_args.extend(["--pooled-text-scale", self.pooled_text_scale_var.get().strip()])
        if hasattr(self, "pooled_text_dropout_var"):
            cmd_args.extend(["--pooled-text-dropout", self.pooled_text_dropout_var.get().strip()])
        
        # Model I/O (Optional Input, Required Output)
        if self.input_model_var.get().strip():
            cmd_args.extend(["--input-model", self.input_model_var.get().strip()])
            
        if self.output_name_var.get().strip():
             cmd_args.extend(["--output-name", self.output_name_var.get().strip()])
        
        # Save Every N Steps
        save_every = self.save_every_var.get().strip()
        if save_every and int(save_every) > 0:
             cmd_args.extend(["--save-every", save_every])

        
        # Handle Duration Logic

        mode = self.duration_mode.get()
        val = self.duration_var.get()
        
        if mode == "epochs":
            cmd_args.extend(["--epochs", val])
            cmd_args.extend(["--max-steps", "0"]) # Disable max steps limit
        else:
            cmd_args.extend(["--max-steps", val])
            cmd_args.extend(["--epochs", "99999"]) # Ensure loop continues until key interrupt or max steps
            
        cmd_args.extend([
            # Loss Balancing
            "--lambda-img", self.lambda_img_var.get(),
            "--alpha-ntp", self.alpha_ntp_var.get(),
            "--alpha-ntp-text-only", self.alpha_text_var.get(),
            # Advanced
            "--workers", self.workers_var.get(),
            "--graph-batch-size", self.graph_batch_var.get(),
            "--ema-every", self.ema_every_var.get(),
            "--prefetch-factor", self.prefetch_var.get(),
            "--log-every", "10",
        ])
        
        # Boolean Flags (Default Enabled in script means --no-X disables it)
        if self.use_8bit_var.get() == 0: cmd_args.append("--no-8bit")
        if self.use_ema_var.get() == 0: cmd_args.append("--no-ema")
        if self.compile_var.get() == 0: cmd_args.append("--no-compile")
        
        # Boolean Flags (Default Disabled in script means --X enables it)
        if self.lazy_load_var.get() == 1: cmd_args.append("--lazy-load")
        if self.parallel_enc_var.get() == 1: cmd_args.append("--parallel-encode")
        if self.use_minsnr_var.get() == 1: cmd_args.append("--use-min-snr")
        if self.freeze_img_var.get() == 1: cmd_args.append("--freeze-img")
        if self.freeze_text_var.get() == 1: cmd_args.append("--freeze-text")
        if self.grad_ckpt_var.get() == 1: cmd_args.append("--grad-checkpointing")
        
        # Multi-Image & Context Packing Options
        if self.multi_image_var.get() == 1:
            cmd_args.append("--multi-image")
            multi_img_data = self.multi_img_data_var.get().strip()
            if multi_img_data:
                cmd_args.extend(["--multi-image-data", multi_img_data])
        
        if self.context_pack_var.get() == 1:
            cmd_args.append("--context-pack")
            
            # Auto-fallback: if selected context > max limit, use fallback
            selected_context = int(self.context_size_var.get())
            max_limit = int(self.max_context_limit_var.get())
            fallback = int(self.fallback_context_var.get())
            
            if selected_context > max_limit:
                actual_context = fallback
                self.log_text.insert('end', f"⚠️ Context {selected_context} > Max {max_limit}, using fallback: {fallback}\n")
            else:
                actual_context = selected_context
            
            cmd_args.extend(["--max-context-length", str(actual_context)])
            cmd_args.extend(["--image-ratio", str(self.image_ratio_var.get())])
            
            if self.allow_cross_attn_var.get() == 1:
                cmd_args.append("--allow-cross-attention")


        # CMD Construction
        if vcvars_path:
            # [FIX] Add -u for unbuffered Python output so logs stream in real-time
            python_cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd_args])
            # Insert -u after python.exe for unbuffered output
            python_cmd_str = python_cmd_str.replace(PYTHON_EXE, f'{PYTHON_EXE} -u', 1)
            cmd = f'"{vcvars_path}" && {python_cmd_str}'
            use_shell = True
            msg = f"Found VCVARS at: {vcvars_path}\nStarting Training...\nCommand: {python_cmd_str}\n"
        else:
            # [FIX] Insert -u flag for unbuffered output
            cmd_args.insert(1, "-u")
            cmd = cmd_args
            use_shell = False
            msg = f"VCVARS not found. Running with default environment...\nCommand: {cmd}\n"
        
        self.train_btn.config(state='disabled', text="Training Running...")
        self.log_text.delete('1.0', 'end')
        self.log_text.insert('end', msg)
        
        # [FIX] Set environment variable for unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Force PyTorch to use expandable segments for CUDA allocations to handle Windows fragmentation
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        
        def read_output(process):
            """Read subprocess output and display in log widget"""
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Schedule UI update on main thread
                        self.root.after(0, lambda l=line: self.log_text.insert('end', l))
                        self.root.after(0, lambda: self.log_text.see('end'))
            except Exception as e:
                self.root.after(0, lambda: self.log_text.insert('end', f'\n[LOG ERROR: {e}]\n'))
            finally:
                try:
                    process.stdout.close()
                except:
                    pass
            
        def thread_target():
            process = None
            try:
                # [FIX] Added env=env for PYTHONUNBUFFERED
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                          text=True, bufsize=1, shell=use_shell, env=env)
                read_output(process)
                exit_code = process.wait()
                if exit_code == 0:
                    self.root.after(0, lambda: messagebox.showinfo("Done", "Training script finished successfully!"))
                else:
                    self.root.after(0, lambda: messagebox.showwarning("Stopped", f"Training ended with exit code {exit_code}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.train_btn.config(state='normal', text="START TRAINING"))
                # Clean up stop signal file
                if os.path.exists("stop_training.signal"):
                    try:
                        os.remove("stop_training.signal")
                    except:
                        pass

        threading.Thread(target=thread_target, daemon=True).start()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"GUI Error: {e}")
        input("Press Enter to exit...")
