import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = r"C:\Users\chatr\Documents\Tech\VLLM\venv\Scripts\python.exe"
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

SCRIPT_METADATA = os.path.join(ROOT_DIR, "precompute_metadata.py")
SCRIPT_LATENTS = os.path.join(ROOT_DIR, "precompute_latents.py")
SCRIPT_MAIN_GUI = os.path.join(ROOT_DIR, "gui_app.py")


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OmniForge X1 - Single Tab Launcher")
        self.root.geometry("1040x860")

        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        self._init_vars()
        self._build_main(outer)

    def _init_vars(self):
        # Shared/Image location
        self.image_dir = tk.StringVar(value="")

        # Metadata
        self.md_output = tk.StringVar(value="")
        self.md_workers = tk.StringVar(value="8")

        # Latent/image settings
        self.lt_cache_dir = tk.StringVar(value=os.path.join(ROOT_DIR, ".latent_cache"))
        self.lt_batch_size = tk.StringVar(value="32")
        self.lt_max_size = tk.StringVar(value="256")
        self.lt_keep_ar = tk.IntVar(value=1)
        self.lt_workers = tk.StringVar(value="8")
        self.lt_prefetch = tk.StringVar(value="4")
        self.lt_compile = tk.IntVar(value=1)
        self.lt_compile_mode = tk.StringVar(value="reduce-overhead")
        self.lt_cuda_graphs = tk.IntVar(value=1)
        self.lt_graph_bs = tk.StringVar(value="64")
        self.lt_shard_gb = tk.StringVar(value="2.0")
        self.lt_flush_every = tk.StringVar(value="5000")

        # Text location/settings
        self.text_location = tk.StringVar(value="Sidecar text/json next to images (auto-detected, cached unpadded)")
        self.lt_text_threads = tk.StringVar(value="8")
        self.lt_max_text_length = tk.StringVar(value="512")
        self.lt_token_dtype = tk.StringVar(value="int32")

        # Latent dtype
        self.lt_latent_dtype = tk.StringVar(value="float16")

    def _build_main(self, parent):
        self._build_image_section(parent)
        self._build_text_section(parent)
        self._build_metadata_section(parent)
        self._build_latent_section(parent)
        self._build_tools_section(parent)
        self._build_log(parent)

    def _build_image_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Image Dir / Image Settings")
        frame.grid(row=0, column=0, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Image Dir").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.image_dir).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="Browse", command=lambda: self._browse_dir(self.image_dir)).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(frame, text="Max Size").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_max_size, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Batch Size").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_batch_size, width=12).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Checkbutton(frame, text="Keep Aspect Ratio (letterbox pad)", variable=self.lt_keep_ar).grid(row=2, column=1, sticky="w", padx=4, pady=4)

    def _build_text_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Text Location / Text Settings")
        frame.grid(row=1, column=0, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Text Location").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.text_location, state="readonly").grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, text="Text Threads").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_text_threads, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Token Dtype").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, textvariable=self.lt_token_dtype, values=["int32", "int64"], state="readonly", width=10).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Max Text Len").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_max_text_length, width=12).grid(row=2, column=1, sticky="w", padx=4, pady=4)

    def _build_metadata_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Metadata Precompute Args (all)")
        frame.grid(row=2, column=0, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Output JSON").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.md_output).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_metadata_output).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(frame, text="Workers").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.md_workers, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Button(frame, text="Run precompute_metadata.py", command=self.run_metadata).grid(row=1, column=2, padx=4, pady=4)

    def _build_latent_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Latent Precompute Args (all)")
        frame.grid(row=3, column=0, sticky="ew", pady=4)

        ttk.Label(frame, text="Cache Dir").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_cache_dir, width=54).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(frame, text="Browse", command=lambda: self._browse_dir(self.lt_cache_dir)).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(frame, text="Workers").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_workers, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Prefetch").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_prefetch, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Compile Mode").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, textvariable=self.lt_compile_mode, values=["default", "reduce-overhead", "max-autotune"], state="readonly", width=18).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(frame, text="--compile", variable=self.lt_compile).grid(row=2, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Graph Batch").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_graph_bs, width=8).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(frame, text="--use-cuda-graphs", variable=self.lt_cuda_graphs).grid(row=3, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Latent Dtype").grid(row=4, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, textvariable=self.lt_latent_dtype, values=["float16", "float32"], state="readonly", width=10).grid(row=4, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Token Dtype").grid(row=4, column=2, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, textvariable=self.lt_token_dtype, values=["int32", "int64"], state="readonly", width=10).grid(row=4, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Shard Size GB").grid(row=5, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_shard_gb, width=10).grid(row=5, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Flush Every").grid(row=5, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.lt_flush_every, width=10).grid(row=5, column=3, sticky="w", padx=4, pady=4)

        ttk.Button(frame, text="Run precompute_latents.py", command=self.run_latents).grid(row=6, column=2, padx=4, pady=6)

    def _build_tools_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Tools")
        frame.grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(frame, text="Launch Main GUI (gui_app.py)", command=self.run_main_gui).pack(side="left", padx=6, pady=6)

    def _build_log(self, parent):
        frame = ttk.LabelFrame(parent, text="Logs")
        frame.grid(row=5, column=0, sticky="nsew", pady=4)
        parent.rowconfigure(5, weight=1)
        parent.columnconfigure(0, weight=1)
        self.log_text = tk.Text(frame, height=16, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _browse_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _browse_metadata_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="dataset_metadata.json",
        )
        if p:
            if not p.lower().endswith(".json"):
                p = os.path.splitext(p)[0] + ".json"
            self.md_output.set(p)

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _run_command(self, cmd):
        def worker():
            self.root.after(0, lambda: self._append_log("Running:\n" + " ".join(cmd) + "\n\n"))
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        self.root.after(0, lambda l=line: self._append_log(l))
                proc.wait()
                self.root.after(0, lambda: self._append_log(f"\nExit code: {proc.returncode}\n"))
            except Exception as e:
                self.root.after(0, lambda: self._append_log(f"\nFailed: {e}\n"))

        threading.Thread(target=worker, daemon=True).start()

    def run_metadata(self):
        image_dir = self.image_dir.get().strip()
        if not image_dir:
            self._append_log("Error: Image Dir is required for metadata.\n")
            return

        cmd = [PYTHON_EXE, SCRIPT_METADATA, "--data-dir", image_dir]

        output = self.md_output.get().strip()
        if output:
            if not output.lower().endswith(".json"):
                output = os.path.splitext(output)[0] + ".json"
            cmd.extend(["--output", output])

        workers = self.md_workers.get().strip()
        if workers:
            cmd.extend(["--workers", workers])

        self._run_command(cmd)

    def run_latents(self):
        image_dir = self.image_dir.get().strip()
        if not image_dir:
            self._append_log("Error: Image Dir is required for latent precompute.\n")
            return

        cmd = [
            PYTHON_EXE,
            SCRIPT_LATENTS,
            "--data-dir",
            image_dir,
            "--cache-dir",
            self.lt_cache_dir.get().strip(),
            "--batch-size",
            self.lt_batch_size.get().strip(),
            "--max-size",
            self.lt_max_size.get().strip(),
            "--max-text-length",
            self.lt_max_text_length.get().strip(),
            "--workers",
            self.lt_workers.get().strip(),
            "--prefetch-factor",
            self.lt_prefetch.get().strip(),
            "--text-threads",
            self.lt_text_threads.get().strip(),
            "--compile-mode",
            self.lt_compile_mode.get().strip(),
            "--graph-batch-size",
            self.lt_graph_bs.get().strip(),
            "--latent-dtype",
            self.lt_latent_dtype.get().strip(),
            "--token-dtype",
            self.lt_token_dtype.get().strip(),
            "--shard-size-gb",
            self.lt_shard_gb.get().strip(),
            "--flush-every",
            self.lt_flush_every.get().strip(),
        ]

        if self.lt_keep_ar.get() == 1:
            cmd.append("--keep-aspect-ratio")
        else:
            cmd.append("--no-keep-aspect-ratio")

        if self.lt_compile.get() == 1:
            cmd.append("--compile")
        if self.lt_cuda_graphs.get() == 1:
            cmd.append("--use-cuda-graphs")

        self._run_command(cmd)

    def run_main_gui(self):
        self._run_command([PYTHON_EXE, SCRIPT_MAIN_GUI])


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
