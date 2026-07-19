import os
import random
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageReviewer(tk.Tk):
    def __init__(
        self,
        downloads_dir="data/downloads",
        keep_dir="dataset/raw",
        reject_dir="data/rejected",
        shuffle=True,
    ):
        super().__init__()

        self.downloads_dir = Path(downloads_dir)
        self.keep_dir = Path(keep_dir)
        self.reject_dir = Path(reject_dir)

        self.keep_dir.mkdir(parents=True, exist_ok=True)
        self.reject_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        self.title("Dataset Reviewer (Keep / Reject)")
        self.geometry("1100x800")
        self.minsize(900, 650)

        self.files = self._scan_files()
        if shuffle:
            random.shuffle(self.files)

        self.idx = 0
        self.current_path: Path | None = None
        self.current_img = None
        self.current_tk = None
        self.last_action_stack = []


        self._build_ui()


        self.bind("<Right>", lambda e: self.keep())
        self.bind("<Left>", lambda e: self.reject())
        self.bind("<space>", lambda e: self.skip())
        self.bind("<BackSpace>", lambda e: self.undo())
        self.bind("<Escape>", lambda e: self.quit_app())

        self.bind("k", lambda e: self.keep())
        self.bind("j", lambda e: self.reject())
        self.bind("s", lambda e: self.skip())
        self.bind("u", lambda e: self.undo())

        self.bind("<Configure>", lambda e: self._render_current())


        if not self.files:
            messagebox.showinfo("No images", f"No images found in {self.downloads_dir}")
            self.destroy()
            return

        self._load_current()

    def _build_ui(self):

        top = tk.Frame(self, padx=10, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.counter_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        tk.Label(top, textvariable=self.counter_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        tk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=15)
        tk.Label(top, textvariable=self.path_var, font=("Consolas", 10), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)


        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        bottom = tk.Frame(self, padx=10, pady=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        btn_keep = tk.Button(bottom, text="KEEP  (K / →)", command=self.keep, width=18, height=2)
        btn_rej = tk.Button(bottom, text="REJECT (J / ←)", command=self.reject, width=18, height=2)
        btn_skip = tk.Button(bottom, text="SKIP  (Space)", command=self.skip, width=18, height=2)
        btn_undo = tk.Button(bottom, text="UNDO  (Backspace)", command=self.undo, width=18, height=2)

        btn_keep.pack(side=tk.LEFT, padx=6)
        btn_rej.pack(side=tk.LEFT, padx=6)
        btn_skip.pack(side=tk.LEFT, padx=6)
        btn_undo.pack(side=tk.LEFT, padx=6)

        help_txt = (
            "Keys: K/→ keep | J/← reject | Space skip | Backspace undo | Esc quit\n"
            "Tip: keep your left hand on J/K like a pro."
        )
        tk.Label(bottom, text=help_txt, justify="left", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

    def _scan_files(self):
        files = []
        for p in self.downloads_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
        return files

    def _load_current(self):
        if self.idx >= len(self.files):
            self._finish()
            return

        self.current_path = self.files[self.idx]
        self.path_var.set(str(self.current_path))
        self.counter_var.set(f"{self.idx + 1}/{len(self.files)}")
        self.status_var.set("")

        try:
            img = Image.open(self.current_path)
            img = ImageOps.exif_transpose(img)
            self.current_img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        except Exception as e:
            self.status_var.set(f"Failed to open image: {e}")
            self.current_img = None

        self._render_current()

    def _render_current(self):
        self.canvas.delete("all")
        if self.current_img is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="(Could not load image)",
                fill="white",
                font=("Segoe UI", 16, "bold"),
            )
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())


        img = self.current_img
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.current_tk = ImageTk.PhotoImage(resized)
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.current_tk)

    def _safe_move(self, src: Path, dst_dir: Path) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name


        if dst.exists():
            stem = src.stem
            ext = src.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}__{i}{ext}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.move(str(src), str(dst))
        return dst

    def keep(self):
        self._move_current(self.keep_dir, "KEPT")

    def reject(self):
        self._move_current(self.reject_dir, "REJECTED")

    def skip(self):
        self.status_var.set("SKIPPED")
        self.idx += 1
        self._load_current()

    def undo(self):
        if not self.last_action_stack:
            self.status_var.set("Nothing to undo")
            return

        src, dst = self.last_action_stack.pop()

        try:
            restored = self._safe_move(Path(dst), self.downloads_dir)
        except Exception as e:
            self.status_var.set(f"Undo failed: {e}")
            return


        self.files.insert(self.idx, restored)
        self.status_var.set("UNDO OK")


        self._load_current()

    def _move_current(self, target_dir: Path, label: str):
        if self.current_path is None:
            return

        src = self.current_path
        try:
            dst = self._safe_move(src, target_dir)
        except Exception as e:
            self.status_var.set(f"Move failed: {e}")
            return

        self.last_action_stack.append((str(src), str(dst)))
        self.status_var.set(label)

        self.idx += 1
        self._load_current()

    def _finish(self):
        msg = (
            "No more images in data/downloads.\n\n"
            f"Kept: {len(list(self.keep_dir.glob('*')))} files in {self.keep_dir}\n"
            f"Rejected: {len(list(self.reject_dir.glob('*')))} files in {self.reject_dir}\n"
        )
        messagebox.showinfo("Done", msg)
        self.destroy()

    def quit_app(self):
        self.destroy()


if __name__ == "__main__":
    app = ImageReviewer(
        downloads_dir="data/downloads",
        keep_dir="dataset/raw",
        reject_dir="data/rejected",
        shuffle=True,
    )
    app.mainloop()