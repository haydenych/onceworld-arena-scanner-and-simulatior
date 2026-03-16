import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageOps, ImageTk


UNITS_DIR = Path("assets/units")
TARGET_DIR = Path("dataset_enemy_icons")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
UNLABELED_STEM_RE = re.compile(r"^img\d{4}$", re.IGNORECASE)

LEFT_PREVIEW_SIZE = (520, 520)
RIGHT_PREVIEW_SIZE = (320, 320)


def list_images(folder: Path, unlabeled_only=False):
    if not folder.exists():
        return []
    out = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if unlabeled_only and not UNLABELED_STEM_RE.match(p.stem):
            continue
        out.append(p)
    return sorted(out)


def next_labeled_path(target_dir: Path, class_name: str, src_ext: str):
    # Always save labels as png for consistency.
    ext = ".png"
    pat = re.compile(rf"^{re.escape(class_name)}_(\d+){re.escape(ext)}$", re.IGNORECASE)
    max_idx = 0
    for p in target_dir.glob(f"{class_name}_*{ext}"):
        m = pat.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return target_dir / f"{class_name}_{max_idx + 1:04d}{ext}"


def render_for_label(path: Path, size):
    img = Image.open(path).convert("RGB")
    contained = ImageOps.contain(img, size, method=Image.NEAREST)
    canvas = Image.new("RGB", size, (28, 28, 28))
    ox = (size[0] - contained.width) // 2
    oy = (size[1] - contained.height) // 2
    canvas.paste(contained, (ox, oy))
    return ImageTk.PhotoImage(canvas)


class LabelToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enemy Icon Label Tool")
        self.root.geometry("1220x760")

        self.unit_paths = list_images(UNITS_DIR)
        self.target_paths = list_images(TARGET_DIR, unlabeled_only=True)
        self.class_names = [p.stem for p in self.unit_paths]

        if not self.unit_paths:
            raise RuntimeError(f"No ground-truth images found in {UNITS_DIR}")
        if not self.target_paths:
            raise RuntimeError(f"No unlabeled img#### files found in {TARGET_DIR}")

        self.index = 0
        self.left_photo = None
        self.right_photo = None

        self._build_ui()
        self._refresh_views()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="left")

        btns = ttk.Frame(top)
        btns.pack(side="right")

        self.prev_btn = ttk.Button(btns, text="Prev", command=self.on_prev)
        self.prev_btn.pack(side="left", padx=4)

        self.next_btn = ttk.Button(btns, text="Next", command=self.on_next)
        self.next_btn.pack(side="left", padx=4)

        self.rename_next_btn = ttk.Button(btns, text="Rename + Next", command=self.on_rename_next)
        self.rename_next_btn.pack(side="left", padx=4)

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        left = ttk.Labelframe(body, text="Current image (dataset_enemy_icons)")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.left_img_label = ttk.Label(left)
        self.left_img_label.pack(fill="both", expand=True, padx=8, pady=8)

        right = ttk.Labelframe(body, text="Ground truth (assets/units)")
        right.pack(side="right", fill="both", expand=True)

        self.right_img_label = ttk.Label(right)
        self.right_img_label.pack(fill="x", padx=8, pady=(8, 4))

        list_wrap = ttk.Frame(right)
        list_wrap.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        self.class_list = tk.Listbox(list_wrap, exportselection=False)
        self.class_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.class_list.yview)
        sb.pack(side="right", fill="y")
        self.class_list.configure(yscrollcommand=sb.set)

        for name in self.class_names:
            self.class_list.insert("end", name)
        self.class_list.selection_set(0)
        self.class_list.bind("<<ListboxSelect>>", lambda _e: self._refresh_right_preview())

        self.root.bind("<Left>", lambda _e: self.on_prev())
        self.root.bind("<Right>", lambda _e: self.on_next())
        self.root.bind("<Return>", lambda _e: self.on_rename_next())

    def _selected_class_name(self):
        sel = self.class_list.curselection()
        if not sel:
            return None
        return self.class_list.get(sel[0])

    def _refresh_right_preview(self):
        class_name = self._selected_class_name()
        if class_name is None:
            self.right_img_label.configure(text="No class selected", image="")
            return
        p = UNITS_DIR / f"{class_name}.png"
        if not p.exists():
            matches = [x for x in self.unit_paths if x.stem == class_name]
            if not matches:
                self.right_img_label.configure(text=f"Missing: {class_name}", image="")
                return
            p = matches[0]
        self.right_photo = render_for_label(p, RIGHT_PREVIEW_SIZE)
        self.right_img_label.configure(image=self.right_photo, text="")

    def _refresh_views(self):
        if not self.target_paths:
            self.status_var.set("All done: no images left.")
            self.left_img_label.configure(text="No images left.", image="")
            return

        self.index = max(0, min(self.index, len(self.target_paths) - 1))
        cur = self.target_paths[self.index]

        self.left_photo = render_for_label(cur, LEFT_PREVIEW_SIZE)
        self.left_img_label.configure(image=self.left_photo, text="")

        self._refresh_right_preview()
        self.status_var.set(
            f"{self.index + 1}/{len(self.target_paths)} | current: {cur.name}"
        )

    def on_prev(self):
        if not self.target_paths:
            return
        self.index = max(0, self.index - 1)
        self._refresh_views()

    def on_next(self):
        if not self.target_paths:
            return
        self.index = min(len(self.target_paths) - 1, self.index + 1)
        self._refresh_views()

    def on_rename_next(self):
        if not self.target_paths:
            return

        class_name = self._selected_class_name()
        if not class_name:
            messagebox.showwarning("No class selected", "Select a class on the right first.")
            return

        cur = self.target_paths[self.index]
        dst = next_labeled_path(TARGET_DIR, class_name, cur.suffix.lower())
        try:
            cur.replace(dst)
        except Exception as e:
            messagebox.showerror("Rename failed", str(e))
            return

        # Keep queue order stable: update current entry in-place, then advance.
        self.target_paths[self.index] = dst
        self.index = min(self.index + 1, len(self.target_paths) - 1)
        self._refresh_views()


def main():
    root = tk.Tk()
    try:
        LabelToolApp(root)
    except Exception as e:
        messagebox.showerror("Startup error", str(e))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
