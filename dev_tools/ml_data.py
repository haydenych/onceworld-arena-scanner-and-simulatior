import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onceworld.core.data_collection import IconDataCollector


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Data Collector")
        self.root.geometry("900x650")

        try:
            self.collector = IconDataCollector()
        except Exception as exc:
            messagebox.showerror("Startup error", str(exc))
            root.destroy()
            return

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        self.save_btn = ttk.Button(top, text="Save Data", command=self.on_save_data)
        self.save_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        self.text = tk.Text(root, wrap="word", font=("Consolas", 11))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self.write("Ready.\n")

    def write(self, s):
        self.text.insert("end", s)
        self.text.see("end")

    def on_save_data(self):
        self.save_btn.config(state="disabled")
        self.status_var.set("Capturing...")
        self.root.update_idletasks()
        try:
            saved, logs = self.collector.capture_and_save()
            for line in logs:
                self.write(line + "\n")
            self.write("\n")
            self.status_var.set(f"Saved {len(saved)}")
        except Exception as exc:
            messagebox.showerror("Capture error", str(exc))
            self.status_var.set("Error")
        finally:
            self.save_btn.config(state="normal")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

