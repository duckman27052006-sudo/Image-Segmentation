"""Ung dung demo phan vung anh: No vung (Region Growing) va Gop vung (Split and Merge)."""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from segmentation import (region_growing, render_result, split_and_merge)
from segmentation import fill_unassigned as fill_nearest

MAX_DIM = 420


def make_sample():
    """Tạo ảnh mẫu: vài hình hình học trên nền sáng, thêm nhiễu Gaussian.

    Ảnh mẫu giúp người dùng thử thuật toán ngay mà không cần file ảnh bên ngoài.
    Nhiễu nhẹ khiến việc chọn ngưỡng trở nên có ý nghĩa (không quá "hoàn hảo").
    """
    img = Image.new("RGB", (460, 340), (238, 240, 245))
    d = ImageDraw.Draw(img)
    d.ellipse([35, 35, 190, 190], fill=(205, 82, 82))
    d.rectangle([235, 45, 425, 160], fill=(72, 140, 200))
    d.ellipse([290, 185, 430, 320], fill=(92, 178, 112))
    d.rectangle([55, 235, 185, 320], fill=(232, 186, 70))
    d.polygon([(200, 190), (255, 300), (145, 300)], fill=(150, 110, 195))

    # Cộng nhiễu chuẩn (mean=0, std=7) rồi kẹp về [0, 255].
    arr = np.asarray(img).astype(np.int16)
    rng = np.random.default_rng(7)
    arr = np.clip(arr + rng.normal(0, 7, arr.shape), 0, 255).astype(np.uint8)
    return arr


class SegmentationApp:
    """Lớp giao diện chính của ứng dụng demo phân vùng ảnh."""

    def __init__(self, root):
        self.root = root
        self.root.title("Demo phân vùng ảnh - Nở vùng & Gộp vùng")
        self.root.geometry("1200x620")
        self.root.minsize(980, 600)

        # --- Trạng thái dữ liệu ---
        self.image = None          # ảnh RGB đang thao tác (numpy uint8)
        self.seeds = []            # danh sách seed người dùng đặt, dạng (x, y)
        self.labels = None         # label map đang hiển thị (đã áp dụng lấp vùng)
        self.raw_labels = None     # label map gốc do thuật toán trả về
        self.last_result = None    # ảnh kết quả đã render để hiển thị/lưu
        self.photo_orig = None     # ảnh Tkinter của canvas gốc (giữ tham chiếu!)
        self.photo_result = None   # ảnh Tkinter của canvas kết quả
        self.orig_geom = (1.0, 0, 0)  # (scale, offset_x, offset_y) khi vẽ ảnh gốc
        self._resize_job = None    # id của job after() đang chờ khi resize

        # --- Biến gắn với widget điều khiển ---
        self.mode = tk.StringVar(value="grow")
        self.grow_thresh = tk.IntVar(value=25)
        self.connectivity = tk.StringVar(value="8")
        self.split_thresh = tk.IntVar(value=24)
        self.merge_thresh = tk.IntVar(value=18)
        self.min_size = tk.IntVar(value=150)
        self.fill_unassigned = tk.BooleanVar(value=False)
        self.fill_regions = tk.BooleanVar(value=True)
        self.show_borders = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Sẵn sàng.")
        self.view_hint = tk.StringVar(value="Nhấp chuột để đặt seed (nở vùng)")

        self._build_style()
        self._build_ui()
        # Khởi động bằng ảnh mẫu để người dùng thấy ngay giao diện có nội dung.
        self._load_image_array(make_sample(), "Ảnh mẫu")

    def _build_style(self):
        """Thiết lập theme và các style dùng chung cho giao diện."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(8, 5))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#666666")

    def _build_ui(self):
        """Dựng toàn bộ bố cục: panel điều khiển bên trái, 2 canvas bên phải."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ===== Panel điều khiển (cột trái) =====
        panel = ttk.Frame(self.root, padding=12)
        panel.grid(row=0, column=0, sticky="nsw")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="PHÂN VÙNG ẢNH", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 2))
        ttk.Label(panel, text="Nở vùng & Gộp vùng", style="Sub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, 10))

        # Hàng nút chọn ảnh: mở file hoặc dùng ảnh mẫu.
        file_row = ttk.Frame(panel)
        file_row.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        file_row.columnconfigure(0, weight=1)
        file_row.columnconfigure(1, weight=1)
        ttk.Button(file_row, text="Mở ảnh...", command=self.open_image).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(file_row, text="Ảnh mẫu", command=self.load_sample).grid(
            row=0, column=1, sticky="ew", padx=(4, 0))

        algo = ttk.LabelFrame(panel, text="Thuật toán", padding=8)
        algo.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(algo, text="Nở vùng (Region Growing)", value="grow",
                        variable=self.mode, command=self._on_mode_change).pack(anchor="w")
        ttk.Radiobutton(algo, text="Gộp vùng (Split and Merge)", value="merge",
                        variable=self.mode, command=self._on_mode_change).pack(anchor="w")

        # Hai khung tham số chồng cùng một ô; _on_mode_change sẽ ẩn/hiện khung
        # tương ứng với thuật toán đang chọn.
        self.param_host = ttk.Frame(panel)
        self.param_host.grid(row=4, column=0, sticky="ew")
        self.param_host.columnconfigure(0, weight=1)

        # --- Khung tham số cho Nở vùng ---
        self.grow_frame = ttk.LabelFrame(self.param_host, text="Tham số nở vùng", padding=8)
        self.grow_frame.grid(row=0, column=0, sticky="ew")
        self._add_scale(self.grow_frame, "Ngưỡng tương đồng", self.grow_thresh, 1, 120)
        conn_row = ttk.Frame(self.grow_frame)
        conn_row.pack(fill="x", pady=(6, 2))
        ttk.Label(conn_row, text="Liên thông:").pack(side="left")
        ttk.Combobox(conn_row, textvariable=self.connectivity, values=["4", "8"],
                     width=4, state="readonly").pack(side="left", padx=(6, 0))
        ttk.Checkbutton(self.grow_frame,
                        text="Lấp vùng chưa gán (theo vùng gần nhất)",
                        variable=self.fill_unassigned,
                        command=self._on_fill_toggle).pack(anchor="w", pady=(8, 0))
        ttk.Button(self.grow_frame, text="Xóa các seed",
                   command=self.clear_seeds).pack(fill="x", pady=(8, 0))
        ttk.Label(self.grow_frame, style="Sub.TLabel", wraplength=225,
                  text="Nhấp chuột lên ảnh gốc để đặt seed. Có thể đặt nhiều seed "
                       "cho nhiều vùng. Thuật toán nở dần sang điểm lân cận có màu "
                       "chênh lệch không quá ngưỡng.").pack(anchor="w", pady=(8, 0))

        # --- Khung tham số cho Gộp vùng ---
        self.merge_frame = ttk.LabelFrame(self.param_host, text="Tham số gộp vùng", padding=8)
        self.merge_frame.grid(row=0, column=0, sticky="ew")
        self._add_scale(self.merge_frame, "Ngưỡng tách", self.split_thresh, 2, 100)
        self._add_scale(self.merge_frame, "Ngưỡng gộp", self.merge_thresh, 0, 80)
        self._add_scale(self.merge_frame, "Kích thước vùng tối thiểu", self.min_size, 0, 1500, step=10)
        ttk.Label(self.merge_frame, style="Sub.TLabel", wraplength=225,
                  text="Ảnh được tách dần thành các khối vuông (quadtree) khi độ lệch "
                       "lớn hơn ngưỡng tách, sau đó các khối kề nhau được gộp lại nếu "
                       "trung bình màu gần nhau.").pack(anchor="w", pady=(8, 0))

        # ===== Hàng nút hành động: chạy / lưu / xóa kết quả =====
        action_row = ttk.Frame(panel)
        action_row.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Button(action_row, text="Chạy phân vùng", style="Accent.TButton",
                   command=self.run).grid(row=0, column=0, sticky="ew")
        ttk.Button(action_row, text="Lưu kết quả", command=self.save_result).grid(
            row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(action_row, text="Xóa kết quả", command=self.clear_result).grid(
            row=2, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(panel, textvariable=self.status, style="Sub.TLabel",
                  wraplength=235, justify="left").grid(row=6, column=0, sticky="w", pady=(10, 0))

        # ===== Vùng hiển thị ảnh: canvas gốc bên trái, kết quả bên phải =====
        view = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.columnconfigure(1, weight=1)
        view.rowconfigure(0, weight=0)

        orig_box = ttk.LabelFrame(view, text="Ảnh gốc", padding=8)
        orig_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        orig_box.rowconfigure(1, weight=1)
        orig_box.columnconfigure(0, weight=1)
        ttk.Label(orig_box, textvariable=self.view_hint,
                  style="Sub.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.orig_canvas = tk.Canvas(orig_box, width=MAX_DIM, height=MAX_DIM * 3 // 4,
                                     bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.orig_canvas.grid(row=1, column=0, sticky="nsew")
        self.orig_canvas.bind("<Button-1>", self.on_canvas_click)
        self.orig_canvas.bind("<Configure>", self._on_canvas_resize)

        # Canvas kết quả: có thêm 2 checkbox tô màu vùng / vẽ biên.
        res_box = ttk.LabelFrame(view, text="Kết quả phân vùng", padding=8)
        res_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        res_box.rowconfigure(1, weight=1)
        res_box.columnconfigure(0, weight=1)
        opts = ttk.Frame(res_box)
        opts.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Checkbutton(opts, text="Tô màu vùng", variable=self.fill_regions,
                        command=self._rerender).pack(side="left")
        ttk.Checkbutton(opts, text="Vẽ biên", variable=self.show_borders,
                        command=self._rerender).pack(side="left", padx=(10, 0))
        self.result_canvas = tk.Canvas(res_box, width=MAX_DIM, height=MAX_DIM * 3 // 4,
                                       bg="#2b2b2b", highlightthickness=0)
        self.result_canvas.grid(row=1, column=0, sticky="nsew")
        self.result_canvas.bind("<Configure>", self._on_canvas_resize)

        self._on_mode_change()

    def _add_scale(self, parent, text, var, from_, to, step=1):
        """Thêm một slider kèm nhãn tên và giá trị hiện tại.

        `var` là IntVar dùng chung với slider; nhãn giá trị được cập nhật
        mỗi khi kéo slider.
        """
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text=text).pack(side="left")
        value_label = ttk.Label(row, text=str(var.get()), width=5, anchor="e")
        value_label.pack(side="right")

        def update(v):
            # Slider trả về float dạng chuỗi, nên làm tròn về số nguyên.
            value_label.config(text=str(int(round(float(v)))))

        ttk.Scale(frame, from_=from_, to=to, variable=var, command=update).pack(
            fill="x", pady=(2, 0))

    def _on_mode_change(self):
        """Ẩn/hiện khung tham số tương ứng với thuật toán đang chọn."""
        if self.mode.get() == "grow":
            self.merge_frame.grid_remove()
            self.grow_frame.grid()
            self.view_hint.set("Nhấp chuột để đặt seed (nở vùng)")
        else:
            self.grow_frame.grid_remove()
            self.merge_frame.grid()
            self.view_hint.set("Thuật toán gộp vùng không cần seed")

    def _load_image_array(self, array, source):
        """Nạp mảng ảnh vào ứng dụng và reset toàn bộ trạng thái cũ.

        - Chuyển ảnh xám sang 3 kênh nếu cần.
        - Thu nhỏ ảnh nếu cạnh dài vượt MAX_DIM để xử lý nhanh.
        """
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        pil = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        w, h = pil.size
        scale = min(1.0, MAX_DIM / max(w, h))
        if scale < 1.0:
            pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        self.image = np.asarray(pil)
        # Ảnh mới thì seed và kết quả cũ không còn ý nghĩa.
        self.seeds = []
        self.labels = None
        self.raw_labels = None
        self.last_result = None
        self._draw_original()
        self._clear_result()
        self._set_canvas_size(self.orig_canvas, self.image)
        self._set_canvas_size(self.result_canvas, self.image)
        h, w = self.image.shape[:2]
        self.status.set(f"Đã nạp {source}: {w}x{h} px.")

    def _set_canvas_size(self, canvas, array):
        """Đặt kích thước canvas bằng kích thước ảnh (đơn vị pixel ảnh)."""
        h, w = array.shape[:2]
        canvas.config(width=w, height=h)

    def _canvas_size(self, canvas):
        """Lấy kích thước hiện tại của canvas, có fallback trước khi widget vẽ xong."""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1:
            cw = int(canvas["width"])
        if ch <= 1:
            ch = int(canvas["height"])
        return max(cw, 1), max(ch, 1)

    def _fit(self, canvas, array):
        """Thu/phóng ảnh vừa canvas và căn giữa.

        Trả về (PhotoImage, scale, offset_x, offset_y). Bộ ba (scale, ox, oy)
        cũng là cơ sở để đổi tọa độ chuột sang tọa độ ảnh.
        """
        cw, ch = self._canvas_size(canvas)
        h, w = array.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        pil = Image.fromarray(array)
        if (nw, nh) != (w, h):
            pil = pil.resize((nw, nh), Image.LANCZOS)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        return ImageTk.PhotoImage(pil), scale, ox, oy

    def _draw_original(self):
        """Vẽ ảnh gốc lên canvas trái kèm các seed đã đặt."""
        canvas = self.orig_canvas
        canvas.delete("all")
        if self.image is None:
            return
        self.photo_orig, scale, ox, oy = self._fit(canvas, self.image)
        # Lưu lại hệ số biến đổi để on_canvas_click đổi ngược tọa độ chuột.
        self.orig_geom = (scale, ox, oy)
        canvas.create_image(ox, oy, anchor="nw", image=self.photo_orig)
        # Mỗi seed là một vòng tròn đỏ kèm số thứ tự.
        for i, (x, y) in enumerate(self.seeds, start=1):
            px = ox + x * scale
            py = oy + y * scale
            canvas.create_oval(px - 6, py - 6, px + 6, py + 6,
                               outline="#ff3b30", width=2)
            canvas.create_text(px + 10, py - 10, text=str(i), fill="#ff3b30",
                               font=("Segoe UI", 9, "bold"))

    def _clear_result(self):
        """Xóa canvas kết quả và hiển thị dòng hướng dẫn mặc định."""
        canvas = self.result_canvas
        canvas.delete("all")
        cw, ch = self._canvas_size(canvas)
        canvas.create_text(cw // 2, ch // 2, text="Kết quả sẽ hiển thị ở đây",
                           fill="#9aa0a6", font=("Segoe UI", 10))

    def _show_result(self):
        """Vẽ ảnh kết quả (self.last_result) lên canvas phải."""
        if self.last_result is None:
            return
        canvas = self.result_canvas
        canvas.delete("all")
        self.photo_result, scale, ox, oy = self._fit(canvas, self.last_result)
        canvas.create_image(ox, oy, anchor="nw", image=self.photo_result)

    def _on_canvas_resize(self, event=None):
        """Gom nhiều sự kiện resize liên tiếp thành một lần vẽ lại (debounce).

        Việc resize cửa sổ phát sinh rất nhiều sự kiện <Configure>; chờ 60ms
        rồi mới vẽ lại giúp giao diện không bị giật.
        """
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(60, self._redraw_display)

    def _redraw_display(self):
        """Vẽ lại cả hai canvas theo kích thước hiện tại."""
        self._resize_job = None
        if self.image is None:
            return
        self._draw_original()
        if self.last_result is not None:
            self._show_result()
        else:
            self._clear_result()

    def _rerender(self):
        """Render lại kết quả khi thay đổi tùy chọn tô màu/biên."""
        if self.image is None or self.labels is None:
            return
        self.last_result = render_result(
            self.image, self.labels,
            filled=self.fill_regions.get(),
            show_boundaries=self.show_borders.get())
        self._show_result()

    def _update_labels(self):
        """Áp dụng (nếu cần) lấp vùng rồi render lại kết quả.

        `raw_labels` giữ kết quả gốc; `labels` là bản đã xử lý để hiển thị.
        Tách hai biến giúp bật/tắt lấp vùng mà không phải chạy lại thuật toán.
        """
        if self.image is None or self.raw_labels is None:
            return
        labels = self.raw_labels
        if self.mode.get() == "grow" and self.fill_unassigned.get():
            labels = fill_nearest(labels)
        self.labels = labels
        self.last_result = render_result(
            self.image, labels,
            filled=self.fill_regions.get(),
            show_boundaries=self.show_borders.get())
        self._show_result()

    def _on_fill_toggle(self):
        """Xử lý khi người dùng bật/tắt checkbox 'Lấp vùng chưa gán'."""
        if self.raw_labels is None:
            return
        self._update_labels()
        if self.fill_unassigned.get():
            n = int(self.raw_labels.max()) + 1
            if n <= 1:
                self.status.set("Chỉ có 1 vùng: lấp vùng sẽ phủ kín ảnh một màu. "
                                "Hãy đặt thêm seed để có nhiều vùng.")
            else:
                self.status.set("Đã lấp vùng chưa gán theo vùng gần nhất.")
        else:
            self.status.set("Đã tắt lấp vùng chưa gán.")

    def clear_result(self):
        """Xóa toàn bộ kết quả phân vùng đang hiển thị."""
        self.labels = None
        self.raw_labels = None
        self.last_result = None
        self.photo_result = None
        self._clear_result()
        self.status.set("Đã xóa kết quả phân vùng.")

    def on_canvas_click(self, event):
        """Đặt một seed tại vị trí nhấp chuột trên ảnh gốc."""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        scale, ox, oy = self.orig_geom
        # Đổi tọa độ màn hình -> tọa độ ảnh bằng cách trừ offset và chia scale.
        x = (event.x - ox) / scale
        y = (event.y - oy) / scale
        # Bỏ qua nếu nhấp ra ngoài vùng ảnh (phần viền canvas).
        if not (0 <= x < w and 0 <= y < h):
            return
        x, y = int(x), int(y)
        self.seeds.append((x, y))
        self._draw_original()
        self.status.set(f"Đã đặt seed {len(self.seeds)} tại ({x}, {y}).")

    def clear_seeds(self):
        """Xóa tất cả seed đã đặt."""
        self.seeds = []
        self._draw_original()
        self.status.set("Đã xóa tất cả seed.")

    def run(self):
        """Chạy thuật toán đang chọn và hiển thị kết quả."""
        if self.image is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy mở ảnh hoặc dùng ảnh mẫu trước.")
            return

        mode = self.mode.get()
        # Nở vùng bắt buộc phải có seed; gộp vùng thì không.
        if mode == "grow" and not self.seeds:
            messagebox.showwarning(
                "Chưa có seed",
                "Hãy nhấp chuột lên ảnh gốc để chọn ít nhất một điểm seed.")
            return

        # Cập nhật trạng thái trước khi chạy để người dùng thấy ứng dụng đang bận.
        self.status.set("Đang xử lý...")
        self.root.update_idletasks()
        t0 = time.perf_counter()

        try:
            if mode == "grow":
                # self.seeds lưu (x, y); thuật toán nhận (y, x) nên hoán đổi.
                seeds = [(y, x) for x, y in self.seeds]
                labels = region_growing(self.image, seeds,
                                        self.grow_thresh.get(),
                                        int(self.connectivity.get()))
            else:
                labels = split_and_merge(self.image,
                                         self.split_thresh.get(),
                                         self.merge_thresh.get(),
                                         self.min_size.get())
        except Exception as exc:  # pragma: no cover - hien thi loi cho nguoi dung
            messagebox.showerror("Lỗi", f"Không thể phân vùng:\n{exc}")
            self.status.set("Xử lý thất bại.")
            return

        elapsed = time.perf_counter() - t0
        # max() < 0 nghĩa là mọi điểm vẫn là -1: không có vùng nào được tạo.
        if labels.max() < 0:
            self.status.set("Không tạo được vùng nào. Hãy tăng ngưỡng hoặc đổi seed.")
            return

        self.raw_labels = labels
        self._update_labels()
        n_regions = int(labels.max()) + 1
        note = ""
        if mode == "grow" and self.fill_unassigned.get() and n_regions == 1:
            note = (" — chỉ có 1 vùng nên lấp phủ kín ảnh một màu; "
                    "đặt thêm seed để tách nhiều vùng")
        self.status.set(f"Xong: {n_regions} vùng, {elapsed:.2f} giây.{note}")

    def open_image(self):
        """Mở hộp thoại chọn file ảnh và nạp ảnh đó."""
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Tất cả", "*.*")])
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh:\n{exc}")
            return
        self._load_image_array(np.asarray(pil), os.path.basename(path))

    def load_sample(self):
        """Nạp lại ảnh mẫu."""
        self._load_image_array(make_sample(), "ảnh mẫu")

    def save_result(self):
        """Lưu ảnh kết quả (self.last_result) ra file."""
        if self.last_result is None:
            messagebox.showinfo("Chưa có kết quả", "Hãy chạy phân vùng trước khi lưu.")
            return
        path = filedialog.asksaveasfilename(
            title="Lưu kết quả", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Bitmap", "*.bmp")])
        if not path:
            return
        try:
            Image.fromarray(self.last_result).save(path)
        except Exception as exc:
            messagebox.showerror("Lỗi", f"Không lưu được ảnh:\n{exc}")
            return
        self.status.set(f"Đã lưu: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    SegmentationApp(root)
    root.mainloop()
