# Demo phân vùng ảnh: Nở vùng & Gộp vùng

Ứng dụng desktop (Tkinter) minh họa hai thuật toán phân vùng ảnh kinh điển:

- **Nở vùng (Region Growing)**: bắt đầu từ một hoặc nhiều điểm seed do người dùng nhấp chọn,
  lan dần sang các điểm lân cận có màu tương đồng.
- **Gộp vùng (Split and Merge)**: tách ảnh thành các khối vuông (quadtree) rồi gộp các khối
  kề nhau có màu trung bình gần giống nhau.

Kết quả được tô màu theo từng vùng và có thể vẽ biên, lưu ra file ảnh.

## Yêu cầu

- Python 3.9+
- Các thư viện:

```bash
pip install numpy pillow opencv-python scipy
```

> `scipy` chỉ cần cho tính năng "Lấp vùng chưa gán" của Region Growing.

## Cách chạy

```bash
python app.py
```

Ứng dụng mở ra với một **ảnh mẫu** (các hình hình học có nhiễu). Bạn có thể bấm
**Mở ảnh...** để chọn ảnh khác.

## Cấu trúc dự án

```
baitap4/
├── app.py            # Giao diện Tkinter + điều phối xử lý
├── segmentation.py   # Cài đặt thuật toán phân vùng và render kết quả
└── README.md
```

### `segmentation.py`

| Hàm | Vai trò |
| --- | --- |
| `region_growing(image, seeds, threshold, connectivity)` | Nở vùng đa seed, trả về label map (mỗi vùng một số nguyên, `-1` = chưa gán). |
| `split_and_merge(image, split_threshold, merge_threshold, min_size)` | Tách quadtree rồi gộp vùng, trả về label map. |
| `fill_unassigned(labels)` | Gán điểm `-1` cho vùng có nhãn gần nhất (dùng distance transform của SciPy). |
| `render_result(image, labels, filled, show_boundaries)` | Tô màu vùng + vẽ biên để hiển thị/lưu. |
| `_adjacent_pairs`, `_merge_pass`, `_relabel`, `_boundary_mask` | Các hàm phụ trợ (nội bộ). |

### `app.py`

| Thành phần | Vai trò |
| --- | --- |
| `make_sample()` | Sinh ảnh mẫu bằng Pillow rồi thêm nhiễu Gaussian. |
| `SegmentationApp` | Lớp giao diện chính: dựng panel điều khiển, canvas ảnh gốc/kết quả, xử lý sự kiện. |
| `_build_ui` / `_build_style` | Dựng bố cục và style. |
| `_load_image_array` | Chuẩn hóa ảnh về RGB, thu nhỏ nếu cạnh dài > `MAX_DIM`, reset trạng thái. |
| `on_canvas_click` | Chuyển tọa độ chuột (đã scale) về tọa độ ảnh để đặt seed. |
| `run` | Gọi thuật toán đang chọn, cập nhật label và hiển thị. |
| `save_result` | Lưu ảnh kết quả ra PNG/JPEG/BMP. |

## Cách dùng

### Nở vùng (Region Growing)

1. Chọn **Nở vùng (Region Growing)**.
2. Nhấp chuột lên ảnh gốc để đặt seed (có thể đặt nhiều seed cho nhiều vùng).
3. Chỉnh **Ngưỡng tương đồng** (chênh lệch màu tối đa được phép lan sang) và **Liên thông** 4/8.
4. Tùy chọn **Lấp vùng chưa gán** để phủ kín ảnh.
5. Bấm **Chạy phân vùng**.

Lưu ý: Mỗi seed tạo một vùng mới. Nếu seed rơi vào điểm đã được vùng trước chiếm, seed đó bị bỏ qua.

### Gộp vùng (Split and Merge)

1. Chọn **Gộp vùng (Split and Merge)**.
2. Chỉnh **Ngưỡng tách** (độ lệch max-min trong khối), **Ngưỡng gộp** (chênh lệch trung bình giữa hai vùng)
   và **Kích thước vùng tối thiểu**.
3. Bấm **Chạy phân vùng**.

Không cần đặt seed ở chế độ này.

## Ghi chú kỹ thuật

- **Label map**: mảng `int32` cùng kích thước ảnh; giá trị `>= 0` là id vùng, `-1` là chưa gán.
- Region Growing dùng `cv2.floodFill` với mask viền 1px bao quanh để chặn tràn biên. Mỗi seed dùng
  một giá trị mask riêng (`region_id`) nhờ bit dịch `<< 8`.
- Split and Merge dùng cấu trúc union-find trong mỗi lượt gộp, lặp tối đa 60 lượt cho đến khi ổn định.
- Màu tô vùng sinh ngẫu nhiên nhưng **cố định** nhờ seed của `numpy.default_rng`, nên chạy lại cho cùng màu.
- Ảnh được giới hạn cạnh dài `MAX_DIM = 420` px khi nạp để xử lý nhanh trên CPU.
- Tọa độ seed lưu trong `self.seeds` theo `(x, y)`, còn thuật toán nhận `(y, x)` nên có bước hoán đổi trong `run`.
