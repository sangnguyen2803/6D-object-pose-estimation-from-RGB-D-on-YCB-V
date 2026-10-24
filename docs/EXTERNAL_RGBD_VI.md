# Chạy ảnh RGB-D riêng

Chuẩn bị RGB và depth đã đăng ký vào cùng hệ pixel, cùng kích thước. Depth phải
là ảnh PNG một kênh `uint16`, giữ nguyên giá trị raw từ camera. Không dùng ảnh
depth đã tô màu hoặc chuẩn hóa về 8 bit. Pixel depth bằng 0 được coi là thiếu.

Lấy ma trận nội tại từ calibration của ảnh đã đăng ký. Thứ tự chín giá trị là
theo hàng: `fx skew cx 0 fy cy 0 0 1`. Nếu resize hoặc crop ảnh, cần cập nhật K
tương ứng trước khi chạy.

Ví dụ sau dùng K minh họa; thay bằng thông số thực của camera:

```powershell
python scripts/inference/infer_rgbd.py --rgb D:/capture/rgb.png --depth D:/capture/depth.png --intrinsics 600 0 320 0 600 240 0 0 1 --depth-scale-mm 1 --output artifacts/ycbv/my_capture
```

`--depth-scale-mm` là số milimét trên mỗi đơn vị raw. Công thức đổi sang mét:
`z_m = raw * depth_scale_mm / 1000`. Với scale 1, raw 1000 là 1 m; với scale
0.1, raw 10000 là 1 m. Scale phải hữu hạn và lớn hơn 0.

Script dùng checkpoint theo `--config` (mặc định `configs/default.yaml`). Chuẩn
bị checkpoint và CAD như phần cài đặt trong [README](../README.md). Không cần
GT labels hoặc dataset index cho ảnh riêng. Model chỉ nhận 21 lớp YCB đã biết.

Kết quả của ví dụ nằm trong `artifacts/ycbv/my_capture/external_000000_0`.
Để xem point cloud:

```powershell
python scripts/inference/visualize_instances.py artifacts/ycbv/my_capture/external_000000_0
```

Không có GT thì không diễn giải diagnostics thành độ chính xác pose. Kiểm tra
overlay và đơn vị depth trước khi so sánh dữ liệu từ các camera.
