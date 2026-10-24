# Xử lý lỗi thường gặp

Chạy các lệnh từ gốc repo và dùng đúng Python của môi trường:

```powershell
.\.venv\Scripts\python.exe main.py --version
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

| Thông báo hoặc hiện tượng | Kiểm tra và xử lý |
|---|---|
| `No module named clutterpick` | Dùng `main.py` từ gốc repo hoặc cài editable bằng `python -m pip install -e '.[test]'` |
| `Checkpoint unavailable` | Chạy `python scripts/data/prepare_ycbv.py`; kiểm tra `detector.checkpoint` trong cấu hình |
| `Requested frame/split unavailable` | Chạy `index` với đúng `--config`; kiểm tra frame key trong index |
| `Training requires separate train/validation scenes` | Tải `train-pbr` và xây lại index; baseline chỉ có test không đủ để train |
| `Configuration must be a YAML mapping` | File phải chứa mapping, ví dụ `seed: 42`, không phải list hoặc file rỗng |
| `Invalid YAML configuration` | Xem đường dẫn, dòng và cột trong lỗi; sửa indent hoặc dấu ngoặc |
| `Expected a positive integer` | `--limit`, `--epochs`, `--val-limit` cần số nguyên lớn hơn 0 |
| `Depth is not a registered single-channel image` | Depth phải cùng kích thước RGB và có một kênh |
| `Expected uint16 metric depth` | Dùng depth raw 16 bit; không dùng ảnh preview 8 bit |
| `Depth scale must be finite and positive` | Dùng scale milimét trên đơn vị raw, lớn hơn 0 |
| Viewer thiếu `pointcloud.ply` | Chạy `infer` với cloud export mặc định; output benchmark không có cloud |

Nếu CUDA không khả dụng, kiểm tra môi trường đã cài đúng bản PyTorch và driver.
CLI có `--device cpu`; tốc độ CPU chưa được benchmark. Renderer Pyrender vẫn
cần môi trường OpenGL hoạt động khi dùng pose backend tương ứng.

Khi báo lỗi, ghi lại lệnh đã chạy, phiên bản Python/PyTorch, cấu hình, frame key
và traceback. Không cần tải lại dataset hoặc chạy lại toàn benchmark để kiểm
tra lỗi cú pháp CLI. Test với một frame và output mới trước.

Tham khảo [cấu hình](../configs/README.md), [RGB-D riêng](EXTERNAL_RGBD_VI.md)
và [các file đầu ra](OUTPUTS_VI.md).
