# Nguồn dữ liệu và phần mềm

- Dataset: [BOP YCB-V](https://bop.felk.cvut.cz/datasets/) và [Hugging Face bop-benchmark/ycbv](https://huggingface.co/datasets/bop-benchmark/ycbv). Revision và LFS SHA256 trong download_manifest.
- Pretrained và kiến trúc: [CosyPose source](https://github.com/ylabbe/cosypose) / source repository yannlabb/cosypose. Vendor commit c90a04f434b1e89f02341cc03899eb63ea8facba, giấy phép MIT trong third_party/cosypose/LICENSE.
- Published model IDs: detector-bop-ycbv-synt+real--292971, coarse-bop-ycbv-synt+real--822463, refiner-bop-ycbv-synt+real--631598. URL file cụ thể và hash ghi sau HTTP transfer trong reports/ycbv/download_manifest.json.
- Torchvision Mask R-CNN, EfficientNet source vendor trong CosyPose, Pyrender, Trimesh, NumPy/SciPy, pycocotools, Matplotlib và Plotly: giữ giấy phép riêng của mỗi thư viện.

Windows adapter nằm trong src/clutterpick/models/cosypose.py, không sửa source vendor. Renderer đổi từ Bullet/EGL sang Pyrender, do đó benchmark tại đây là kết quả implementation hiện tại, không khẳng định là điểm original paper. Tham khảo LICENSE của source và điều khoản dữ liệu khi phân phối lại dataset/checkpoint.

