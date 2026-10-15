"""Install the explicit experimental hybrid config and update project entry points."""
import sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import load_config,resolve

def main():
    cfg=load_config('configs/research.yaml')
    cfg['detector']['architecture']='fusion'
    cfg['detector']['checkpoint']='artifacts/ycbv/research/fusion/gated_query/best.pt'
    cfg['point_refiner']={'enabled':True,'checkpoint':'artifacts/ycbv/research/point_network/best.pt'}
    resolve('configs/hybrid.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    point_cfg=load_config('configs/enhanced.yaml')
    point_cfg['point_refiner']=cfg['point_refiner'].copy()
    resolve('configs/point_refined.yaml').write_text(yaml.safe_dump(point_cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    section='''
## Ba module học mới (v0.4)

Đã triển khai bốn chế độ RGB/RGB-D fusion, query Transformer mask head 56×56 và local point-attention foreground refinement trước robust ICP. Protocol mới dùng 300 PBR train / 60 validation / 60 holdout theo scene, index riêng; checkpoint và threshold chọn bằng validation. Published pretrained có thể đã gặp các PBR scenes này.

Xem [thiết kế và pipeline đầy đủ](docs/LEARNED_HYBRID_VI.md), [kết quả thí nghiệm](reports/ycbv/research/RESULTS_VI.md) và [kiểm tra bàn giao](reports/ycbv/research/delivery_validation.json). Các thí nghiệm mới trên 96 test frames / 24 pose frames là pilot, tách biệt với baseline full 900 ảnh. Kết quả chỉ hoàn tất khi report và delivery validation tồn tại với `passed=true`.

`configs/hybrid.yaml` kích hoạt đủ ba module (gated_query + learned point filter + robust ICP). Đây là cấu hình thử nghiệm, chưa mặc định rằng tốt hơn baseline. Chạy sau khi checkpoints train xong:

```powershell
.\\.venv\\Scripts\\python.exe scripts/run.py --config configs/hybrid.yaml infer --frame-key ycbv_test/000048/1074 --output artifacts/ycbv/hybrid_demo
```

Lệnh toàn bộ: `powershell -ExecutionPolicy Bypass -File scripts/run_research.ps1`. Downloader subset `scripts/prepare_pbr_subset.py` chỉ tải các ZIP members cần thiết; không cần tải toàn archive train_pbr 21 GB.
'''
    path=resolve('README.md'); content=path.read_text(encoding='utf-8')
    heading='## Fine-tune và ảnh RGB-D riêng'
    if '## Ba module học mới (v0.4)' not in content:
        content=content.replace(heading,section+'\n'+heading)
    content=content.replace('Hiện **train=0, val=0, test=900**; kết quả là inference với pretrained, chưa có kết quả train mới.',
                            'Index baseline ghi **train=0, val=0, test=900**; kết quả baseline dùng pretrained. Index research riêng có **train=300, val=60, holdout=60, test=900** cho các module mới.')
    content=content.replace('vì chưa tải train/val.','trong benchmark baseline ban đầu; training mới không retune ICP trên test.')
    content=content.replace('GT mask chỉ được dùng cho oracle ablation và đánh giá.',
                            'GT mask được dùng cho supervision train/validation của các module mới, oracle ablation và đánh giá; inference thực không dùng GT.')
    content=content.replace('`configs/default.yaml`: cấu hình YCB-V duy nhất.',
                            '`configs/default.yaml`: baseline; `enhanced.yaml`: robust ICP; `research.yaml`: protocol train/ablation; `hybrid.yaml`: ba module mới.')
    path.write_text(content,encoding='utf-8')
    path=resolve('PROJECT_SUMMARY_VI.md'); content=path.read_text(encoding='utf-8')
    content=content.replace('Chưa tải train_real/train_pbr để tiết kiệm dung lượng.',
                            'Baseline ban đầu chưa tải train_real/train_pbr. Phiên bản v0.4 đã tải subset 420 PBR frames và có index research riêng; train_real vẫn chưa tải.')
    content=content.replace('không tuyên bố có mạng 3D segmentation riêng.',
                            'baseline không có mạng 3D riêng; cấu hình hybrid v0.4 có local point-attention foreground network, xem phần nâng cấp bên dưới.')
    content=content.replace('- Chưa fine-tune detector hoặc train pose network mới.',
                            '- v0.4 fine-tune segmentation trên subset PBR và train point network; pose networks vẫn dùng pretrained.')
    content=content.replace('91 trường hợp cải thiện và 0 trường hợp giảm chất lượng.',
                            '91 GT fail→pass và 0 pass→fail tại ngưỡng 0.1 diameter; không khẳng định mọi numerical pose error đều giảm.')
    if '## Ba module học mới (v0.4)' not in content:
        content+='\n'+section
    path.write_text(content,encoding='utf-8')

if __name__=='__main__': main()
