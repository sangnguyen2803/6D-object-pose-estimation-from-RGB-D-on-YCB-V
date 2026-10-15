"""Produce reports from measured YCB-V artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import resolve,read_json,write_json
from clutterpick.reporting import summarize,per_object
import numpy as np
def main():
    output=resolve('reports/ycbv'); output.mkdir(parents=True,exist_ok=True)
    detection=read_json(resolve('artifacts/ycbv/evaluation/detection.json'))
    rows=read_json(resolve('artifacts/ycbv/benchmark_full/run_summary.json'))
    index=read_json(resolve('data/ycbv/index.json'))
    pose=summarize(rows); pose['per_object']=per_object(rows,index)
    frame_lookup={frame['key']:frame for frame in index['frames']}
    visibility={key:{'gt':0,'success':0} for key in ['0_25','25_50','50_75','75_100']}
    failures=[]
    for row in rows:
        matches={item['reference_instance_id']:item for item in row['pose_diagnostics']}
        for obj in frame_lookup[row['frame_key']]['instances']:
            bucket=list(visibility)[min(3,max(0,int(obj['visib_fract']*4)))]
            visibility[bucket]['gt']+=1
            item=matches.get(obj['instance_id'])
            success=bool(item and item['ADD_or_S_success_0_1d'])
            visibility[bucket]['success']+=int(success)
            if not success and len(failures)<40:
                failures.append({'frame_key':row['frame_key'],'object_id':obj['obj_id'],
                                 'reference_instance_id':obj['instance_id'],'visibility':obj['visib_fract'],
                                 'failure_type':'pose_error' if item else 'no_matched_pose',
                                 'overlay':str(Path(row['output'])/'final_visualization.png')})
    for result in visibility.values():
        result['recall']=result['success']/result['gt'] if result['gt'] else None
    pose['visibility_recall']=visibility
    write_json(output/'failure_examples.json',failures)
    ep=resolve('artifacts/ycbv/experiments/summary.json')
    experiments=read_json(ep) if ep.exists() else None
    write_json(output/'measured_summary.json',{'dataset':'YCB-V BOP19','detector':detection['metrics'],
               'pose':pose,'experiments':experiments,'complete_pose_test':len(rows)==detection['frame_count']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['ADD','ADD-S','ADD(-S)']; values=[pose['recall'][key]*100 for key in pose['recall']]
    fig,ax=plt.subplots(figsize=(7,4)); bars=ax.bar(labels,values,color=['#4477aa','#66ccee','#228833'])
    ax.bar_label(bars,fmt='%.1f%%'); ax.set_ylim(0,105)
    ax.set_ylabel('Recall / all GT instances (%)')
    ax.set_title(f"YCB-V: {pose['frames']} frames, {pose['gt_instances']} GT instances")
    fig.tight_layout(); fig.savefig(output/'pose_recall.png',dpi=180); plt.close(fig)
    per_class=detection['metrics']['segm']['per_class_AP']
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(list(per_class),[value*100 if value is not None else 0 for value in per_class.values()],color='#4477aa')
    ax.set(xlabel='YCB-V object ID',ylabel='Mask AP@[.50:.95] (%)',ylim=(0,100),title='Instance segmentation: 900 test frames')
    fig.tight_layout(); fig.savefig(output/'mask_ap_per_object.png',dpi=180); plt.close(fig)
    lines=['# Kết quả YCBPose','','Báo cáo sinh từ artifact đo thực tế.','',
           f"- Detector: {detection['frame_count']} ảnh, {detection['gt_instances']} GT instances.",
           f"- Bbox AP@[.50:.95]: {detection['metrics']['bbox']['AP_50_95']*100:.2f}%.",
           f"- Mask AP@[.50:.95]: {detection['metrics']['segm']['AP_50_95']*100:.2f}%.",
           f"- Mask instance precision/recall @IoU0.5, score0.5: {detection['instance_mask_iou50_score50']['precision']*100:.2f}% / {detection['instance_mask_iou50_score50']['recall']*100:.2f}%.",
           f"- Pose: {pose['frames']}/900 ảnh, {pose['gt_instances']} GT instances.",
           *[f"- {label} recall@0.1d: {value:.2f}%." for label,value in zip(labels,values)],
           f"- Median inference (bỏ frame đầu): {pose['median_inference_s_excluding_first']:.3f} s/ảnh.",
           '', '![Pose recall](pose_recall.png)','','![Mask AP theo lớp](mask_ap_per_object.png)','',
           '## Định nghĩa và giới hạn','',
           'ADD dùng khoảng cách các đỉnh tương ứng, ADD-S dùng láng giềng gần nhất. ADD(-S) '
           'dùng ADD-S cho lớp có metadata đối xứng, ADD cho lớp còn lại. Mẫu số gồm tất cả GT, '
           'kể cả detector bỏ sót. Hungarian ghép cùng lớp với mask IoU >= 0.5. '
           'Tối đa 10.000 đỉnh CAD được lấy xác định. Đây là chỉ số chẩn đoán, không phải AR chính thức BOP.',
           '', 'COCO AP thu prediction từ score >=0.05 để tính precision–recall curve; '
           'pipeline pose và instance recall dùng score >=0.5. Không diễn giải AP thành tỷ lệ ảnh đúng.',
           '', 'Pretrained CosyPose SYNT+REAL chưa fine-tune. Train/val hiện trống, test không dùng để train. '
           'Windows dùng Pyrender thay Bullet/EGL; checkpoint load strict nhưng không khẳng định '
           'tái lập chính xác điểm công bố. ICP có giới hạn cập nhật. Point cloud chỉ gồm bề mặt '
           'nhìn thấy từ depth. Không có đánh giá grasp hoặc robot thực.',
           '', 'Pilot lấy từ test đã được dùng kiểm tra port renderer và xác định cấu hình ICP trước '
           'benchmark toàn tập. Chưa có validation độc lập; kết quả là đánh giá implementation '
           'mang tính thăm dò, không phải holdout hoàn toàn chưa quan sát. Trọng số không được train trên test.',
           '', '## Thí nghiệm cùng tập ảnh','']
    if experiments:
        lines+=['| Phương án | Ảnh | ADD(-S) | Nhiễm point cloud | Oracle masks |',
                '|---|---:|---:|---:|---|']
        for name,result in experiments.items():
            lines.append(f"| {name} | {result['frames']} | {result['recall']['ADD_or_S_success_0_1d']*100:.2f}% | "
                         f"{result['mean_point_contamination']*100:.2f}% | {result['oracle_masks']} |")
    else: lines+=['Thí nghiệm chưa hoàn tất.']
    lines+=['','Nhiễm point cloud là trung bình không trọng số của tỷ lệ điểm nằm ngoài visible GT mask '
            'trên các instance ghép được. Đây là proxy dựa mask chiếu, không phải đánh giá semantic '
            '3D đầy đủ. Mẫu không ghép được không vào trung bình này nhưng vẫn là thất bại của pose recall.']
    lines+=['','## Theo lớp','','| Object ID | GT | ADD(-S) recall |','|---|---:|---:|']
    for obj,result in pose['per_object'].items():
        lines.append(f"| {obj} | {result['gt']} | {result['ADD_or_S_success_0_1d']*100:.2f}% |")
    lines+=['','## Ảnh hưởng visibility','','| Visible fraction | GT | ADD(-S) recall |','|---|---:|---:|']
    for bucket,result in visibility.items():
        value=f"{result['recall']*100:.2f}%" if result['recall'] is not None else 'N/A'
        lines.append(f"| {bucket} | {result['gt']} | {value} |")
    lines+=['','Danh sách tối đa 40 ví dụ lỗi đầu tiên: [failure_examples.json](failure_examples.json). '
            'Đây là danh sách minh họa, không phải tỷ lệ lỗi theo phân phối. no_matched_pose bao gồm '
            'bỏ sót detector, sai class/mask matching hoặc không có pose hợp lệ.']
    (output/'RESULTS_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({key:value for key,value in pose.items() if key!='per_object'})
if __name__=='__main__': main()

