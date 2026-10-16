# Tính nốt T1 từ checkpoint seed 42

Tái sử dụng 6 checkpoint baseline trong `mmfi_a1_native_comparison_s42_v3`,
A1 trong `mmfi_proposed_factorial_ablation_s42_v1/full` và Clean trong
`mmfi_tables_s42_20261009_003044/clean`. Mỗi checkpoint đã train 50 epochs.
Chỉ chạy inference trên full TEST MM-Fi protocol1-s1 (33264 mẫu).

Bảng mới có BadNets, WaNet, FTrojan, FIBA, SIG, BppAttack, Proposed A1 và Clean
reference. Không chạy Blended. Các cột chính: T-MPJPE@1, Target-PCK@.2@1,
Clean MPJPE, Clean PA-MPJPE. Hai cột attack của Clean để trống vì Clean không
tham gia xếp hạng attack; diagnostic Clean + trigger A1 nằm trong JSON.

## Chạy trên MICA

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
set -o pipefail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
bash remote_linux/19_eval_a1_target_pck.sh --device cuda:0 \
  2>&1 | tee "$HOME/backdooranalog/runs/mmfi_a1_target_pck_s42_v1.log"
```

`CUDA_VISIBLE_DEVICES=3` chọn GPU vật lý 3; bên trong process nó là `cuda:0`.
Một GPU đủ cho đánh giá tuần tự 8 checkpoint. Có thể đổi GPU vật lý đang rảnh.
Không có lời gọi train hoặc optimizer trong evaluator.

Để kiểm tra metadata/đường dẫn trước, thêm `--dry-run`. Chế độ đó không đọc
checkpoint và không mở dataset/GPU. Khi chạy thật, evaluator xác minh epoch=49,
fingerprint config, trạng thái trigger, thứ tự TRAIN/TEST, và tái tính các metric
cũ. Chỉ nhận Target-PCK mới khi các metric cũ lệch tối đa **0.05 mm**. Nếu vượt
dung sai, gửi log để kiểm tra nguyên nhân; không nới ngưỡng theo kết quả mong muốn.

## Định nghĩa và dữ liệu xuất

Target-PCK dùng khoảng cách đến **attacker target** tại d=1, chỉ trên khớp [2,3].
Mỗi khớp đạt nếu sai số không quá `0.2 * ||true_pose[5]-true_pose[12]||`.
Trung bình trên mẫu và khớp, xuất phần trăm. Scale luôn lấy từ pose thật sạch.
Không dùng clean PCK để thay thế Target-PCK. Metric này không phải pose-level ASR.

Mỗi victim có diagnostic no-trigger với cùng attacker target d=1. Clean victim
được kiểm tra thêm với đúng trigger A1. Các baseline có stochastic cover được
khôi phục trigger state từ checkpoint. Các số sạch của Clean được đối chiếu với
cache cũ; Clean có cùng recipe/split hiện tại nhưng lịch sử không có hash ordered
pair IDs tại thời điểm training, điều này được ghi trong limitations.

Kết quả nằm trong thư mục mới bên trong run v3:

```text
mmfi_a1_native_comparison_s42_v3/target_pck_eval/
  a1_target_pck_complete.md
  a1_target_pck_complete.csv
  a1_target_pck_complete.json
  per_sample_badnets.npz
  ... (8 file per_sample_*.npz)
```

NPZ lưu ordered CSI/pose/frame identities, sai số từng khớp mục tiêu của input
có/không trigger và scale sạch. Mở bằng `numpy.load(path, allow_pickle=False)`.
Các record cũ không bị ghi đè. JSON completion marker chỉ được ghi sau khi đủ
8 checkpoint đạt replay checks. Cùng output đã hoàn tất sẽ từ chối chạy đè;
muốn đánh giá lại thì chọn `--outdir` mới.

## Xem bảng và đóng ZIP

Chỉ chạy sau khi evaluator báo `t1_target_pck_complete_single_seed`:

```bash
DOSE_T1_RUN="$HOME/backdooranalog/runs/mmfi_a1_native_comparison_s42_v3"
cat "$DOSE_T1_RUN/target_pck_eval/a1_target_pck_complete.md"

cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_a1_native_comparison_s42_v3_target_pck_results.zip \
  mmfi_a1_native_comparison_s42_v3 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png' '*.npz'
unzip -tq mmfi_a1_native_comparison_s42_v3_target_pck_results.zip
ls -lh mmfi_a1_native_comparison_s42_v3_target_pck_results.zip
```

ZIP chứa bảng mới, per-sample errors và kết quả v3 gốc; không cần gửi checkpoint.
Đây là phần bổ sung T1 seed 42, không có nghĩa cả 7 nhóm thực nghiệm đã xong.
