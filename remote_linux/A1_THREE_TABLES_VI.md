# MM-Fi A1: chỉ T1, T2, T3 — seeds 42, 43, 44

## Phạm vi

- Chỉ chạy **comparison (T1)**, **dose × poison (T2)** và **ablation (T3)**.
- Không mở thêm defense, stealthiness, backbone, split hoặc dataset khác.
- Baseline giữ nguyên toán tử và tham số nguồn của comparison v3 đã chạy:
  BadNets-CSI, WaNet-CSI, FTrojan-CSI, FIBA-CSI, SIG-CSI và BppAttack-CSI.
- Không cải tiến baseline, không tối ưu thêm Proposed và không áp dụng peak
  projection mới. Đây là xác nhận lại recipe đã chốt ở ba seed.
- A1 đầy đủ là **một victim cho mỗi seed** được dùng chung giữa T1, T2 tại rho=0.1
  và T3. Không train A1 ba lần cho ba bảng.
- Profile: `mmfi_a1_three_tables_v1`; manifest: `three_tables.resolved.json`.
- Output mặc định: `runs/mmfi_a1_three_tables_s42_43_44_v1`.
- Đây là hướng dẫn thực thi, không xác nhận code đã push hoặc job đã khởi chạy.
  Kiểm thử local/dry-run không phải kết quả benchmark MM-Fi.

## Kiểm thử local (2026-10-16)

- 335 tests đạt: các module mới, evaluator, export mean/SD, launcher và hồi quy
  comparison/ablation/Target-PCK/DCE cũ. Cảnh báo pin-memory trên CPU là dự kiến.
- Có bài test route 50 epochs với model/dữ liệu CPU nhỏ cho A1 và clean,
  load lại checkpoint và xác nhận không train lại khi cache còn hợp lệ.
- Đã đối chiếu config và poison manifest của 13 nguồn seed 42 trong các ZIP
  được cung cấp; không có thay đổi toán tử/recipe baseline. Checkpoint thực trên
  MICA và dataset được kiểm tra tiếp ở preflight, không giả định đã kiểm thử GPU.
- Chưa có kết quả benchmark seed 43/44 cho tới khi người dùng thực thi trên MICA.

## Số victim cần train

Mỗi seed có 13 cấu hình khoa học duy nhất:

| Nhóm | Số victim/seed | Cấu hình |
| --- | ---: | --- |
| Baseline T1 | 6 | BadNets, WaNet, FTrojan, FIBA, SIG, BppAttack |
| Ablation T3 | 4 | A1, A2, A3, A4; A1 dùng chung T1/T2 |
| A1 ở poison-rate khác | 2 | rho=0.2, rho=0.4 |
| Clean control | 1 | Không poison |

Mặc định **reuse 13 victim seed 42 đã chạy**, train mới 13 victim seed 43 và 13
victim seed 44: **26 victim mới**, không phải 39. Runner kiểm tra source config,
manifest, cache/audit và checkpoint trước khi sử dụng. Không sửa file nguồn cũ,
không nhập con số điền tay, không lấy kết quả từ các run có peak budget khác.

| Đối số override | Nguồn seed 42 mặc định trong `data-home/runs` |
| --- | --- |
| `--seed42-comparison` | `mmfi_a1_native_comparison_s42_v3` — sáu baseline |
| `--seed42-ablation` | `mmfi_proposed_factorial_ablation_s42_v1` — full, w_o_coupling, w_o_auxiliary, w_o_both |
| `--seed42-dose-poison` | `mmfi_a1_dose_poison_s42_v1` — rho0p2, rho0p4 |
| `--seed42-clean` | `mmfi_tables_s42_20261009_003044/clean` — clean control |

Giữ nguyên checkpoints seed 42 trên server. ZIP nhẹ chỉ có bảng/cache **không thay
thế checkpoint** khi cần tính Target-PCK bổ sung. Cache seed 42 cũ không có
Target-PCK phải được đánh giá bổ sung trên checkpoint, không đổi tên clean PCK
hoặc suy Target-PCK từ T-MPJPE. Đây là inference, không train lại victim.

Chỉ khi thực sự muốn train lại cả seed 42 mới dùng `--retrain-seed42`, khi đó có
**39 victim mới**. Không dùng cờ này trong lệnh mặc định. Nếu reference không
đầy đủ/không khớp, xác minh nguồn trước; không tự động chuyển sang retrain.

## Recipe chung đã chốt

Full MM-Fi protocol1-s1: train ratio 0.8, split seed 0; 133056 TRAIN / 33264 đánh
giá, không cap mẫu; HPELi từ đầu; 50 epochs; SGD lr=0.001, momentum=0.9,
weight decay=0, batch size 32. T1/T3 dùng rho=0.1; T2 dùng rho=0.1, 0.2, 0.4.
Trigger/payload dose U(0.2,1); pivot 1, target joints [2,3], rotation Z 40*d độ.
Native operator giữ source settings; không match realized Linf/L2 hoặc RF budget.

Giữ cả những khác biệt native đã công khai ở comparison v3: cover/negative của
WaNet/FIBA/Bpp, loader policy và task adaptation từ classification sang CSI/HPE.
SIG giữ delta=40, f=6; Bpp giữ squeeze_num=8, no dithering, neg_ratio=0.1 của
recipe v3; không chuyển sang setting paper/README khác giữa ba seed.
Không tuyên bố toàn bộ minibatch/cover RNG giống hệt nhau hoặc tái hiện đầy đủ
pipeline classification nguồn. Không tìm hyperparameter trên tập đánh giá.

## Ba bảng và metric

### T1 — Comparison

Cột chính dùng tên gọn: **T-MPJPE ↓**, **Target-PCK ↑**, **Clean MPJPE ↓**,
**Clean PA-MPJPE ↓**. Hai metric tấn công đo tại d=1 trên target joints [2,3].
Target-PCK dùng pose mục tiêu của attacker và ngưỡng tương đối
`0.2 * ||clean_gt[5] - clean_gt[12]||`, cùng clean-GT scale cho mọi phương pháp.
Không phải 20 mm, clean PCK hoặc tiêu chí thành công toàn bộ pose.
T-MPJPE/MPJPE/PA-MPJPE dùng mm; Target-PCK dùng %.

### T2 — Dose × Poison

Ma trận **hàng là dose** d=0, 0.2, 0.4, 0.6, 0.8, 1; **cột là rho** 0.1, 0.2,
0.4. Mỗi ô là T-MPJPE theo cùng pose target tại dose đó. Một victim cho mỗi rho,
đánh giá sáu dose; không train sáu victim theo sáu dose. Có phần tóm tắt utility
sạch và Dose-control error theo rho; không tạo thêm cấu hình poison-rate.

### T3 — Ablation

| Cấu hình | Auxiliary carrier | Dose coupling |
| --- | :---: | :---: |
| A1 (Full) | ✓ | ✓ |
| A2 | ✓ | ✗ |
| A3 | ✗ | ✓ |
| A4 | ✗ | ✗ |

Cột kết quả: **T-MPJPE ↓**, **Dose-control error ↓**, **Clean MPJPE ↓**.
Không peak bound ở cả bốn nhánh. Dose-control error là sai số displacement trên
target joints so với displacement target, trung bình năm dose dương; không phải
Spearman hoặc Schedule MAD. A1 là đúng victim dùng trong T1/T2 của seed đó.

Mọi số liệu chính báo **mean ± sample SD (ddof=1) trên ba seed 42, 43, 44**.
Trước hết gộp các mẫu thành metric trong từng seed, sau đó mới tính mean/SD từ
ba giá trị seed. Không coi frame/joint là các seed độc lập, không dùng SD giữa
frame thay SD giữa seed. Một seed thiếu hoặc không hợp lệ không được giả làm đủ
ba seed; không sao chép kết quả seed 42 sang seed khác.

## 1. Pull và dry-run

Chạy sau khi đã xác nhận code được push, trên terminal MICA:

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
git log -1 --oneline
nvidia-smi

bash remote_linux/23_run_a1_three_tables.sh \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Dry-run dựng/kiểm tra plan, không train hoặc dùng GPU. Không coi dry-run là đã
xác minh checkpoint/dataset trên server. Có thể chỉ rõ data-home/Python có quote:

```bash
bash remote_linux/23_run_a1_three_tables.sh \
  --python "$HOME/backdooranalog/envs/dose-backdoor/bin/python" \
  --data-home "$HOME/backdooranalog" --dry-run
```

## 2. Chạy ba GPU

Phải xem `nvidia-smi` mới và có quyền dùng GPU trước. GPU vật lý 0, 2, 3 dưới đây
chỉ là ví dụ mapping, không khẳng định chúng đang rảnh. Không dừng process của
người khác. Ít VRAM/utilization tại một thời điểm không bảo đảm không ảnh hưởng.

```bash
cd "$HOME/backdooranalog/code"
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-a1-three-s42-43-44 \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/23_run_a1_three_tables.sh --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_a1_three_tables_s42_43_44_v1.startup.log"'
```

Sau mapping GPU vật lý 0, 2, 3, truyền ID logic `cuda:0 cuda:1 cuda:2`, không
truyền `cuda:3`. Mỗi GPU train một victim độc lập, tối đa ba victim song song;
các cell còn lại chờ slot. Không mở controller thứ hai cho cùng output đang chạy.
Lệnh mặc định không retrain seed 42, không train thêm ngoài T1/T2/T3.

`--fresh` từ chối thư mục không rỗng. Sau dry-run đã có manifest, chạy bình
thường, không thêm `--fresh` vào cùng output. Muốn run tách biệt, chọn `--outdir`
mới. Chạy tiếp sau gián đoạn chỉ dùng cache/checkpoint đúng plan trong output.

Có thể chọn cell khoa học tường minh như `--cells s43__badnets`; không dùng
subset chưa đủ để xuất bảng mean±SD như thể mọi cell đã hoàn tất.

## 3. Tiến độ và export

```bash
DOSE_THREE_OUT="$HOME/backdooranalog/runs/mmfi_a1_three_tables_s42_43_44_v1"
tmux capture-pane -pt mmfi-a1-three-s42-43-44 -S -35
pgrep -af '[r]un_a1_three_tables.py'
tail -n 12 "$DOSE_THREE_OUT"/seed*/*/console.log
bash remote_linux/23_run_a1_three_tables.sh --status
find "$DOSE_THREE_OUT" -name eval_cache.json | wc -l
ls -lh "$DOSE_THREE_OUT"/*.md "$DOSE_THREE_OUT"/*.csv
```

Đếm cache chỉ là chẩn đoán: các seed-42 victim được reuse từ ngoài output nên
không suy số cell hoàn tất chỉ bằng số file trong output mới. Cần controller báo
hoàn tất, `--status` báo **39/39** và xuất đủ ba bảng. `--status` chỉ đọc artifact,
không dùng GPU hoặc train. Seed 42 xuất `evaluation.json` vào output mới, nhưng
checkpoint vẫn nằm tại folder nguồn. Khi chỉ có một số cell xong, không coi bảng
partial là kết quả đủ ba seed.

Evaluation xuất riêng vào `evaluations/s42__badnets/` (và tương tự cho từng
seed/cell); không nằm trong checkpoint tree. `seed42/badnets/console.log` chỉ
là log mới của cell tham chiếu. Thư mục seed 43/44 chứa checkpoint mới.

Artifact chính: `table1_comparison.*`, `table2_dose_poison.*`,
`table2_poison_summary.csv`, `table3_ablation.*`, dose curves và
`per_seed_metrics_s42.csv`, `per_seed_metrics_s43.csv`, `per_seed_metrics_s44.csv`.
File `three_main_tables.json` được ghi sau cùng khi aggregate hoàn tất.

Xuất lại bảng từ các kết quả đầy đủ, không train:

```bash
bash remote_linux/23_run_a1_three_tables.sh --export-only
```

Ctrl+C tại `tail -f` chỉ dừng xem log, không dừng training trong tmux.

## 4. Đóng ZIP và tải về Admin

Trên MICA, đóng artifacts nhẹ sau khi chạy xong. Giữ nguyên checkpoint trên
server, không đưa `.pt` hoặc raw dataset vào ZIP:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_a1_three_tables_s42_43_44_v1_results.zip \
  mmfi_a1_three_tables_s42_43_44_v1 \
  -i '*.json' '*.csv' '*.md' '*.tex' '*.log' '*.yaml' '*.yml' '*.png' '*.npz'
ls -lh mmfi_a1_three_tables_s42_43_44_v1_results.zip
```

Giữ cả các artifact nguồn seed 42 đã dùng để đối chiếu reference; ZIP mới không
phải bản sao mọi checkpoint/reference gốc. `.npz` là artifact số liệu được giữ
để kiểm tra khi runner có xuất, không phải lý do đóng raw dataset.

Trên PowerShell máy Admin, không chạy lệnh này trong SSH MICA:

```powershell
scp "mica-server:~/backdooranalog/runs/mmfi_a1_three_tables_s42_43_44_v1_results.zip" "$env:USERPROFILE\Downloads\"
```
