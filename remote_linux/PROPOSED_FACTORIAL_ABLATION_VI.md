# Ablation factorial Proposed — seed 42, rho=0.1, full MM-Fi

## Material Passport

- Loại: bộ ablation offline CSI/HPE, không phải đánh giá RF over-the-air.
- Phạm vi: bốn tổ hợp của hai thành phần Proposed: auxiliary carrier và dose coupling. Không sửa hoặc chạy lại baseline truyền thống.
- Dữ liệu: full MM-Fi protocol1-s1 hiện có, 133056 mẫu TRAIN và 33264 mẫu đánh giá; không giới hạn số mẫu.
- Profile mới: `mmfi_proposed_factorial_ablation_v1`, mặc định seed 42, rho=0.1, 50 epoch.
- Trạng thái: hướng dẫn thực thi, không phải báo cáo kết quả đã chạy. Việc kiểm thử local không tạo kết quả benchmark MICA.

## Bốn tổ hợp

| Cell | Auxiliary carrier | Dose coupling | Toán tử |
| --- | --- | --- | --- |
| `full` | Có | Paired | `md_multicarrier`, không peak projection |
| `w_o_coupling` | Có | Shuffled | `md_multicarrier`, không peak projection |
| `w_o_auxiliary` | Không | Paired | `micro_dropper` Original unit RMS |
| `w_o_both` | Không | Shuffled | `micro_dropper` Original unit RMS |

Full mới **không có peak bound**. Shuffled chỉ hoán vị dose payload giữa các mẫu
poison; giữ cùng tập poison, trigger dose và phân phối dose biên. Bỏ auxiliary
khôi phục carrier chính Original unit RMS; L2 thực tế không được match.
Hàng `w_o_both` cho phép xem cả tổ hợp bỏ hai thành phần, không chỉ từng thành
phần riêng. Không fit trigger hoặc lựa chọn tham số trên tập đánh giá.

Setting chung: full 50 epoch; HPELi khởi tạo mới; seed victim 42; SGD lr=0.001,
momentum=0.9, weight decay=0, batch size 32; rho=0.1; epsilon=0.185. RNG của
auxiliary carrier giữ seed 42; pattern Original dùng seed thí nghiệm. Trong
lượt seed 42 này cả hai đều là 42. Giữ split, optimizer và pose mục tiêu như hợp đồng ablation.
TRAIN dùng trigger dose U(0.2,1); payload ghép paired hoặc shuffled tùy hàng.
Mỗi victim đánh giá tại sáu dose 0, 0.2, 0.4, 0.6, 0.8, 1; không train theo từng dose.

`--factorial` là profile độc lập với `--no-peak` của launcher 16. Bộ ba bản cũ
rho=0.4 vẫn giữ nguyên tại `mmfi_proposed_no_peak_ablation_s42_v1`; không ghi đè,
nhập cache hoặc lấy checkpoint cũ làm kết quả cho profile rho=0.1 mới.

## Metric

Bảng Markdown chính giữ **T-MPJPE tại d=1**, **Dose-control error** trung bình
năm dose dương và **MPJPE sạch**. Các sai số dùng mm, càng nhỏ càng tốt.
Dose-control error so sánh `(pred_d - pred_0)` với `(target_d - true_0)` trên
khớp mục tiêu, không phải Spearman hoặc Schedule MAD chuẩn hóa endpoint.

CSV/JSON giữ thêm PA-MPJPE sạch và **Clean PCK@0.2** tương thích bộ đánh giá
hiện có. PCK này đo dự đoán trên **CSI không trigger** so với pose thật sạch,
trên tất cả khớp: đúng khi sai số không lớn hơn `0.2 * ||gt[5] - gt[12]||`.
Ngưỡng tương đối này **không phải 20 mm**; kết quả xuất ra là %.
Đây **không phải Target-PCK**, attack PCK hay attack success rate. Target-PCK
chưa được triển khai trong bộ chạy này và không cần cho ba cột ablation chính.
Không đổi tên clean PCK thành Target-PCK để trình bày kết quả.

Linf, L2 tương đối và SNR digital vẫn được xuất riêng trên cùng các mẫu/dose,
mặc định 256 mẫu đánh giá. Original chỉ là mốc tham chiếu nhiễu: **không giới
hạn peak và không match L2** giữa các hàng. Vượt peak Original là số đo chẩn
đoán, không phải lỗi budget. Epsilon và định nghĩa toán tử vẫn được giữ nguyên.
Các số đo đầu vào digital không chứng minh RF stealthiness hoặc over-the-air.

## 1. Pull và kiểm tra plan

Chạy trên terminal Linux MICA sau khi code mới đã được push:

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
git log -1 --oneline
nvidia-smi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 \
bash remote_linux/17_run_proposed_factorial_ablation.sh \
  --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

GPU vật lý 0, 2, 3 chỉ là **ví dụ**; phải xem `nvidia-smi` mới trước khi chọn.
Tài liệu không khẳng định GPU nào đang rảnh. Không dừng process của người khác;
ít VRAM/utilization trong một snapshot không bảo đảm chia sẻ không ảnh hưởng.

Dry-run chỉ tạo/kiểm tra manifest, không đọc dataset hoặc dùng GPU. Output mới:

```text
~/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1/
```

Sau khi đặt `CUDA_VISIBLE_DEVICES=0,2,3`, ID logic phải là `cuda:0`, `cuda:1`,
`cuda:2` tương ứng GPU vật lý 0, 2, 3; không truyền `cuda:3`.

## 2. Một seed, tối đa ba cell song song

Ví dụ này chỉ dùng sau khi đã kiểm tra GPU và chọn đúng ba GPU được phép dùng:

```bash
cd "$HOME/backdooranalog/code"
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-prop-factorial-s42 \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/17_run_proposed_factorial_ablation.sh --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1.startup.log"'
```

Mỗi cell train độc lập trên một GPU. Tối đa ba cell song song; cell thứ tư chờ
slot trống. `--num-workers 4` nghĩa là bốn DataLoader worker **mỗi cell**.
Không tạo controller thứ hai nếu session/job này vẫn đang chạy.

Mặc định chỉ resume/cache từ chính output có plan hợp lệ. Muốn chạy mới hoàn
toàn, dùng `--outdir` mới. `--fresh` từ chối thư mục không rỗng: sau dry-run đã
có manifest thì chạy bình thường, **không thêm `--fresh`** vào cùng output.

## 3. Kiểm tra tiến độ và kết quả

```bash
DOSE_ABL_OUT="$HOME/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1"
tmux ls
pgrep -af '[r]un_proposed_ablation.py'
tail -n 30 "$HOME/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1.startup.log"
tail -n 12 "$DOSE_ABL_OUT"/*/console.log
find "$DOSE_ABL_OUT" -name eval_cache.json | wc -l
cat "$DOSE_ABL_OUT/ablation_summary.md"
```

Cần đủ **bốn cache hợp lệ** và summary xuất thành công. Đếm file đơn thuần
không thay cho đối chiếu config/cache/log; `pgrep` có thể hiện profile khác.
Ctrl+C ở `tail -f` chỉ dừng xem log, không dừng training trong tmux.

Xuất lại summary từ cache của chính profile mới, không train thêm:

```bash
bash remote_linux/17_run_proposed_factorial_ablation.sh --seed 42 --summary-only
```

File cần giữ: `ablation_summary.md/csv/json`, `dose_response.csv/json/png`,
`input_distortion.csv/json`, manifest resolved, config, poison manifest,
input audit, log và `eval_cache.json` từng cell.

## 4. ZIP nhẹ và tải về Admin

Trên MICA, không đóng checkpoint/model hoặc raw data:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_proposed_factorial_ablation_s42_v1_results.zip \
  mmfi_proposed_factorial_ablation_s42_v1 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png'
ls -lh mmfi_proposed_factorial_ablation_s42_v1_results.zip
```

Trên PowerShell máy Admin, không gõ lệnh này trong SSH MICA:

```powershell
scp "mica-server:~/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1_results.zip" "$env:USERPROFILE\Downloads\"
```

Trước mắt chỉ seed 42. Khi được yêu cầu chạy thêm, dùng `--seed 43` hoặc
`--seed 44`, mỗi seed có output mới tương ứng. RNG của auxiliary carrier vẫn
42, nhưng pattern Original đi theo seed thí nghiệm: toàn bộ trigger không
được đóng băng giữa các seed.
Một seed không cho mean/std giữa seed hay kết luận hơn có ý nghĩa thống kê.
Việc chọn cấu trúc mới sau các run trước là phát triển phương pháp, không phải
đánh giá mù độc lập hoặc lý do bỏ các kết quả không thuận lợi.
