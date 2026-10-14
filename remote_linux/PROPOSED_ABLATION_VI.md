# Ablation Proposed trên MICA — seed 42, full MM-Fi

## Material Passport

- Loại: bộ thực nghiệm ablation offline CSI/HPE; không phải đánh giá RF over-the-air.
- Phạm vi: chỉ bốn bản của Proposed đã chốt. Không sửa hoặc chạy lại baseline truyền thống.
- Dữ liệu: full MM-Fi, 133056 mẫu TRAIN và 33264 mẫu đánh giá theo split hiện có.
- Trạng thái tài liệu: hướng dẫn chạy; không phải báo cáo kết quả thực nghiệm mới.

## Bốn bản chạy

| Cell | Thay đổi duy nhất so với Full |
| --- | --- |
| `full` | Multi-carrier với giới hạn peak theo toán tử Original; giữ nguyên bản đã chốt. |
| `w_o_coupling` | Hoán vị dose payload giữa các mẫu poison; giữ tập poison và phân phối dose biên. |
| `w_o_auxiliary` | Bỏ carrier phụ; dùng carrier chính `micro_dropper` chuẩn hóa unit RMS, vẫn giữ quy tắc peak của Original. |
| `w_o_peak` | Dùng cùng multi-carrier nhưng bỏ phép giới hạn peak; đo nhiễu thực tế riêng. |

Setting chung: seed victim 42, 50 epoch, SGD lr=0.001, momentum=0.9,
batch size 32, rho=0.4, epsilon=0.185; cùng victim và optimizer.
Carrier được cố định bằng method/carrier seed 42, không fit trigger hoặc
chọn tham số trên tập đánh giá. Dose đánh giá: 0, 0.2, 0.4, 0.6, 0.8, 1.
Bộ chạy này không nạp kết quả hoặc checkpoint lịch sử.

`w_o_peak` cố ý khác ngân sách peak. Vì vậy, T-MPJPE tốt hơn của bản này
không tự chứng minh thành phần nào ưu việt: phải đọc cùng `input_distortion`.
Cùng peak ceiling giữa các bản còn lại cũng không có nghĩa L2 thực tế bằng nhau.

## 1. Pull và kiểm tra plan

Chạy trong terminal Linux của MICA:

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
git log -1 --oneline

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 \
bash remote_linux/15_run_proposed_ablation.sh \
  --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Dry-run chỉ tạo/kiểm tra plan, không train và không cần đọc full dataset/GPU.
Output mặc định:

```text
~/backdooranalog/runs/mmfi_proposed_ablation_s42_v1/
```

## 2. Chạy một lần trên ba GPU

Theo snapshot người dùng cung cấp, GPU vật lý 1 đang có job khác. Chọn GPU
vật lý **0, 2, 3**. GPU 0 vẫn có API khoảng 688 MiB; chạy chung không bảo đảm
API không bị ảnh hưởng. Không dừng bất kỳ process của người khác nào.
Kiểm tra `nvidia-smi` lại ngay trước khi chạy vì snapshot có thể đã thay đổi.

```bash
cd "$HOME/backdooranalog/code"
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-prop-abl \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/15_run_proposed_ablation.sh --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_proposed_ablation_s42_v1.startup.log"'
```

Không chạy lại lệnh tạo tmux nếu session/job này vẫn đang hoạt động. Bốn cell
được xếp hàng: tối đa ba cell chạy song song; cell còn lại chạy khi có slot rảnh.

Sau `CUDA_VISIBLE_DEVICES=0,2,3`, số GPU trong PyTorch được đánh lại:

| GPU vật lý trong `nvidia-smi` | ID phải truyền cho runner |
| --- | --- |
| 0 | `cuda:0` |
| 2 | `cuda:1` |
| 3 | `cuda:2` |

**Không truyền `cuda:3` sau khi đã lọc còn ba GPU** — sẽ lỗi Invalid device id.

Mặc định runner kiểm tra plan trước khi dùng cache/resume trong chính thư mục
đầu ra này. Muốn một lượt mới hoàn toàn, chọn một `--outdir` mới. `--fresh`
không được dùng với thư mục đã có manifest từ dry-run; ví dụ:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 \
bash remote_linux/15_run_proposed_ablation.sh \
  --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 \
  --outdir "$HOME/backdooranalog/runs/mmfi_proposed_ablation_s42_repeat1" --fresh
```

## 3. Xem tiến độ từ terminal khác

```bash
DOSE_ABL_OUT="$HOME/backdooranalog/runs/mmfi_proposed_ablation_s42_v1"
tmux ls
pgrep -af '[r]un_proposed_ablation.py'
tail -n 30 "$HOME/backdooranalog/runs/mmfi_proposed_ablation_s42_v1.startup.log"
tail -n 12 "$DOSE_ABL_OUT"/*/console.log
find "$DOSE_ABL_OUT" -name eval_cache.json | wc -l
```

Theo dõi một cell:

```bash
tail -f "$DOSE_ABL_OUT/full/console.log"
```

Ctrl+C ở lệnh `tail -f` chỉ đóng phần xem log, không dừng training trong tmux.
Muốn vào session: `tmux attach -t mmfi-prop-abl`; thoát khỏi session nhưng giữ
job chạy bằng Ctrl+B rồi D. Không dùng phím mũi tên để cuộn khi Python đang chạy.

## 4. Kết quả cần lấy

Sau khi đủ bốn cache hợp lệ và runner xuất summary thành công:

```bash
cat "$DOSE_ABL_OUT/ablation_summary.md"
ls -lh "$DOSE_ABL_OUT"/ablation_summary.* \
       "$DOSE_ABL_OUT"/dose_response.* \
       "$DOSE_ABL_OUT"/input_distortion.*
```

- `ablation_summary.md/csv/json`: T-MPJPE tại d=1, MPJPE sạch và sai số điều
  khiển dose trên các khớp mục tiêu. Đọc định nghĩa metric từ metadata đã xuất.
- `dose_response.csv/json` và `dose_response.png`: đủ sáu dose của mỗi cell.
- `input_distortion.csv/json`: nhiễu digital trên cùng các input/dose, gồm
  peak và L2 thực tế; mặc định 256 mẫu, không phải bằng chứng RF stealthiness.
- Manifest resolved, config, log và `eval_cache.json` từng cell: provenance.

Nếu đủ cache nhưng cần xuất lại báo cáo, không train thêm:

```bash
bash remote_linux/15_run_proposed_ablation.sh --seed 42 --summary-only
```

Đóng gói nhẹ, không lấy checkpoint/model hay raw data:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_proposed_ablation_s42_v1_results.zip \
  mmfi_proposed_ablation_s42_v1 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png'
ls -lh mmfi_proposed_ablation_s42_v1_results.zip
```

Tải từ PowerShell trên máy Admin (không chạy lệnh này trên MICA):

```powershell
scp mica-server:~/backdooranalog/runs/mmfi_proposed_ablation_s42_v1_results.zip "$env:USERPROFILE\Downloads\"
```

## Ba seed sau

Trước mắt chỉ seed 42. Sau khi kiểm tra bộ này, thêm `--seed 43` và `--seed 44`
vào cùng lệnh; mỗi seed tự có output riêng `mmfi_proposed_ablation_s43_v1`
hoặc `mmfi_proposed_ablation_s44_v1`. Carrier/method seed vẫn giữ 42: đây là
lặp victim-training seed, không chọn lại phương pháp theo từng seed.
Không chạy nhiều controller cùng lúc dùng chung output hoặc cùng các GPU.

Chỉ một seed hiện tại: không có mean/std nhiều seed và không kết luận superiority
thống kê. Thực nghiệm mới có thể không lặp đúng từng chữ số của run lịch sử.
