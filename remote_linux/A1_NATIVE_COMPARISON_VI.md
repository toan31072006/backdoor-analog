# Comparison MM-Fi với A1 đã chạy — seed 42

## Phạm vi

Profile `mmfi_a1_native_comparison_v3` tạo output mới ở
`runs/mmfi_a1_native_comparison_s42_v3`. Bản v3 thêm hai baseline từ danh sách
BackdoorBench: `sig_source` và `bpp_source`. Sáu baseline được train mới là
`badnets`, `wanet_source`, `ftrojan`, `fiba`, `sig_source`, `bpp_source`;
Blended không thuộc lượt comparison này theo yêu cầu. A1 là tham
chiếu bất biến từ `runs/mmfi_proposed_factorial_ablation_s42_v1/full`: runner
kiểm tra config, input audit và cache trước khi dùng; Không train lại A1 hoặc
A2–A4. Không nhập kết quả baseline từ các run cũ có peak projection.

Đây là bảng phương pháp **được chọn**, không phải tất cả backdoor attack. Kết
quả Blended cũ vẫn được giữ; bảng thiếu Blended không chứng minh A1 hơn Blended.
Không lấy độ nhiễu từ một run khác ghép vào bảng này.

## Setting chung và giới hạn công bằng

Từng baseline dùng cùng split MM-Fi P1S1 (133056 TRAIN / 33264 TEST), HPELi
khởi tạo mới, seed 42, 50 epochs, SGD lr=0.001 momentum=0.9, batch size 32,
rho=0.1, cùng chỉ số poison và paired trigger/payload dose U(0.2,1). Pose target:
pivot 1, joints [2,3], quay Z 40*d độ. Toán tử đầu vào và các cover giữ
đặc trưng của phương pháp nguồn; không chỉnh chúng để giúp hoặc làm yếu một
baseline. Không dùng shared-peak projection. **Không match peak thực tế hoặc L2**
giữa các phương pháp: đây là so sánh native-operator, không phải equal-noise
budget. Nhiễu CSI số không chứng minh RF stealthiness hay khả thi over-the-air.

| Baseline | Adapt sang CSI/HPE |
| --- | --- |
| BadNets-CSI | Masked patch từ BackdoorBench, vùng 3×3 subcarrier–packet ở góc cuối; độ đục theo dose. Kích thước patch là cấu hình CSI đã khai báo, không phải mặc định chung của paper. |
| WaNet-CSI | Grid warp gốc với cường độ theo dose; 20% clean-label covers. |
| FTrojan-CSI | Hệ số DCT và magnitude nguồn; ánh xạ rank sang CSI chữ nhật, không trộn RGB/YUV. |
| FIBA-CSI | FFT amplitude blend, alpha=0.15, beta=0.1; CSI TRAIN làm key/donor, giữ cross samples 10%. |
| SIG-CSI | Sinusoid theo trục packet; tham số mặc định BackdoorBench `delta=40`, `f=6` được map vào CSI 10 packet và nhân dose. Đây là **dirty-label pose adaptation**, không giữ clean-label target-class threat model của SIG gốc. f=6 trên lưới 10 packet alias thành tần số âm f=4, không phải tần số Hz. |
| BppAttack-CSI | Lượng tử hóa với cấu hình BackdoorBench `squeeze_num=8`, `neg_ratio=0.1`; residual-negative covers lấy từ TRAIN CSI, dose nội suy residual. Đây chỉ là **CSI/pose adaptation của nhánh quantization**, **không phải tái hiện đầy đủ** PGD/supervised-contrastive của paper. TRAIN donor được lấy lazy có hoàn lại, khác residual bank ảnh tăng cường lấy theo batch ở nguồn. |

SIG/BppAttack đều có trong [BackdoorBench](https://github.com/SCLBD/BackdoorBench),
nhưng mã ở đây là implementation độc lập của toán tử BDB trên CSI, không vendored
nguyên pipeline ảnh/phân loại. BppAttack trong BDB dùng ordinary classification
loss; tất cả các arm ở đây dùng cùng HPE ordinary ERM. Việc thêm dose/pose và
thay ảnh bằng CSI là task adaptation, không phải một thí nghiệm tái hiện nguyên
bài báo nguồn. Version nguồn và hash file được khóa trong
`ATKBackd/attack/static_source_baselines.py`.
Riêng BppAttack, BackdoorBench chọn lại vị trí mẫu negative recipient theo
minibatch mỗi epoch; `PoisonedDataset` ở đây cố định 10% recipient cover trong
suốt 50 epochs, chỉ TRAIN donor sinh residual thay đổi giữa các lần lấy mẫu.
Vì vậy phân bố cover không đồng nhất với pipeline BackdoorBench.

WaNet, FIBA và BppAttack dùng `num_workers=0` để giữ RNG cover khi checkpoint/resume;
BadNets, FTrojan và SIG giữ worker policy của A1. Vì vậy minibatch/cover RNG không
được tuyên bố giống hệt nhau. Số đầu vào bị sửa cũng khác nhau và có audit.

Ba metric chính trong `a1_native_comparison.md` là T-MPJPE@1, MPJPE sạch và
PA-MPJPE sạch, đơn vị mm, càng nhỏ càng tốt. Sáu dose 0,0.2,…,1 xuất riêng
trong CSV/JSON. Clean PCK@0.2 trong CSV nếu có chỉ là chẩn đoán utility sạch,
threshold tương đối `0.2 * ||gt[5] - gt[12]||`, không phải 20 mm hay Target-PCK.
Target-PCK chưa được tính; không gọi clean PCK là ASR. Một seed không cho phép
kết luận về độ bất định hay ưu thế thống kê.

## Chạy trên MICA sau khi xác nhận commit mới đã push

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
git log -1 --oneline
nvidia-smi

bash remote_linux/18_run_a1_native_comparison.sh \
  --a1-reference "$HOME/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1" \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Dry-run không đọc dataset/reference, không dùng CUDA và không train. Run thật
sẽ validate reference A1 trước khi train; thiếu reference thì dừng, không tự
train lại. Mặc định `--data-home` là `~/backdooranalog` và Python là
`~/backdooranalog/envs/dose-backdoor/bin/python`; có thể chỉ định `--python`
hoặc `--data-home` khác khi cần.

Chỉ dùng mapping ví dụ dưới đây nếu `nvidia-smi` mới cho thấy GPU 0, 2, 3
có thể sử dụng; không dừng job của người khác. GPU 1 đã có job trong snapshot
trước, nhưng snapshot đó không đảm bảo trạng thái hiện tại.

```bash
cd "$HOME/backdooranalog/code"
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-a1-compare-s42 \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/18_run_a1_native_comparison.sh --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_a1_native_comparison_s42_v3.startup.log"'
```

Sau `CUDA_VISIBLE_DEVICES=0,2,3`, ID logic là `cuda:0`, `cuda:1`, `cuda:2`.
Ba baseline chạy song song tối đa; ba baseline còn lại chờ slot. Không chạy
`cuda:3` trong mapping này. Không tạo controller thứ hai cho cùng output.
`tmux` có thể in cảnh báo `libtinfo` mà vẫn chạy; kiểm tra session/log để biết.

```bash
DOSE_A1_OUT="$HOME/backdooranalog/runs/mmfi_a1_native_comparison_s42_v3"
tmux capture-pane -pt mmfi-a1-compare-s42 -S -30
pgrep -af '[r]un_a1_native_comparison.py'
tail -n 12 "$DOSE_A1_OUT"/*/console.log
find "$DOSE_A1_OUT" -name eval_cache.json | wc -l
cat "$DOSE_A1_OUT/a1_native_comparison.md"
```

Tiêu chí: sáu cache baseline hợp lệ, summary bảy hàng (sáu baseline + A1).
Đếm file đơn thuần chưa chứng minh config/cache/log khớp nhau. Ctrl+C ở
`tail -f` chỉ dừng xem log. Có thể chạy `--summary-only` để xuất lại bảng
mà không train. `--fresh` từ chối output không rỗng; sau dry-run đã có
manifest thì chạy bình thường, không thêm `--fresh` cùng output. Nếu muốn
khởi tạo từ đầu với plan khác, dùng `--outdir` mới; không ghi đè output cũ.

## ZIP và tải về

Trên MICA, chỉ đóng artifact nhẹ, không checkpoint/raw dataset:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_a1_native_comparison_s42_v3_results.zip \
  mmfi_a1_native_comparison_s42_v3 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png'
ls -lh mmfi_a1_native_comparison_s42_v3_results.zip
```

Trên PowerShell máy Admin, không chạy lệnh này trong SSH MICA:

```powershell
scp "mica-server:~/backdooranalog/runs/mmfi_a1_native_comparison_s42_v3_results.zip" "$env:USERPROFILE\Downloads\"
```
