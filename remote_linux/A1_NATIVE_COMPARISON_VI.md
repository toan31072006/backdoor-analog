# Comparison MM-Fi với A1 đã chạy — seed 42

## Phạm vi và trạng thái

- Profile: `mmfi_a1_native_comparison_v1`.
- Output mới: `runs/mmfi_a1_native_comparison_s42_v1`.
- Train mới đúng sáu baseline: `badnets`, `wanet_source`, `ftrojan`, `fiba`,
  `sig_source`, `bpp_source`.
- A1 là hàng tham chiếu **bất biến** từ
  `runs/mmfi_proposed_factorial_ablation_s42_v1/full`. Runner kiểm tra manifest,
  config, input audit và cache trước khi sử dụng. Không train lại A1, không chạy
  lại A2–A4, không sửa file nguồn tham chiếu.
- Không nhập kết quả baseline từ các run cũ có peak projection. Các run cũ được
  giữ nguyên, không bị xóa hoặc ghi đè.
- Blended không thuộc lượt comparison được chọn này theo yêu cầu; kết quả cũ
  vẫn giữ nguyên. Không dùng bảng thiếu Blended để khẳng định A1 vượt nó hoặc
  vượt mọi backdoor attack.
- Đây là hướng dẫn chạy, không phải báo cáo đã hoàn tất. Kiểm thử local/dry-run
  không tạo kết quả MM-Fi. Tài liệu không xác nhận SSH đã kết nối, code đã push
  hoặc job đã khởi chạy; phải kiểm tra các trạng thái này riêng.

## Setting chung và các khác biệt được công khai

Giữ nguyên setting của A1: seed 42; HPELi khởi tạo mới; full MM-Fi P1S1,
train ratio 0.8 và split seed 0; 133056 mẫu TRAIN / 33264 mẫu đánh giá;
50 epochs; SGD lr=0.001, momentum=0.9, weight decay=0, batch size 32;
rho=0.1; paired trigger/payload dose U(0.2,1); pivot 1, target joints [2,3],
rotation Z 40*d độ. Không đổi backbone, split, optimizer hoặc mục tiêu để giúp
một baseline. Epsilon=0.185 được giữ ở những toán tử có tham số này; các baseline
khác giữ cường độ toán tử nguồn đã khai báo, không biến epsilon thành tham số
chung không có trong bài gốc.

Không áp dụng `comparison_peak_budget` hoặc shared-peak projection cho baseline
mới. **Không match peak thực tế hoặc L2** giữa các phương pháp. A1 cũng không có
peak bound. Đây là so sánh các toán tử native được chuyển sang bài toán CSI/HPE,
không phải so sánh cùng một ngân sách nhiễu. Runner này xuất accuracy/dose/audit,
chưa xuất bảng input-distortion cho bộ sáu baseline mới; không lấy số nhiễu
từ run peak cũ ghép vào hoặc tuyên bố RF stealthiness/over-the-air.

| Baseline | Chuyển bài toán, không cải tiến toán tử |
| --- | --- |
| BadNets-CSI | Toán tử masked-patch BackdoorBench với vùng 3×3 subcarrier–packet ở góc cuối; độ đục theo dose; giữ pose target chung. Kích thước 3×3 là cấu hình đã khai báo cho CSI, không phải mặc định chung của paper gốc. |
| WaNet-CSI | Grid warp nguồn, biên độ theo dose; giữ 20% clean-label native covers. |
| FTrojan-CSI | Hệ số DCT và mức nguồn; ánh xạ rank sang lưới CSI chữ nhật, không trộn RGB/YUV. |
| FIBA-CSI | FFT amplitude blend nguồn, alpha=0.15, beta=0.1; CSI TRAIN làm key/donor, giữ native cross samples 10%. |
| SIG-CSI | Sinusoid mức 20/255, frequency=6 theo setting GTSRB trong paper; toán tử tham chiếu BackdoorBench. Dose scale và pose target là task adaptation. Đây là code bên thứ ba, không tuyên bố code tác giả. Bản này dùng dirty-label pose target, không tuyên bố tái lập clean-label classification của bài gốc. |
| Bpp-CSI | Biến thể CE/residual của code tác giả: mức lượng tử hóa 32 theo README, không dithering, 20% native TRAIN-donor negatives. Không tuyên bố tái lập biến thể paper PGD/SupCon. |

Mọi baseline dùng cùng tập poison/dose theo plan chung. Cover/negative chỉ dùng
TRAIN và giữ nhãn pose sạch; **số đầu vào bị sửa không giống nhau**. WaNet, FIBA
và Bpp dùng num_workers=0 để giữ native cover/negative RNG khi checkpoint/resume;
các phương pháp còn lại dùng `--num-workers` yêu cầu, mặc định 4 với persistent
workers như A1. Vì vậy không tuyên bố toàn bộ RNG theo batch/cover giống hệt nhau.
Không tìm hyperparameter trên tập đánh giá, không thêm học trigger, không chọn
baseline theo kết quả chạy mới.

Nguồn hai phương pháp mới:

- SIG: [paper ICIP 2019](https://arxiv.org/pdf/1902.11237),
  [toán tử BackdoorBench đã pin commit](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/utils/bd_img_transform/sig.py).
  Paper dùng mức 20 trên scale 255 ở GTSRB; mặc định 40 của code nguồn không được
  gọi nhầm là setting 20. Không chọn mức này bằng kết quả TEST.
- BppAttack: [paper CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.html),
  [repo tác giả đã pin commit](https://github.com/RU-System-Software-and-Security/BppAttack/tree/f6b6580d53b2f2bca8e92cfe83aae7d68332ff4c).
  Chọn 32 theo README (5 bit), không lấy mặc định 8 trong config. Dose trung gian
  dùng `x + d * (Q(x) - x)` là khai báo chuyển bài toán; không phải tối ưu trigger mới.
  Giữ cơ chế quantization/residual của nhánh code CE; loss classification CE được
  thay bằng MPJPE của victim HPE chung, không giữ loss phân loại cho hồi quy pose.

SIG giữ nguyên tham số f=6 nhưng MM-Fi chỉ có 10 packet: chuỗi số bị aliasing
thành âm của f=4. Đây không phải 6 Hz hay cùng tần số vật lý với dữ liệu ảnh;
không đổi f sau khi xem TEST. Bpp lấy donor TRAIN chưa augmentation theo cách
tải lười, độc lập **có hoàn lại** cho từng cover; code tác giả dùng
`random.sample` không hoàn lại trong batch từ residual bank ảnh được tạo qua
nhiều lượt augmentation. Đây là khác biệt sampling/task adaptation được công
khai, không phải tái hiện byte-identical hoặc cải tiến hiệu quả Bpp.

## Metric và hàng A1

Bảng `a1_native_comparison.md` có bảy hàng: sáu baseline mới và A1 tham chiếu.
Ba metric chính: **T-MPJPE tại d=1**, **MPJPE sạch**, **PA-MPJPE sạch**, đơn vị mm,
càng nhỏ càng tốt. Dose-response đầy đủ vẫn xuất riêng ở sáu dose
0, 0.2, 0.4, 0.6, 0.8, 1.

Clean PCK@0.2 nếu có trong CSV/JSON chỉ là chẩn đoán sạch: CSI không trigger so
với pose sạch trên toàn bộ joints, threshold tương đối
`0.2 * ||gt[5] - gt[12]||`, không phải 20 mm. **Không phải Target-PCK** hoặc attack
success rate. A1 cache cũ không có Target-PCK; bộ này không đặt lại tên clean PCK
hoặc suy ra Target-PCK từ T-MPJPE.

A1 tham chiếu đã được chọn trong quá trình phát triển. Một seed không cho độ
lệch chuẩn giữa seed, không chứng minh ưu thế thống kê hoặc bảo đảm xếp hạng.

## 1. Kiểm tra code, reference và dry-run

Chạy trên terminal MICA **sau khi xác nhận code mới đã push**:

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
git log -1 --oneline
nvidia-smi

bash remote_linux/18_run_a1_native_comparison.sh \
  --a1-reference "$HOME/backdooranalog/runs/mmfi_proposed_factorial_ablation_s42_v1" \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Dry-run chỉ dựng/kiểm tra plan, không đọc dataset hoặc dùng CUDA, không train.
Chạy thực sẽ xác minh reference A1. `--a1-reference` nhận folder gốc factorial
hoặc ZIP kết quả đầy đủ; không trỏ tới một baseline hay checkpoint tùy ý.
Mặc định đã dùng folder factorial trên. Nếu reference không đầy đủ/không khớp,
runner phải dừng thay vì train lại hoặc lấy các con số điền tay làm kết quả.

Có thể đổi Python/data-home bằng các đối số có quote:

```bash
bash remote_linux/18_run_a1_native_comparison.sh \
  --python "$HOME/backdooranalog/envs/dose-backdoor/bin/python" \
  --data-home "$HOME/backdooranalog" --dry-run
```

## 2. Train sáu baseline trên ba GPU

Chỉ dùng mapping dưới đây **sau khi xem `nvidia-smi` mới và có quyền dùng GPU**.
0, 2, 3 là ví dụ GPU vật lý, không phải khẳng định hiện tại chúng đang rảnh.
Không dừng process của người khác; ít VRAM/utilization trong một snapshot không
bảo đảm việc chia sẻ không ảnh hưởng.

```bash
cd "$HOME/backdooranalog/code"
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-a1-compare-s42 \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/18_run_a1_native_comparison.sh --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_a1_native_comparison_s42_v1.startup.log"'
```

Sau `CUDA_VISIBLE_DEVICES=0,2,3`, ID logic là `cuda:0`, `cuda:1`, `cuda:2`.
Mỗi GPU train một victim độc lập; tối đa ba job song song, các cell còn lại xếp
hàng chờ slot. Không dùng `cuda:3` sau mapping này. Không tạo controller thứ hai
nếu session/job cùng output vẫn đang chạy. Không có GPU nào dành để train A1.

Muốn chạy mới hoàn toàn, chọn `--outdir` mới. `--fresh` từ chối output không rỗng;
sau dry-run đã có manifest thì chạy bình thường, không thêm `--fresh` vào cùng
output. Resume/cache chỉ được dùng từ chính plan hợp lệ trong output mới.

## 3. Kiểm tra tiến độ và xuất lại bảng

```bash
DOSE_A1_OUT="$HOME/backdooranalog/runs/mmfi_a1_native_comparison_s42_v1"
tmux capture-pane -pt mmfi-a1-compare-s42 -S -30
pgrep -af '[r]un_a1_native_comparison.py'
tail -n 12 "$DOSE_A1_OUT"/*/console.log
find "$DOSE_A1_OUT" -name eval_cache.json | wc -l
cat "$DOSE_A1_OUT/a1_native_comparison.md"
```

Tiêu chí: đủ **sáu cache baseline hợp lệ** và summary bảy hàng xuất thành công.
Đếm file đơn thuần chưa đủ xác nhận config/cache/log khớp nhau. Hàng A1 không tạo
cache train mới trong output comparison. Ctrl+C ở `tail -f` chỉ dừng xem log.

Xuất lại summary, không train:

```bash
bash remote_linux/18_run_a1_native_comparison.sh --summary-only
```

## 4. ZIP kết quả và tải về Admin

Trên MICA, chỉ đóng artifact nhẹ, không checkpoint/raw dataset:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_a1_native_comparison_s42_v1_results.zip \
  mmfi_a1_native_comparison_s42_v1 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png'
ls -lh mmfi_a1_native_comparison_s42_v1_results.zip
```

Giữ ZIP factorial A1 ban đầu cùng kết quả comparison để đối chiếu nguồn tham
chiếu. Không cần đóng lại checkpoints hoặc dữ liệu.

Trên PowerShell máy Admin, không chạy lệnh sau trong SSH MICA:

```powershell
scp "mica-server:~/backdooranalog/runs/mmfi_a1_native_comparison_s42_v1_results.zip" "$env:USERPROFILE\Downloads\"
```
