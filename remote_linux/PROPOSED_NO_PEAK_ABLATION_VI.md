# Ablation Proposed không có peak bound — full MM-Fi, seed 42

## Material Passport

- Loại: bộ thực nghiệm offline CSI/HPE; không phải RF over-the-air.
- Phạm vi: ba bản của Proposed không có peak bound. Không thay đổi hoặc chạy lại các baseline truyền thống.
- Dữ liệu: full MM-Fi protocol1-s1 hiện có, 133056 mẫu TRAIN và 33264 mẫu đánh giá; không giới hạn số mẫu.
- Trạng thái: profile đã được kiểm thử local; tài liệu không xác nhận đã khởi chạy hoặc có kết quả benchmark MICA.
- Profile mới: `mmfi_proposed_no_peak_ablation_v1`. Kết quả bốn bản cũ trong `mmfi_proposed_ablation_s42_v1` được giữ nguyên; không ghi đè hoặc nhập kết quả cũ.

Đây là profile **ba nhánh, rho=0.4** đã chuẩn bị trước. Lượt factorial mới nhất
dùng **bốn nhánh, rho=0.1** trong `17_run_proposed_factorial_ablation.sh`; xem
`PROPOSED_FACTORIAL_ABLATION_VI.md`. Không chạy cả hai bộ nếu chỉ muốn lượt mới.

Runner mới thay đổi source hash và plan nên không resume/export trực tiếp manifest
cũ. Có thể đọc báo cáo/ZIP cũ; nếu cần chạy tiếp profile cũ, dùng snapshot code
tương ứng (`bc2065f`) trong một checkout riêng, không ghi đè code mới.

## Ba bản chạy

| Cell | Toán tử | Thay đổi so với Full mới |
| --- | --- | --- |
| `full` | `md_multicarrier`, paired | Multi-carrier không có phép giới hạn peak. |
| `w_o_coupling` | `md_multicarrier`, shuffled | Chỉ hoán vị dose payload giữa các mẫu poison; giữ tập poison, trigger dose và phân phối dose biên. |
| `w_o_auxiliary` | `micro_dropper`, paired | Bỏ carrier phụ, trở về carrier chính Original chuẩn hóa unit RMS. |

Không có hàng `w_o_peak` trong bộ mới: Full đã không có peak bound nên hàng đó
không còn là một ablation riêng. Không học trigger hoặc chọn tham số trên tập đánh giá.

Setting chung: seed victim 42, full 50 epoch, HPELi khởi tạo mới, SGD lr=0.001,
momentum=0.9, weight decay=0, batch size 32, rho=0.4, epsilon=0.185.
Giữ split, victim, optimizer và pose mục tiêu như bộ ablation đã chốt. Carrier/method
seed của RNG carrier phụ cố định 42; pattern Original vẫn theo seed thực nghiệm.
Do đó khi chạy thêm seed 43/44, không gọi toàn bộ trigger là cố định xuyên seed.
Training ghép trigger/payload theo dose U(0.2,1), trừ hàng shuffled.
Mỗi victim được đánh giá tại 0, 0.2, 0.4, 0.6, 0.8, 1; không train lại theo từng dose.

## Metric và ý nghĩa PCK

Bảng Markdown chính giữ ba metric: **T-MPJPE tại d=1**, **Dose-control error**
trung bình năm dose dương, và **MPJPE sạch**; cả ba dùng mm và càng nhỏ càng tốt.
CSV/JSON vẫn giữ thêm **PA-MPJPE sạch** và đúng **một PCK sạch tại 0.2**.

`Clean PCK@0.2` được tính trên **CSI không có trigger**, so với pose thật của mẫu
sạch, trên tất cả khớp. Đây **không phải attack PCK** hoặc attack success rate.
Với MM-Fi, một khớp đúng khi khoảng cách dự đoán–ground truth không lớn hơn
`0.2 * ||gt[5] - gt[12]||` của chính mẫu đó, theo chuẩn hóa DT-Pose hiện dùng.
Do đó 0.2 là ngưỡng **tương đối**, **không phải 20 mm**. Giá trị xuất ra là %.
Giữ metric này để kiểm tra chất lượng pose bình thường sau poisoning; T-MPJPE
và Dose-control error đo hành vi tấn công/điều khiển, không thay bằng clean PCK.

Dose-control error so sánh vector thay đổi dự đoán `(pred_d - pred_0)` với vector
thay đổi mục tiêu `(target_d - true_0)` trên các khớp mục tiêu. Nó không phải
Spearman, Schedule MAD, hoặc sai số curve đã chuẩn hóa theo endpoint.

`input_distortion.csv/json` vẫn đo Linf, L2 tương đối và SNR đầu vào digital trên
cùng tối đa 256 mẫu đánh giá và sáu dose. Original chỉ làm **mốc tham chiếu**;
**không áp giới hạn peak**, không yêu cầu Linf thấp hơn Original, và **không match
L2** giữa các bản. Vượt mốc Original là số đo chẩn đoán, không phải lỗi budget.
Việc bỏ phép peak projection không đồng nghĩa bỏ mức epsilon hay mọi ràng buộc
trong định nghĩa toán tử `micro_dropper`. Không suy ra RF stealthiness hoặc tính
khả thi over-the-air từ các số đo digital này.

## Kiểm tra plan local

Ví dụ với Python của môi trường phát triển, không cần GPU hoặc dữ liệu:

```powershell
python .\ATKBackd\run_proposed_ablation.py --no-peak --dry-run --seed 42 --devices cpu --data-home "E:\wifi\example-data-home" --outdir "E:\wifi\example-no-peak-plan"
```

Dry-run chỉ tạo/kiểm tra `proposed_ablation.resolved.json`; không train hoặc nạp
dataset. Chọn output mới cho mỗi cấu hình khác; không dùng output bộ cũ.

## Chạy trên MICA sau khi được yêu cầu

Chỉ dùng các lệnh này **sau khi đã pull commit chứa launcher 16 lên server**.
Kiểm tra `nvidia-smi` trước
mỗi lần chạy; các GPU rảnh trong snapshot cũ có thể đã thay đổi.

```bash
cd "$HOME/backdooranalog/code"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 \
bash remote_linux/16_run_proposed_no_peak_ablation.sh \
  --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Output mặc định mới:

```text
~/backdooranalog/runs/mmfi_proposed_no_peak_ablation_s42_v1/
```

Ba GPU vật lý 0, 2, 3 được đánh lại thành `cuda:0`, `cuda:1`, `cuda:2`.
Không truyền `cuda:3` sau khi đã lọc còn ba GPU. Không dừng process người khác;
GPU 0 có thể còn dịch vụ API dù utilization tại một thời điểm thấp.

```bash
mkdir -p "$HOME/backdooranalog/runs"
tmux new-session -d -s mmfi-prop-no-peak-abl \
  -c "$HOME/backdooranalog/code" \
  'set -o pipefail; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3 bash remote_linux/16_run_proposed_no_peak_ablation.sh --seed 42 --devices cuda:0 cuda:1 cuda:2 --num-workers 4 2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_proposed_no_peak_ablation_s42_v1.startup.log"'
```

Tối đa ba victim chạy song song, một cell/GPU. `--num-workers 4` là bốn
DataLoader worker **mỗi cell**, không phải tổng cộng bốn worker. Không tạo thêm
controller nếu job/session này còn hoạt động. Muốn chạy mới hoàn toàn, dùng
`--outdir` mới; `--fresh` từ chối thư mục không rỗng, kể cả có manifest dry-run.

## Theo dõi và lấy kết quả sau này

```bash
DOSE_ABL_OUT="$HOME/backdooranalog/runs/mmfi_proposed_no_peak_ablation_s42_v1"
pgrep -af '[r]un_proposed_ablation.py'
tail -n 30 "$HOME/backdooranalog/runs/mmfi_proposed_no_peak_ablation_s42_v1.startup.log"
tail -n 12 "$DOSE_ABL_OUT"/*/console.log
find "$DOSE_ABL_OUT" -name eval_cache.json | wc -l
cat "$DOSE_ABL_OUT/ablation_summary.md"
```

Phải đủ **ba cache hợp lệ** và summary trạng thái hoàn tất; chỉ đếm file chưa
chứng minh config/log hợp lệ. `pgrep` có thể hiện job profile khác nên đối chiếu
manifest/output. Ctrl+C ở `tail -f` chỉ dừng xem log, không dừng training trong tmux.

Xuất lại báo cáo từ cache của chính bộ mới, không train thêm:

```bash
bash remote_linux/16_run_proposed_no_peak_ablation.sh --seed 42 --summary-only
```

Các file cần giữ: `ablation_summary.md/csv/json`, `dose_response.csv/json/png`,
`input_distortion.csv/json`, manifest resolved, config, log, poison manifest,
audit và `eval_cache.json` từng cell. Không cần checkpoint/raw data để gửi báo cáo:

```bash
cd "$HOME/backdooranalog/runs"
zip -r -q mmfi_proposed_no_peak_ablation_s42_v1_results.zip \
  mmfi_proposed_no_peak_ablation_s42_v1 \
  -i '*.json' '*.csv' '*.md' '*.log' '*.yaml' '*.yml' '*.png'
```

Một seed không cho mean/std giữa seed hoặc kết luận hơn có ý nghĩa thống kê.
Sau này có thể lặp `--seed 43`, `--seed 44` vào output riêng tương ứng, nhưng
carrier/method seed vẫn giữ 42. Việc chọn bỏ peak sau khi xem bộ trước là quyết
định phát triển phương pháp; không gọi bộ mới là một đánh giá độc lập mù.
