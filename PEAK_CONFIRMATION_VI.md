# Xác nhận peak-control trên full MM-Fi, seed 42

`ATKBackd/run_peak_confirmation.py` train mới hai victim: **Original**
(`original`, trigger `micro_dropper`) và **multicarrier peak-capped**
(`md_multicarrier_peak_matched`). Cả hai dùng toàn bộ train/test của split
MM-Fi `protocol1-s1`, split seed 0, train seed 42, **50 epochs** (log epoch
`0` đến `49`). Đây là bước xác nhận full sau draft 15 epochs; không dùng
subset hay validation holdout của draft, không nhập checkpoint/cache cũ.
Runner yêu cầu đủ **133,056 train / 33,264 test frames** của P1-S1;
dataset thiếu không được tự chuyển thành subset.

Hai method giữ cùng HPELi, ordinary ERM/MPJPE, SGD lr `0.001`, momentum
`0.9`, batch 32, poison rate `rho=0.4`, poison IDs/doses và payload. Payload
xoay z `40*d` độ quanh pivot 1, target joints `[2,3]`; dose train
`U(0.2,1)`, eval `0,0.2,...,1`. Nominal epsilon giống nhau; candidate bị
giới hạn peak nhiễu trên từng input/dose theo Original. Giới hạn này **không
đồng nghĩa bằng L2, cùng năng lượng hoặc đã chứng minh stealth**.

## Chạy trên server

Kiểm tra trạng thái GPU **hiện tại** bằng `nvidia-smi`, rồi chọn GPU được
phép dùng và đủ tài nguyên. Ví dụ bên dưới dùng hai GPU vật lý **2 và 3**;
đây là ví dụ ánh xạ, không phải khẳng định hai GPU đang rảnh.

Paste khối này trước để tạo tmux:

```bash
cd "$HOME/backdooranalog/code"
nvidia-smi
tmux new -s mmfi-peak-confirmation
```

Khi đã ở shell trong tmux, paste riêng khối train:

```bash
cd "$HOME/backdooranalog/code"
export CUDA_VISIBLE_DEVICES=2,3
bash remote_linux/07_run_peak_confirmation.sh \
  --data-home "$HOME/backdooranalog" \
  --outdir "$HOME/backdooranalog/runs/mmfi_peak_confirmation_s42_v1" \
  --devices cuda:0 cuda:1 --num-workers 4 --distortion-samples 256 --fresh
```

`cuda:0` là GPU vật lý 2, `cuda:1` là GPU vật lý 3 sau khi đặt
`CUDA_VISIBLE_DEVICES=2,3`. Hai GPU chạy **hai job full độc lập**, mỗi GPU
một victim; không phải DDP chia một model. Một GPU cũng chạy được bằng
`CUDA_VISIBLE_DEVICES=2` và `--devices cuda:0`, khi đó hai job chạy lần lượt.

Wrapper mặc định dùng `$HOME/backdooranalog/envs/dose-backdoor/bin/python`
khi code nằm ở `$HOME/backdooranalog/code`. Nếu bố trí khác, đặt `DOSE_PY`
trỏ tới Python của môi trường và truyền `--data-home` đúng thư mục chứa
`datasets/Compress` và `actions/data_bend.npy`.

Nhấn `Ctrl-b`, rồi `d` để detach; attach bằng
`tmux attach -t mmfi-peak-confirmation`. Không paste khối tạo tmux và khối
train thành một lần.

## Preview, resume và export

Preview cấu hình trước khi train, dùng thư mục preview riêng. `--dry-run`
chỉ ghi plan, không load dataset, khởi tạo CUDA hoặc train:

```bash
bash remote_linux/07_run_peak_confirmation.sh \
  --data-home "$HOME/backdooranalog" \
  --outdir "$HOME/backdooranalog/runs/mmfi_peak_confirmation_s42_preview" \
  --devices cuda:0 cuda:1 --num-workers 4 --distortion-samples 256 --dry-run
```

Lần train đầu tiên phải dùng **outdir mới** với `--fresh`. Không dùng lại
thư mục draft, peak-control 15 epochs hoặc full matrix. `--fresh` không phải
lệnh xóa/ghi đè kết quả cũ. Resume bằng đúng lệnh train trên, **bỏ
`--fresh`**, giữ nguyên outdir và cấu hình khoa học. Runner từ chối budget
hoặc source không tương thích; thay code/config phải tạo run mới. CLI không
cho đổi epochs, seed hay cap train/test. Không sửa manifest để ép cache-hit.

Sau khi cả hai job hoàn thành, dựng lại báo cáo từ chính run này bằng:

```bash
bash remote_linux/07_run_peak_confirmation.sh \
  --data-home "$HOME/backdooranalog" \
  --outdir "$HOME/backdooranalog/runs/mmfi_peak_confirmation_s42_v1" \
  --devices cuda:0 cuda:1 --num-workers 4 --distortion-samples 256 --export-only
```

## Đọc kết quả

Trong outdir, xem `peak_confirmation.resolved.json` để kiểm tra contract.
Mỗi method có thư mục riêng `original/`, `md_multicarrier_peak_matched/`,
`console.log` và checkpoint được cập nhật mỗi epoch. `confirmation_inputs.json`
ghi hash danh sách pair IDs có thứ tự và action trước khi train; resume kiểm
tra lại các hash này. Theo dõi log bằng:

```bash
tail -f "$HOME/backdooranalog/runs/mmfi_peak_confirmation_s42_v1/original/console.log"
```

Báo cáo tổng hợp gồm `confirmation_summary.csv/json/md`,
`dose_response.csv`, `input_distortion.csv/json`. Đọc clean MPJPE cùng
T-MPJPE và đường dose-response; file distortion chỉ đo nhiễu trên số mẫu
đã cấu hình (256 ở lệnh trên), còn train/test vẫn full.

Thiết lập này xác nhận hai method trên full MM-Fi với **một seed**; bản
thân việc chạy full không chứng minh candidate mới về học thuật, tốt hơn
Original, hay đã đủ điều kiện viết paper. Không tune tiếp method trên test
chính thức; cần đánh giá nhiều seed và bằng chứng bổ sung trước các kết luận
tổng quát.
