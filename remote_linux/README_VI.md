# Chạy Dose Backdoor trên MICA/Linux

Bundle này dành cho máy chủ MICA/Linux. Nó không chứa dataset hay
action skeleton; launcher truyền đường dẫn Linux vào runner bằng CLI nên các
đường dẫn Windows/Colab còn nằm trong YAML không được sử dụng.

## 1. Cấu trúc dữ liệu mặc định

```text
~/backdooranalog/
├── datasets/
│   ├── Compress/                    # MM-Fi: E01, E02, E03, E04
│   └── Person-in-WiFi-3D/           # train_data, test_data
├── actions/
│   └── data_bend.npy                # bắt buộc cho main
└── runs/                            # tự tạo khi chạy
```

Hai dataset đã được kiểm tra với cấu trúc này. `data_cross.npy` và
`data_nod.npy` là tùy chọn; ma trận paper hiện tại chỉ chạy scenario `bend`.

Nếu file bend đang ở máy Windows, mở PowerShell **trên máy Windows** và tải qua
SSH alias đã cấu hình:

```powershell
ssh mica-server "mkdir -p ~/backdooranalog/actions"
scp "C:\duong-dan\data_bend.npy" mica-server:~/backdooranalog/actions/data_bend.npy
```

Nếu file ở Google Drive, có thể tải theo ID sau khi đã cấu hình `gdrive:`:

```bash
mkdir -p ~/backdooranalog/actions
rclone -P backend copyid gdrive: FILE_ID ~/backdooranalog/actions/
```

Sau đó đổi tên file thực tế thành `data_bend.npy` nếu cần. Không đổi tên
`data_nod.npy` thành bend: nội dung chuyển động phải đúng action.

## 2. Môi trường Python

Khuyến nghị Python 3.10 hoặc 3.11. Nếu chưa có môi trường phù hợp:

```bash
conda create -n dose-backdoor python=3.11 -y
conda activate dose-backdoor
python -m pip install --no-cache-dir -r remote_linux/requirements-linux.txt
```

Cài PyTorch CUDA phù hợp với driver của server riêng; nếu môi trường hiện tại đã
`import torch` và thấy GPU thì không cài lại. Kiểm tra nhanh:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Thứ tự chạy bắt buộc

Đứng tại thư mục `Dose_Backdoor_CODE` của bundle đã giải nén:

```bash
bash remote_linux/00_preflight.sh --device cuda:0
bash remote_linux/01_dry_run.sh --dataset both --device cuda:0
bash remote_linux/02_smoke_test.sh
bash remote_linux/02b_pilot_test.sh --dataset mmfi --device cuda:0 --num-workers 4
```

Chỉ khi bốn bước trên đều qua mới chạy ma trận chính:

```bash
bash remote_linux/03_run_main.sh --dataset both --device cuda:0 --parallel 1 --num-workers 4
```

Main gồm 18 lần train: `Clean`, `Proposed`, `TSBA` x ba seed `42,0,1` x hai
dataset. MM-Fi dùng 50 epochs; PiW3D dùng 200 epochs. `parallel=1` là mặc định an
toàn trên một GPU; chỉ tăng sau khi đã đo VRAM.

Pilot dùng toàn bộ dữ liệu nhưng chỉ một seed: Proposed/Clean 2 epochs và TSBA
11 epochs. Vì TSBA có warm-up 10 epochs, pilot này vẫn có thể chạy lâu.

## 4. Giữ job khi ngắt Remote SSH

Khuyến nghị dùng `tmux`:

```bash
tmux new -s dose
cd /duong/dan/toi/Dose_Backdoor_CODE
bash remote_linux/03_run_main.sh --dataset both --device cuda:0 --parallel 1 --num-workers 4
```

Nhấn `Ctrl-b`, rồi `d` để tách phiên. Quay lại bằng:

```bash
tmux attach -t dose
```

Nếu không có tmux:

```bash
mkdir -p ~/backdooranalog/logs
nohup bash remote_linux/03_run_main.sh --dataset both --device cuda:0 --parallel 1 --num-workers 4 \
  > ~/backdooranalog/logs/main.log 2>&1 &
echo $!
tail -f ~/backdooranalog/logs/main.log
```

## 5. Resume và kiểm tra tiến độ

Chạy lại đúng lệnh main sẽ nạp checkpoint tương thích và bỏ qua cell đã có
`eval_cache.json`; không cần xóa output sau khi SSH rớt hoặc server reboot.

```bash
bash remote_linux/04_status.sh
```

Kết quả chính nằm tại:

```text
~/backdooranalog/runs/paper_main/
```

Mỗi cell lưu checkpoint, metadata, cache đánh giá và `result.json`; mỗi nhóm lưu
`results.csv`, `results.json`, và `results_agg.csv`.

## 6. Contract không được tự ý đổi

- PiW3D phải dùng `pivot=7` (`right_hip`, target right knee/right ankle).
- MM-Fi dùng `pivot=1` theo topology 17-joint hiện tại.
- Victim học bằng MPJPE chuẩn, ordinary ERM; launcher không thêm attack-specific loss.
- Main dùng đúng seeds `42,0,1` và cùng recipe giữa attacked/clean control.
- Không trộn kết quả pilot, pivot 3, hoặc output từ bundle Windows vào `paper_main`.

Các launcher chỉ thay đường dẫn/runtime. Chúng không sửa hyperparameter khoa học
trong config, vì vậy việc chuyển từ Windows sang Linux không làm đổi method.
