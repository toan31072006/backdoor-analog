# Kiểm tra MetaFi++ / PiW3D: khớp phiên bản Vast, chỉ smoke Clean trước

Đây là **run chẩn đoán**, không thay bảng paper hoặc ghi đè kết quả Vast/3090 cũ.
**Lượt hiện tại chỉ Setup → Prepare → đọc smoke comparison rồi dừng. Không
launch training 200 epoch.** Smoke chỉ một model **Clean, seed 42**, B32, FP32,
forward/backward và một AdamW update trong RAM; không lưu checkpoint train.
Không có A1, TSBA, Smooth LF hay baseline khác trong hàng đợi này.

Nếu sau đó người dùng yêu cầu full train riêng thì recipe đã khóa là đủ TRAIN
27876 / TEST 2586, 200 epoch, checkpoint final199; AdamW LR 0.01, weight decay
0.01, B32, FP32. Việc code có chế độ `Launch` không tự cấp quyền chạy full train.

Mục tiêu là kiểm tra khi khớp các phiên bản thư viện đã ghi trên Vast thì Clean
trên Windows/3090 có gần lại kết quả cũ không. **Không hứa kết quả trùng tuyệt đối**:
Windows, GPU và các thư viện hệ thống vẫn khác Linux/Vast. Không gọi bộ phiên bản
này là yêu cầu do DT-Pose gốc công bố.

Giữ chính sách luồng như run Windows/3090 trước: `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, `MKL_THREADING_LAYER=SEQUENTIAL`. Log Vast cũ ghi các biến
này chưa đặt; code frozen vốn cũng đặt SEQUENTIAL khi chạy Windows. Vì vậy đây
là phép thử **khớp phiên bản thư viện trên Windows**, không phải tái tạo chính
xác toàn bộ môi trường Linux/Vast hoặc chỉ thay duy nhất một thư viện.

| Thành phần | Pin cho run chẩn đoán |
| --- | --- |
| Python | 3.12.15 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| NumPy | 2.5.4 |
| SciPy | 1.17.1 |
| PyYAML | 6.0.2, phụ trợ để đọc cấu hình |

`Setup` chỉ tạo/cài vào `backdooranalog\envs\piw-metafi-vast211`, không sửa
`eclare-env`, Python hệ thống, driver, checkout/run đang train hoặc checkpoint cũ.
`BasePython` mặc định là `C:\Users\ssh3090\eclare-env\Scripts\python.exe`, chỉ
được dùng để tạo venv mới và phải đúng 3.12.15. Không activate venv mới.
Nếu thư mục env đã tồn tại mà không có marker sở hữu của launcher, script sẽ dừng,
không tự xóa/ghi đè. Có thể chọn tên env **mới** khác bên trong `DataHome\envs`
bằng `-EnvDir`, và phải dùng lại đối số đó trong mọi lệnh tiếp theo.

Launcher dừng trước mọi lệnh Python nếu phiên SSH đang đặt `PIP_TARGET`,
`PIP_PREFIX`, `PIP_ROOT`, `PIP_USER`, `PYTHONHOME` hoặc `PYTHONPATH`: các biến
này có thể làm cài/import nhầm ra ngoài venv. Đây là lỗi bảo vệ có chủ đích;
không tự sửa cấu hình toàn máy hoặc env bạn đang dùng. Mở phiên SSH sạch hoặc
kiểm tra rồi bỏ riêng biến đó trong phiên hiện tại trước khi thử lại.
Pip chạy isolated, không cài user site, prefix luôn là env mới; cấu hình pip còn
chỉ định target/prefix/root/user cũng bị chặn. Cấu hình `extra-index-url`,
`find-links`, `no-index` còn được pip nhìn thấy cũng bị chặn, không sửa cấu hình
toàn máy để vượt kiểm tra nguồn package. Không đọc package từ user site. Launcher
không chấp nhận symlink/junction ở đường dẫn hoặc các thư mục cha của env,
DataHome, source/output/data/action/Python; đặc biệt kiểm tra trước khi Setup
tạo env để không vô tình cài ra ngoài phạm vi đã chỉ định.

## 1. Tải code đã kiểm tra

Dùng checkout **riêng** cho lần kiểm tra này, không pull/sửa checkout mà job khác
đang sử dụng. Lệnh bàn giao trong chat sẽ pin commit đã test. Khi tải thủ công từ
GitHub, có thể dùng ZIP source dưới đây nếu máy chưa có Git; đây chỉ là source,
không kèm dữ liệu/checkpoint. Không giải nén đè thư mục đã tồn tại.

```powershell
$ErrorActionPreference = 'Stop'
$CodeDir = 'C:\Users\ssh3090\code_clean_vastenv_download'
$CodeZip = 'C:\Users\ssh3090\Downloads\clean_vastenv_source.zip'
if ((Test-Path -LiteralPath $CodeDir) -or (Test-Path -LiteralPath $CodeZip)) {
    throw 'Duong dan da ton tai; chon ten moi, khong ghi de.'
}
Invoke-WebRequest -Uri 'https://codeload.github.com/toan31072006/backdoor-analog/zip/refs/heads/main' -OutFile $CodeZip
Expand-Archive -LiteralPath $CodeZip -DestinationPath $CodeDir
$S = "$CodeDir\backdoor-analog-main\remote_tools\Start-PiWMetaFiCleanVastEnv3090.ps1"
if (!(Test-Path -LiteralPath $S -PathType Leaf)) { throw 'Thieu launcher; dung.' }
```

Nếu đã có Git, có thể clone vào thư mục **mới** thay cho khối download trên:

```powershell
$CodeDir = 'C:\Users\ssh3090\code_clean_vastenv'
if (Test-Path -LiteralPath $CodeDir) { throw 'Checkout da ton tai; khong pull de.' }
git clone --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git $CodeDir
if ($LASTEXITCODE -ne 0) { throw 'Git clone loi; dung.' }
git -C $CodeDir rev-parse HEAD
$S = "$CodeDir\remote_tools\Start-PiWMetaFiCleanVastEnv3090.ps1"
```

## 2. Setup riêng, chưa train

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Setup
```

PyTorch/Torchvision được tải từ index cu128 chính thức của PyTorch; NumPy/SciPy/
PyYAML từ PyPI. Có lỗi tải hoặc không có wheel đúng phiên bản thì dừng, **không
tự đổi phiên bản**. Setup ghi owner marker, ready receipt và
`clean_vastenv_setup.freeze.txt`; chạy `pip check`. Setup thành công không có nghĩa
CUDA/B32 đã chạy được. Nếu Setup đã hoàn tất, gọi lại chỉ kiểm tra phiên bản/pip,
không tự cập nhật package. Nếu bị ngắt giữa lúc cài, chỉ được chạy lại Setup cho
đúng env có marker; không xóa env hoặc nâng cấp tùy tiện để vượt lỗi.

## 3. Audit và smoke thật, vẫn chưa train đủ 200 epoch

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

**Dừng tại đây.** `Prepare` không tự gọi `Launch`. Sau `PREPARED_ONLY`, đọc file:

```powershell
$Smoke = 'C:\Users\ssh3090\backdooranalog\runs\piw_metafi_clean_vastenv_s42_3090_v1\smoke_comparison.json'
[IO.File]::ReadAllText($Smoke)
```

Gửi toàn bộ report này và output Prepare để đối chiếu. Mốc lịch sử của Vast là
**Clean loss trước update = 3.8719141483306885**, batch32, seed42, 32 mẫu đầu của
ordered TRAIN (indices0–31). Đây là ordinary MPJPE loss ở đơn vị nội bộ (mét),
**không phải Clean full-TEST MPJPE, không phải kết quả hội tụ sau train**.
Không so với loss trung bình epoch hoặc batch ngẫu nhiên khác. Không đặt một
ngưỡng “gần” tùy ý để quyết định thành công; report giữ giá trị gốc và chênh lệch.
Loss ban đầu gần nhau chỉ kiểm tra một phần khởi tạo/input/runtime; không bảo đảm
clean MPJPE cuối hoặc whole training trajectory sẽ giống nhau.

Report lưu `old_vast.loss`, `new_smoke.loss`, chênh lệch tuyệt đối và tương đối,
`train_indices` từ0 đến31, cùng seed42. `SMOKE_ONLY_PASS` nghĩa B32 smoke hợp lệ,
**không phải** tiêu chí loss mới “đủ gần” Vast hoặc full-training reproduction.
Hai cờ `full_training_launched` và `checkpoint_saved` phải là `false` ở bước này.
Log stdout/stderr được giữ riêng trong `logs\prepare.*.log`, kể cả lỗi để gửi lại;
không cần chỉ dựa vào màn hình terminal hay PID cũ.

Giữ GPU không có job compute khác trong bước Prepare/Launch. Script không kill
process hoặc bỏ qua process lạ. Driver 560.94 đang báo CUDA 12.6; chỉ số này không
tự nó kết luận wheel cu128 chạy được hay không. Phải qua phép toán CUDA và
forward/backward/update B32 thật của Prepare. **Không cài/đổi driver tự động**;
nếu smoke lỗi thì gửi toàn bộ lỗi và dừng, không fallback CPU/batch nhỏ/AMP.

Dataset/action mặc định đọc nguyên trạng từ:

- `C:\Users\ssh3090\backdooranalog\datasets\Person-in-WiFi-3D_processed_v1`
- `C:\Users\ssh3090\backdooranalog\actions\data_bend.npy`

Có thể đặt `-DataHome`, `-DatasetRoot`, `-ActionNpy`, `-Outdir`, `-EnvDir`,
`-BasePython` tường minh khi đường dẫn khác. Dùng cùng bộ đường dẫn giữa các
pha. Không di chuyển/thay dữ liệu hay checkpoint cũ để vượt gate.

## 4. Chưa launch full training ở lượt này

Không chạy `-Mode Launch` sau smoke chỉ vì Prepare PASS. Đợi người dùng xem
`smoke_comparison.json` và yêu cầu full training riêng. `-Mode CheckOnly` chỉ
kiểm receipt/gate; không thay smoke comparison hoặc quyết định launch.

Nếu được yêu cầu full training sau này: đây là **foreground**, phải giữ SSH mở.
Không dùng tmux giả trên Windows,
Task Scheduler hay WMI; không hứa sống sau khi ngắt SSH. Tối đa một Clean worker,
không tự retry khi lỗi, không ghép với A1/baseline. Chỉ cho phép resume checkpoint
của chính run chẩn đoán sau kiểm tra nguồn/config/runtime; không import checkpoint
Vast hay run 3090 cũ vào đây. Không xóa lock hoặc receipt để ép launch.

Output riêng mặc định:
`C:\Users\ssh3090\backdooranalog\runs\piw_metafi_clean_vastenv_s42_3090_v1`.
Checkpoint đầy đủ lưu từng epoch cùng optimizer/RNG; không chọn best bằng TEST.
Trạng thái cuối phải dựa vào receipt/evaluation hoàn tất và log, không chỉ GPU 0%.

Trong terminal SSH khác, xem trạng thái:

```powershell
# Dung lai $S cua checkout rieng da tai o buoc 1.
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

## 5. Mốc so sánh đã có

| Run Clean | Clean MPJPE mm | Clean PA-MPJPE mm |
| --- | ---: | ---: |
| Vast cũ, báo cáo đã lưu | 136.164459 | 76.865811 |
| Windows/3090, env 2.14.1+cu126 trước đó | 200.126077 | 90.898174 |
| Run env-matched mới | Chưa chạy | Chưa chạy |

Chỉ so Clean toàn bộ 14 khớp trên full TEST. Không dùng Target-PCK/T-MPJPE hoặc
số từ 32 mẫu chẩn đoán để thay Clean full TEST. Một lần train mới gần/xa mốc cũ
chỉ là bằng chứng kiểm tra giả thuyết thư viện; không tự chứng minh quan hệ nhân
quả tuyệt đối hoặc mean/std nhiều seed.
