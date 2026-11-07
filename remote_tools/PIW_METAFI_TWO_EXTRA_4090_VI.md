# Person-in-WiFi-3D / MetaFi++: TSBA + Smooth-LF-CSI / RTX 4090

## Phạm vi

Material Passport: implementation/test handoff, 2026-11-07. Chưa launch full
training trên Vân, chưa có kết quả thực nghiệm mới. Người dùng yêu cầu hai
attack này chạy từng job một và cho phép mở evaluation bên ngoài mà không
chặn theo số PID. Đây là hai hàng bổ sung của comparison, không phải toàn bộ
bảng attack/defense. Không sửa LaTeX hay ghép số vào bảng tự động.

- TSBA-adapted rồi Smooth-LF-CSI, mỗi victim độc lập từ initialization gốc;
  không warm-start từ Clean/A1 và không train lại Clean/A1.
- PiW one-person, đúng ordered TRAIN27876 (lọc next-frame từ list28121) /
  TEST2586, seed42, rho=.1 (2787 poison), không cover. Giữ poison IDs/doses
  và body-hip-axis target của A1: pivot7, joints11/13, hip axis5-7, 40deg*d.
- MetaFi++ source frozen: official ResNet34 V1 initialization rồi native
  metafi_init; AdamW lr=.01/wd=.01, betas(.9,.999), eps1e-8; physicalB32,
  workers0, drop_last, FP32, 200epoch/final199. Không scheduler/AMP/TEST selection.
- TSBA: ba generator độc lập cho ba amplitude groups, relative eps=.1,
  hidden32, Adam generator lr=.001, warmup10, bốn gen steps/batch có poison.
  Đây là white-box training adaptation, khác quyền truy cập data-only của A1.
- Smooth-LF: fixed seeded DCT low-frequency control, eps=.185, seed42;
  không gọi đây là nguyên bản optimized LF image attack. Cả hai giữ phase
  rows90:180 nguyên trạng, không normalize CSI lần hai. Không claim equal-L2.
- Đủ sáu dose0,.2,.4,.6,.8,1 trên cùng final victim. T-MPJPE/PCK chấm target
  joints11/13 tại d1, PCK threshold.2 dùng cleanGT elbow distance6-4 (không
  dùng MM-Fi5-12), DCE mean chỉ năm dose dương, clean MPJPE/PA toàn14khớp.
  Giữ no-trigger và clean-model+trigger controls, NPZ đủ poses/targets/IDs.

## File và prerequisites trên máy Vân

Ưu tiên clone code vào thư mục mới (không đè checkout/job đang chạy):

```powershell
$Code = Join-Path $env:USERPROFILE 'PiWMetaFiTwoExtra4090'
git clone --depth 1 --single-branch --branch main https://github.com/toan31072006/backdoor-analog.git $Code
$S = Join-Path $Code 'remote_tools\Start-PiWMetaFiTwoExtra4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
# Chỉ sau khi Prepare PASS:
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Nếu thư mục clone đã tồn tại, không xóa/ghi đè; chọn tên thư mục code mới.
DataHome vẫn mặc định là USERPROFILE/backdooranalog, độc lập với thư mục code.
Không cần chép lại dataset hoặc Clean checkpoint. Giữ PowerShell Launch mở.

Hoặc dùng ZIP đã bàn giao:

Giải nén gói vào thư mục code mới, ví dụ:
`C:\Users\Van\Downloads\PiWMetaFiTwoExtra4090`.
Không chép đè checkout/job BackTime đang chạy. Gói gồm frozen source và các
sidecar cũ nguyên byte; source được extract vào output riêng và kiểm SHA.

Dùng môi trường có sẵn:
`C:\Users\Van\backdooranalog\envs\piw-metafi-vast211-4090\Scripts\python.exe`.
Không dùng dose-backdoor hoặc base thay thế; runtime/source phải khớp Clean.
Không cài package tự động. Data/action ở backdooranalog như các run PiW cũ.

Clean reference bắt buộc đã complete:
`backdooranalog\runs\piw_metafi_clean_vastenv_s42_4090_v1`.
Runner kiểm checkpoint, prediction hashes, full audit, ordered IDs, file stats,
runtime/GPU và source. Nếu chưa có, Prepare dừng; không tự train Clean hoặc
giả receipt. Official ResNet34 V1 weights phải còn trong TORCH_HOME/cache cũ.

Output mặc định nằm trên U nếu U tồn tại:
`U:\backdooranalog\runs\piw_metafi_two_extra_s42_4090_serial_v1`;
nếu không thì dùng DataHome/runs. Cần ít nhất4GiB trống trên output volume.
Có thể chọn parent khác bằng `-Outdir`, nhưng giữ đúng tên thư mục run cuối.

## Lệnh

```powershell
$S = "$env:USERPROFILE\Downloads\PiWMetaFiTwoExtra4090\remote_tools\Start-PiWMetaFiTwoExtra4090.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

Prepare chỉ audit và hai CUDA B32 smokes nối tiếp, có đủ bốn TSBA gen steps,
không full train, không checkpoint training. Sau khi Prepare PASS:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch
```

Foreground: giữ PowerShell mở. TSBA train200 -> Smooth-LF train200, sau đó
evaluate cả hai. Có atomic checkpoint mỗi epoch kèm optimizer/RNG và
TSBA generator/optimizer. Gọi Launch lại là resume đúng checkpoint của run
này sau khi đã xem lỗi; không tự retry. Completed final199 được skip có kiểm.

Status không dùng CUDA hoặc training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
```

Chỉ evaluate checkpoint final199 đã có, không train:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Evaluate -Method tsba_adapted
# Hoặc bỏ -Method để evaluate cả hai checkpoint final199.
```

## Không giới hạn PID, nhưng vẫn bảo vệ dữ liệu

Không query/chặn số Python PID hay GPU processes, không global GPU lock và
không kill. Chỉ khóa output riêng để không có hai writer cùng sửa checkpoint.
Có thể mở evaluation script khác với output riêng trong khi train. Không bảo
đảm đủ VRAM hoặc không ảnh hưởng tốc độ: smoke chỉ chứng minh tại lúc chạy,
job mới thêm sau đó vẫn có thể gây OOM. Nếu OOM: giữ log/checkpoint, không tự
giảm batch/chạy lại/khởi chạy method sau. Giới hạn riêng của BackTime hoặc
launcher lịch sử khác không bị thay đổi bởi bộ này.

Logs trong output/logs; checkpoint/predictions/evaluation trong
attacks/metafiplusplus/seed42/{tsba_adapted,smooth_lf_csi}.
RESULT.md và results.json là report riêng của hai hàng. Metrics phải replay
chính xác từ NPZ trước khi handoff; log PASS local không chứng minh training
CUDA/full-data thành công hoặc kết quả khoa học tốt.

## Kiểm chứng local và giới hạn bàn giao

Đã kiểm tra điều phối bằng mock/CPU và test integration evaluation với arrays
giả đúng shape2586x6x14x3: restore generator TSBA, giữ Clean read-only,
replay metrics từ NPZ và từ chối metric bị sửa. Đây không phải dữ liệu thật.

Test gradient CPU riêng chưa xác nhận: subprocess bị timeout300s ngay ở bước
import PyTorch trong môi trường local thiếu RAM (khoảng260MiB trống tại lúc
kiểm tra). Không tính timeout là PASS hoặc bằng chứng lỗi thuật toán; không
cài package, đổi recipe hay dừng ứng dụng khác để vượt qua. Test được giữ
trong tests/test_piw_metafi_two_extra_4090.py để chạy lại khi đủ tài nguyên.
Prepare trên RTX4090 vẫn là gate bắt buộc: cả hai smoke physicalB32 phải PASS,
riêng TSBA phải có đúng4generator updates, trước khi gọi Launch.
