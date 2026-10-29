# PiW3D MetaFi++: bốn attack, một job trên RTX 3090

Material Passport: academic-research-suite / experiment-agent (fidelity),
2026-10-29 Asia/Bangkok; handoff setup, chưa launch hoặc xác minh CUDA trên
máy 3090 từ máy Admin. Đây là continuation bốn victim, không phải bảng tám hàng.

Scope cố định: **FTrojan, BppAttack, FIBA, WaNet**, tuần tự trên cuda:0.
Không train Clean/A1/TSBA/Smooth LF-CSI. Dù ZIP chứa Smooth, scheduler không
chạy nó. Không kill process, không đổi recipe/source khoa học hoặc reset run cũ.

| Attack | Backup gốc | Bắt đầu tiếp |
| --- | ---: | ---: |
| FTrojan | 84 | 85 |
| BppAttack | 72 | 73 |
| FIBA | 33 | 34 |
| WaNet | không có | epoch 0 nếu không có bản AdamW tương thích mới hơn ở máy 3090 |

Tới final199; AdamW lr=.01/wd=.01, physical B32, FP32, seed42, PiW3D
one-person TRAIN27876/TEST2586 và body-hip-axis v2 như checkpoint gốc.
Operator, poison/cover keys, optimizer moments và RNG được kiểm trước import;
không silent epoch0 fallback cho ba backup. Linux->Windows/GPU continuation
được ghi provenance riêng, không bảo đảm bitwise trajectory.

## Chạy trên PowerShell 3090, trong env đã activate

Ưu tiên kiểm tra env đang mở, không giả định tên eclare-env đồng nghĩa CUDA
hoạt động. Cần Python >=3.11, Torch/torchvision tương thích, NumPy/SciPy và
CUDA thấy RTX3090. Không tự cài/chỉnh Torch trong env bạn khác. Nếu probe lỗi,
dừng và gửi output; chưa launch full. Bộ tải mới lấy processed PiW3D nếu chưa
có dữ liệu; vẫn cần data_bend.npy đúng SHA tại DataHome như helper sáu hàng.
Nếu chưa có dataset, cần chỗ cho ZIP 8,577,781,802 bytes + toàn bộ dữ liệu giải
nén + ZIP checkpoint 1.20 GiB + bản import và atomic saves. Bộ tải đọc tổng
expanded bytes thật từ ZIP và kiểm dung lượng trước giải nén; không bảo đảm
rằng 8 GiB hoặc một con số cố định nhỏ là đủ cho cả dataset.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Py = (python -c 'import sys; print(sys.executable)').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot find active Python.' }
    & $Py -c "import sys,torch,torchvision,numpy,scipy; assert sys.version_info >= (3,11), 'Need Python >=3.11'; assert torch.cuda.is_available(), 'CUDA unavailable'; assert '3090' in torch.cuda.get_device_name(0), 'Expected RTX3090'; print(sys.executable, torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'Environment probe failed; stop.' }

    $Repo = 'C:\Users\ssh3090\code-piw-metafi-4'
    if (Test-Path -LiteralPath "$Repo\.git") {
        $Origin = git -C $Repo remote get-url origin
        if ($LASTEXITCODE -ne 0 -or $Origin.Trim() -ne 'https://github.com/toan31072006/backdoor-analog.git') { throw 'Unexpected origin.' }
        git -C $Repo pull --ff-only origin main
    } else {
        if (Test-Path -LiteralPath $Repo) { throw 'Clone target exists but is not the expected repo.' }
        git clone https://github.com/toan31072006/backdoor-analog.git $Repo
    }
    if ($LASTEXITCODE -ne 0) { throw 'Git failed; no reset or deletion performed.' }

    powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\remote_tools\Get-PiW3DProcessedData.ps1" -PythonExe $Py
    if ($LASTEXITCODE -ne 0) { throw 'Dataset download/extraction failed; stop.' }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\remote_tools\Import-PiWMetaFiAdamWBackup.ps1" -PythonExe $Py
    if ($LASTEXITCODE -ne 0) { throw 'Backup import failed; stop.' }

    $S = "$Repo\remote_tools\Start-PiWMetaFiRemaining3090.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -FourAttacksOnly -PythonExe $Py
    if ($LASTEXITCODE -ne 0) { throw 'Prepare failed; no full training launched.' }
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -FourAttacksOnly -PythonExe $Py
    if ($LASTEXITCODE -ne 0) { throw 'Launch failed; inspect logs.' }
}
```

Bộ tải tạo venv nhỏ riêng `DataHome/tools/gdown-backup-env`, dùng
[gdown chính thức](https://github.com/wkentaro/gdown) bản 5.2.0; không cài vào
eclare-env. URL Drive do user cung cấp chưa được đọc thành công bằng browser
của agent. SHA256 ZIP phải khớp bản local đã xác minh:
`4607f4e3524af9f0cfb1b719ad89e6fee0e23c1765a150508604726fe720d43d`.
Nếu Drive sai quyền/quota hoặc nội dung khác, dừng, không nới hash. Có thể tải
bằng browser rồi truyền `-ZipPath` tới bộ import để không dùng downloader.

Dataset mirror đã dùng trong run gốc:
`https://drive.google.com/file/d/1tr98LAenpHUxhV3ibq91Qkjk6cxsftik/view`, SHA256
`bdd348909f1ff7d67d5674c40637c24b14d75e6bd9274cc9b3d8cfc3f2a35cf1`.
Browser của agent chưa đọc được Drive; command trên máy user mới xác minh
khả năng tải và SHA thực tế. Không thay bằng raw PiW ZIP thiếu csi_ap.
Bộ tải hỗ trợ `-DatasetRoot`/`-ZipPath` nếu chuyển sang ổ lớn; khi Prepare/Launch
phải truyền cùng `-DatasetRoot`. Có sẵn hai list thì bộ tải không sửa dữ liệu,
nhưng **không** gọi đó là full audit; Prepare mới kiểm full ordered/content.
Nếu giải nén dở, không tự overwrite hoặc xóa thư mục để retry.

PowerShell 5.1: phép tính dung lượng ZIP >8 GB dùng explicit Int64 ở cả hai
đối số Math.Max; bản dd4c85a cũ có thể ép về Int32 khi file chưa tải, gây
MethodArgumentConversionInvalidCastArgument trước khi tải/train. Chỉ cập nhật
Get-PiW3DProcessedData.ps1; không xóa ZIP/.part, checkpoint hoặc env để sửa lỗi.

Nếu NVIDIA WDDM trả `[Insufficient Permissions]` thay cho tên executable,
controller đọc ProcessId/Name qua Windows CIM để phân loại theo desktop
allowlist (bao gồm AnyDesk/TeamViewer). Không cần tắt ứng dụng remote hoặc
chạy quyền admin. PID không resolve được, Python và executable ngoài allowlist
vẫn bị chặn; lỗi CIM không tự bỏ guard. Đây là kiểm tra đồng chạy theo tên
process, không phải security boundary chống chương trình cố tình đổi tên.

Importer chỉ nhận đúng 12 member và bốn hash checkpoint đã biết, không overwrite
file import khác nội dung. Import cả backup Smooth để giữ hồ sơ nhưng không
train nó. Chuẩn bị kiểm full source/data/ordered IDs và physical B32 CUDA;
không auto retry, đổi batch hoặc AMP. Prepare hoàn tất mới Launch trong block.

## Status và log

```powershell
$Repo = 'C:\Users\ssh3090\code-piw-metafi-4'
$Py = (python -c 'import sys; print(sys.executable)').Trim()
powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\remote_tools\Start-PiWMetaFiRemaining3090.ps1" -Mode Status -FourAttacksOnly -PythonExe $Py
$R = 'C:\Users\ssh3090\backdooranalog\runs\piw_metafi_four_attacks_s42_3090_v1'
Get-ChildItem "$R\logs\train.*.log" -File | ForEach-Object { $_.Name; Get-Content -LiteralPath $_.FullName -Tail 4 }
nvidia-smi
```

Output riêng `runs/piw_metafi_four_attacks_s42_3090_v1`; training checkpoint
ở `baseline_six/attacks/metafiplusplus/seed42/<method>/checkpoint.pt`.
Một worker chết thì queue dừng, giữ checkpoint gần nhất, không retry hoặc chạy
job sau. Sau khi kiểm log/không còn controller-worker cũ, Launch cùng options
tiếp tục từ bản hợp lệ, không Prepare/reset/xóa gate để ép chạy.

## Phân biệt train xong với kết quả đã evaluate

ZIP này không có Clean AdamW final199. Vì user chỉ yêu cầu bốn attack, profile
**không tự train thêm Clean/A1** và không dùng Clean SGD cũ thay thế. Xong bốn
victim ghi `training_results.json` và `ALL_FOUR_TRAINED_NOT_EVALUATED`, không
gọi là comparison COMPLETE. Evaluator có clean-trained-model trigger controls
cần Clean AdamW cùng plan/dataset/target trước khi xuất toàn bộ bảng. Các metric
attack/no-trigger không tự thay bằng số của SGD. Khi có reference phù hợp, có
thể nhập bốn final checkpoint vào batch đánh giá tương ứng mà không train lại.
