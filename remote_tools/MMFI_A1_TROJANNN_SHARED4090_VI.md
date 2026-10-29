# Vân4090: A1 + TrojanNN, giữ nguyên nscaling

## Material Passport

- Origin: academic-research-suite / experiment-agent, 2026-10-30.
- Scope: execution-only exception explicitly requested by the user.
- Hai victim: HPE-Li/MM-Fi P3S3 A1 seed42 (resume, không scratch) và HPE-Li/MM-Fi
  P1S1 TrojanNN seed42 đã Prepare. Đây là hai split khác nhau, không phải hai
  hàng so sánh trực tiếp cùng protocol.
- Giữ đúng job ngoài đã được user đồng ý: `scripts\nscaling.py
  LandS,gbd,storm,20term,ssn S1,S3 5`, parent PID44248 tại
  `U:\ijoc\.venv\Scripts\python.exe`, child PID26344 tại
  `C:\Users\Van\AppData\Local\Python\pythoncore-3.12-64\python.exe`.
- Không coi GPU-util 1% tức thời là không có job hoặc bảo đảm hai victim vừa
  VRAM. Chính sách mới: tối đa hai model workers + một external job đã chỉ rõ.
- Verification local: **302 tests passed** trong một lượt (104 shared-context
  tests +198 operator/runner/launcher regressions). Independent review không
  thấy blocker còn lại. Chưa chạy CUDA hoặc training thật trên máy Vân từ đây.

## Bảo toàn thí nghiệm

Sidecars mới chỉ ở `remote_tools/`, không thêm/sửa scientific file trong
`ATKBackd`. Vì vậy source fingerprint của TrojanNN đã Prepare không đổi. A1
tiếp tục dùng checkout cũ `C:\Users\Van\code`; không pull/sửa checkout đó.

Giữ nguyên old `plan.json`, `gpu_gate.json`, `trigger.pt`, inputs/poison receipt
và checkpoint. Không giả old gate là chứng nhận cho bối cảnh chia sẻ mới.
Request và smoke mới ghi append-only dưới
`backdooranalog\tools\shared4090\executions\<UTC_uuid>`. Quyền ngoại lệ ràng
buộc PID, creation time, executable, toàn bộ argv và parent/child của nscaling;
không cho phép mọi Python/GPU job, SIG hay process mang tên tương tự. PID bị
reuse/đổi command bị chặn. Job đã kết thúc không cấp quyền cho job mới.

TrojanNN vẫn phải kiểm tra original gate/config/source/input/runtime và giữ
frozen trigger, sau đó smoke hai update B32 thật trong context mới. Không tối ưu
lại trigger hoặc đổi target activation vì loss plateau. Smoke chứng minh các
bước đó chạy được tại thời điểm thử, không chứng minh mọi tải tương lai vừa VRAM
hoặc trigger tấn công hiệu quả. Original victim sẽ fresh theo recipe đã duyệt
nếu chưa có victim checkpoint; checkpoint victim đã có thì native strict resume.

A1 chỉ được start nếu không thấy đúng A1 peer đang sống và còn checkpoint hợp
lệ. Audit CPU dùng chính native runner/plan/preflight/smoke receipt, kiểm tra
strict resume, epoch0..49, configuration fingerprint và đủ model/optimizer/RNG.
Thiếu checkpoint hoặc binding khác thì dừng, không fallback epoch0. Nếu A1
đang chạy, giữ nguyên; nếu đã hoàn tất đúng cache/evaluation thì không chạy lại.
Không gọi six-cell controller, không chạy Clean/S1/S2/SIG/TrojanNN P3S3.

## Lệnh trên PowerShell máy Vân

Chỉ cập nhật `code_trojannn`, không cập nhật `code` của A1. Helper mới chạy
foreground: giữ SSH mở, không để client hoặc máy Vân sleep. Không tạo Task
Scheduler, WMI service, tmux hay thay quyền. Không bảo đảm sống qua disconnect.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Repo = 'C:\Users\Van\code_trojannn'
    git -C $Repo pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { throw 'Pull loi; dung, khong reset/xoa file.' }
    $S = "$Repo\remote_tools\Start-MMFiA1TrojanNNShared4090.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -AllowKnownNScaling
    if ($LASTEXITCODE -ne 0) { throw 'Shared launch loi; xem log, khong Prepare lai.' }
}
```

Log mới của A1 nằm `p3_s3\a1\console.shared.*.log` ngay trong run cũ; log cũ
không bị ghi đè. Log TrojanNN và metadata nằm dưới execution folder mới được in
ra. Để xem bằng SSH thứ hai:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\Van\code_trojannn\remote_tools\Start-MMFiA1TrojanNNShared4090.ps1' -Mode Status
```

PID tạo thành công không chứng minh epoch đang tiến. Đọc log thay đổi/checkpoint
và kết quả evaluate; A1/TrojanNN vẫn chấm đủ các dose theo hợp đồng gốc.
Worker lỗi thì không retry, không tự dừng peer. Nếu smoke sau khi resume A1 lỗi,
A1 có thể vẫn đang chạy; controller chờ owned peer kết thúc trước khi nhả lock,
không khởi chạy thêm job. Đọc log trước khi thử thao tác khác. Không xóa lock,
checkpoint hay receipt để ép chạy. Global launcher lock cùng native cell locks
ngăn các helper này tạo duplicate; snapshot guard không thể ngăn chương trình
bên ngoài tự khởi chạy thêm sau đó.
