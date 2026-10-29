# Tiếp tục bốn attack PiW3D/MetaFi++ trên Windows 3090, không giữ SSH

## Phạm vi / Material Passport

- 2026-10-29, academic-research-suite / experiment-agent: sửa execution, không sửa khoa học.
- Helper mới: `Start-PiWMetaFiFourAttacks3090Task.ps1`.
- Chỉ tiếp tục run `piw_metafi_four_attacks_s42_3090_v1`: FTrojan → BPP → FIBA → WaNet,
  tuần tự một worker `cuda:0`. Không thêm Clean/A1/TSBA/Smooth LF.
- Giữ source/controller, gate, input, checkpoint, optimizer/RNG và recipe đã prepare.
  MetaFi++ AdamW lr0.01/wd0.01, physicalB32, FP32, seed42, fixed final199.
- Không chạy Prepare lại, không lấy checkpoint import cũ ghi đè progress mới.
  Theo log người dùng, FTrojan đã lưu epoch183; controller kiểm tra rồi resume184.
- Bốn attack vẫn chỉ train; không có Clean AdamW tương ứng để tự xuất bảng evaluate.

## Vì sao không dùng tmux?

Máy này chạy Windows native PowerShell. Không cài WSL/tmux hoặc đổi môi trường
CUDA chỉ để giữ phiên. Task Scheduler chạy dưới chính tài khoản hiện tại,
`S4U`, quyền `Limited`, không lưu mật khẩu, không cần phiên SSH sống liên tục.
Scheduled action chờ controller foreground đến khi kết thúc, giữ exit code và
stdout/stderr riêng; không dùng một process SSH chạy ẩn rồi coi như bền phiên.

Task chỉ chạy khi người dùng gọi Launch, không có trigger boot/logon, retry,
stop/kill process hay tự khởi chạy lại khi lỗi. Không đổi registry, security policy,
power plan hoặc package/environment. Execution time limit bằng0 (không bị default72h),
IgnoreNew để không tạo instance trùng. Controller vẫn giữ khóa output và chặn GPU đang bận.

S4U cần quyền **Log on as a batch job** và quyền tạo task cho chính tài khoản.
Nếu máy chặn, helper báo lỗi; không tự nâng quyền hoặc fallback về cách SSH cũ.
Nhờ chủ máy cấp quyền phù hợp, không gửi mật khẩu/token cho assistant.

S4U không dùng được network share/EFS. Dataset/checkpoint/code/venv phải là file
local, không mã hóa EFS. ResNet34 cache chính thức đã tải trước phải có tại
`backdooranalog\torch-cache\hub\checkpoints\resnet34-b627a593.pth` (hoặc TORCH_HOME
local hiện dùng); helper kiểm tra hash và không tự tải qua mạng.

## Dùng đúng source đã prepare trên máy3090

Source hiện tại do người dùng giải nén, không phải Git clone:

`C:\Users\ssh3090\piw4\backdoor-analog-dd4c85a67c294d9c085c7f404b54a1eda15d4305`

Controller đã sửa WDDM ở đó phải có SHA256:

`1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2`.

Chỉ tải **helper mới** từ GitHub vào một file local riêng, truyền `-Repo` tới
source cũ. Không pull/extract source mới đè controller; việc đó làm lệch gate.
Giữ nguyên helper/request sau khi task đã đăng ký, không sửa file trong lúc chạy.
Lệnh tải pinned helper kèm hash được cung cấp khi bàn giao commit.

Sau khi đã có helper `$S`, chạy cả block PowerShell sau trên3090:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $S = 'C:\Users\ssh3090\backdooranalog\tools\piw_task3090\Start-PiWMetaFiFourAttacks3090Task.ps1'
    $Repo = 'C:\Users\ssh3090\piw4\backdoor-analog-dd4c85a67c294d9c085c7f404b54a1eda15d4305'
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch `
        -Repo $Repo -PythonExe 'C:\Users\ssh3090\eclare-env\Scripts\python.exe'
    if ($LASTEXITCODE -ne 0) { throw 'Task launch loi; dung, khong chay launcher cu.' }
}
```

Launch kiểm tra code/cache, chạy gate check gốc, ghi immutable request rồi
đăng ký/start task. Task kiểm tra lại helper/source/account, cấp phát CUDA thật
trong scheduled context, check gate và chạy controller. Request có hash, task
đã tồn tại phải đúng action/principal/settings; không dùng Force để đè task khác.
Nếu gọi lại khi task đang Running/Queued thì không start instance thứ hai.

**TASK_START_REQUESTED chỉ có nghĩa đã yêu cầu start, chưa chứng minh train.**
Chờ xác nhận `SCHEDULED_CUDA_ALLOCATION_PASS`, gate check thành công và worker
log tiến epoch mới. Khi đó có thể đóng SSH; máy vẫn phải bật và không sleep.
Restart/shutdown vẫn làm job dừng, cần Launch thủ công để resume checkpoint sau đó.

## Xem task, log và checkpoint

```powershell
$S = 'C:\Users\ssh3090\backdooranalog\tools\piw_task3090\Start-PiWMetaFiFourAttacks3090Task.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status

$R = 'C:\Users\ssh3090\backdooranalog\runs\piw_metafi_four_attacks_s42_3090_v1'
Get-ChildItem "$R\logs\train.*.log" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Tail 10 -Encoding UTF8 }
```

Task status/`LastTaskResult=0` không thay cho kết quả training. Chỉ coi train
bốn job xong khi controller có `ALL_FOUR_TRAINED_NOT_EVALUATED` và
`training_results.json`. Worker lỗi thì controller dừng, không launch job sau
và không retry. Xem `controller.task.*.stderr.log`, `.wrapper-error.log`,
`.cuda.stderr.log`, `.check.stderr.log` và log worker tương ứng.

Helper mới không sửa `.pt`, gate hoặc file source đã bind. Checkpoint hiện có
được controller kiểm tra model/optimizer/RNG/config rồi mới resume; không có
fallback âm thầm train từ0 khi checkpoint sai.

## Giới hạn xác minh

Local gate29/10: 56 kiểm thử đạt, gồm PowerShell AST/native quoting, process
CPU thật ghi đồng thời stdout/stderr160KiB và trả exit37, definition Scheduler
mocked, chống task lạ/instance trùng, request hash ổn định, và từ chối request sửa.
Readonly COM/client-only CIM xác nhận enum/settings; không đăng ký OS task.
111 kiểm thử hồi quy controller/handoff cũng đạt; hai test gọi script dataset
được chạy lại với ExecutionPolicy ByPass riêng process, giống CLI bàn giao,
không đổi policy máy. Không có sửa đổi scientific controller/launcher cũ.

Kiểm thử local parse/quoting/mocked task definition và foreground-process logs
không phải bằng chứng task đã chạy trên3090. Việc đăng ký task thật, S4U CUDA
và sống qua disconnect cần xác nhận bằng log trên chính máy người dùng.

Nguồn chính thức: [task security contexts](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks),
[S4U restrictions](https://learn.microsoft.com/en-us/windows/win32/api/taskschd/ne-taskschd-task_logon_type),
[execution time limit](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasksettings-executiontimelimit).
