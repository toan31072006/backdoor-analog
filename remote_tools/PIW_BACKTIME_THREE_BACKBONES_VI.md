# PiW BackTime: HPE-Li → WiFlow → MetaFi++ — setup 2026-11-10

## Material Passport

- Yêu cầu: chuẩn bị BackTime cho ba backbone trên Person-in-WiFi-3D và kiểm tra code.
- Phạm vi: code/config/test; người dùng đã yêu cầu push GitHub và pull vào checkout PiW riêng sau test. Không kill job hay sửa manuscript.
- Trạng thái: exploratory digital CSI/HPE adaptation, một seed; không phải full T1/T6, original-author reproduction hoặc kết quả efficacy.
- Giữ nguyên scientific source MM-Fi và mọi artifact/checkpoint lịch sử.

## Thứ tự chạy đã chốt với người dùng

**Một process PiW duy nhất**, chạy tuần tự HPE-Li → WiFlow → MetaFi++,
song song với **job WiFlow/MM-Fi đã có**. Không phải ba PiW jobs cùng lúc.
Sau khi fit attacker PiW100, queue cố định trong `plan.json` là:

1. HPE-Li Clean200 → HPE-Li BackTime200.
2. WiFlow Clean200 → WiFlow BackTime200.
3. MetaFi++ Clean200 → MetaFi++ BackTime200.

Mỗi victim khởi tạo riêng từ scratch. Clean và attacker không phải model
BackTime victim. HPE-Li-only không phải scope hiện hành. Khi một cell lỗi,
runner dừng; không retry hoặc chuyển qua model tiếp theo. Evaluation vẫn
đợi cả sáu final199, không dùng kết quả TEST để thay đổi các cell còn lại.
Không stop/suspend/restart/pull vào checkout của WiFlow/MM-Fi. GPU chia sẻ
có thể chậm hơn; ảnh nvidia-smi ít VRAM chưa chứng minh đủ capacity.

## Cấu hình đã khai báo trước khi chạy

PiW one-person: official TRAIN sau next-frame filter **27,876**, TEST **2,586**;
input **3x180x20**, output **1x14x3**, tọa độ mét. Native loader normalize
amplitude/phase một lần. Không đổi split hoặc cắt subset để lấy kết quả chính.

Một attacker HPE-Li mới được fit bằng TRAIN, sau đó dùng chung frozen trigger,
adaptive poison IDs và paired doses cho ba victim độc lập. Không mang attacker
MM-Fi/17 khớp hoặc checkpoint đã train sang PiW. Đây là **shared-surrogate
cross-backbone retraining**, không phải fit attacker riêng cho từng backbone.

Attacker: native alternating BackTime loop, 100 epochs, Adam surrogate 1e-4,
Adam generator .005, physical batch64, warmup10 theo điều kiện `epoch > warmup`
(11 clean + 89 alternating epochs), sparse variable fraction .3, eps .2,
alpha .5, lambda_norm .05, G scheduler milestones20/40 gamma.9. Chọn poison
một lần từ TRAIN online SmoothL1 top2K+1 pool; K=floor(.1*27876)=**2787**.
Generator cập nhật từng poison (batch1), không TEST-based selection/tuning.

Adapter PiW: graph gồm **270 amplitude variables** (3 receiver rows x 90
amplitude rows, gồm các physical groups30), không coi 90 là90 physical
subcarriers. Packet window20 chia **12 history + 8 replacement**, giữ tỉ lệ
60/40 của adapter MM-Fi. Graph FFT/statistics chỉ từ TRAIN amplitude.
Thay thế final packets theo native historical-anchor rule; không phải pattern
WiFiDoor, không copy eps.185 hoặc peak-cap khác. **90 phase rows giữ nguyên
từng bit**; không re-normalize sau trigger. Không claim physical/OTA feasibility.

Target giữ nguyên PiW body-hip-axis v2: pivot7 -> [11,13], axis normalize(GT5-GT7),
40deg*d; không pad14 thành17. Paired TRAIN dose U(.2,1), seed42, rho.1;
poison identities/doses freeze sau adaptive selection. Không dùng lại uniform
IDs của WiFiDoor rồi gọi native adaptive BackTime.

Ba **Clean controls mới** + ba **BackTime victims mới**, tất cả scratch,
200epochs/final199, B32, workers0, FP32, fixedLR, drop_lastTrue, ordinary all-joint
MPJPE, no pretrained/TEST checkpoint selection. Không có gradient clip mới.

| Backbone | Optimizer | LR | Weight decay | Accumulation / effective batch |
| --- | --- | --- | --- | --- |
| HPE-Li | AdamW | .001 | .01 | 1 / 32 |
| WiFlow | AdamW | .0001 | .0001 | 2 / 64 |
| MetaFi++ | AdamW | .01 | .01 | 1 / 32 |

AdamW betas(.9,.999), eps1e-8. MetaFi++ dùng recipe PiW đã sửa ở
`piw_metafi_clean_vastenv_3090.py`, KHÔNG dùng SGD của MM-Fi hoặc recipe cũ trong
`run_piw_two_models_attacks.py`. WiFlow flush accumulation tail với đúng số
mẫu của effective group cuối. Sáu victim train tuần tự trên một owned GPU
worker; không tự thay đổi capacity hoặc can thiệp các job ngoài.

Matched Clean được train lại ở output mới để tránh ghép checkpoint khác recipe,
backend hoặc provenance. Nếu muốn reuse Clean lịch sử để tiết kiệm compute,
cần thêm adapter audit/replay được duyệt riêng; hiện KHÔNG tự import.

TEST chỉ mở sau khi cả sáu victim đạt final199. Sáu dose0,.2,.4,.6,.8,1;
T-error/PCK endpoint1; Target-PCK .2*cleanGT distance(6,4); DCE mean5 positive
doses, clean MPJPE/PA all14, same-victim no-trigger controls, matched Clean+trigger,
triggered-GT errors, nontarget drift. Save full XYZ/GT/targets/IDs/doses/displacement
residuals; replay metrics và hashes trước COMPLETE. Không coi Target-PCK là pose ASR.

## File và lệnh trên Vân (chưa chạy tại đây)

Runner `ATKBackd/run_piw_backtime_three_backbones.py`; helper
`remote_tools/Start-PiWBackTimeThreeBackbones.ps1`. Chỉ cập nhật vào checkout
**idle**, không pull vào checkout đang có job chạy hoặc đã Prepare. DataHome
giữ dataset/env ở C:, output mới trên U:. Không cần action prior vì BackTime
learned trigger không dùng `data_bend.npy`.

Nếu dùng ZIP handoff, copy ZIP sang máy Vân và **giải nén vào folder mới**
`C:\Users\Van\PiWBackTimeThreeBackbones` (không ghi đè checkout đang train).
Bundle chứa code/config/docs, không chứa dataset/env/checkpoint. Dataset và
Python dùng lại ở `C:\Users\Van\backdooranalog` như ví dụ dưới; output ở U:.
Theo yêu cầu push/pull mới, sau khi commit được xác nhận trên GitHub, có thể
clone vào folder mới thay cho ZIP. Không pull vào checkout MM-Fi đang train.

```powershell
git clone --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git C:\Users\Van\PiWBackTimeThreeBackbones
```

Nếu folder PiW riêng đã là git checkout **idle, chưa Prepare** và không có
thay đổi local, dùng:

```powershell
git -C C:\Users\Van\PiWBackTimeThreeBackbones pull --ff-only origin main
```

Không xóa/ghi đè folder đã giải nén từ ZIP để ép clone hoặc pull;
nếu folder không phải git repo, chọn folder mới và dùng đường dẫn mới ở dưới.
`git pull` không phải launch. SSH bị từ chối thì assistant chưa pull trên Vân;
người vận hành chạy clone/pull trực tiếp trong terminal của máy Vân.

```powershell
$repo = 'C:\Users\Van\PiWBackTimeThreeBackbones'
$python = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
Set-Location $repo
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Inspect -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode CheckOnly -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Prepare -PythonExe $python
```

`Inspect` không đọc dataset/CUDA hoặc ghi output. `CheckOnly` full-data CPU audit,
khóa source/runtime/input/recipe vào plan. `Prepare` thêm disposable GPU smokes:
surrogateB64, generatorB1, từng modelB32 (WiFlow2 microbatches), cả clean/poison.
Không save smoke weights thành checkpoint train. Prepare không phải Launch.
Chạy Prepare trong lúc WiFlow/MM-Fi vẫn chạy để thử cùng điều kiện chia sẻ GPU.
Nếu fail/OOM, không Launch, không tự giảm batch hay dừng job MM-Fi để vượt gate.

Sau khi xem gate và chủ động quyết định chạy **100 attacker + 6x200 victim epochs**:

```powershell
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Launch -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Status -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Verify -PythonExe $python
```

Launch chạy foreground: giữ terminal/máy thức. Lock output mới, checkpoint atomic
mỗi epoch có optimizer/RNG/execution/checksum. Manual Launch sau interruption
resume checkpoint của chính run; không retry tự động. Không xóa stale lock để
vượt guard: kiểm tra PID/output trước rồi hỏi người vận hành. Không kill job,
shrink batch, AMP fallback hoặc đổi recipe sau Prepare. Nếu Prepare fail/OOM,
giữ log, không tiếp tục Launch.

Output mặc định `U:\backdooranalog\runs\piw_backtime_three_backbones_s42_v1`:
`plan.json`, `capacity.json`, `attacker/fit_checkpoint.pt`, `attacker/trigger.pt`,
`attacker/provenance.json`; `victims/<backbone>/{clean,backtime}/checkpoint.pt`,
và mỗi BackTime cell có `predictions.npz`/`evaluation.json`.

## Verification boundaries

Local synthetic/CPU unit tests không chứng nhận full official data trên Vân,
CUDA/VRAM, tốc độ hội tụ hoặc attack efficacy. Cần chạy CheckOnly/Prepare thật
trên checkout mới trước Launch. Không có kết quả TEST mới hoặc số paper nào
được tạo bởi việc setup này. Các recipe/target từng được xem TEST ở giai đoạn
phát triển; không gọi đây là blind evaluation đầu tiên.

Local gate 2026-11-10 sau khi khóa queue: **125 tests passed**, exit0 (34 PiW +91 BackTime
MM-Fi/runner/transfer regressions), 4 warnings pin_memory do CPU không có
accelerator. Gồm actual model CPU forward/backward của cả ba backbone (B2),
actual HPE-Li generator backward B1/native train-mode, phase/history/sparse
support invariance, 14-joint geometry, exact attacker resume, WiFlow tail
accumulation, source/config/capacity/lock guards, và full toy lifecycle
attacker100 +6victims200 +saved-metric replay/tamper rejection. Toy models/data
không thay thế native full-data convergence. PowerShell AST và helper Inspect
đạt, Inspect không tạo output. NumPy/MKL conflict của local Conda được cô lập
bằng MKL_THREADING_LAYER=SEQUENTIAL trước khi mở Python; không đổi metric hoặc
dùng công thức PA thay thế để vượt test. Scientific BackTime MM-Fi hash không
đổi so với HEAD; không chạy hay sửa các job MM-Fi trên Vân.

Hai test dispatch kiểm tra exact HPE-Li/Clean → HPE-Li/BackTime → WiFlow/Clean
→ WiFlow/BackTime → MetaFi++/Clean → MetaFi++/BackTime và fail-stop không retry/
mở TEST/chạy model tiếp. Capacity receipt từ chối đổi thứ tự, config binding
từ chối đảo queue. Đây là kiểm tra điều phối local, không phải launch thật.
SSH BatchMode tới máy Vân bị từ chối xác thực; chưa chạy Prepare/Launch từ đây.
