# PiW BackTime v4: train trước, Clean chỉ để đánh giá — 2026-11-10

## Material Passport — hiện hành

- Người dùng yêu cầu tận dụng phần đã chạy, không đợi đủ ba Clean, giữ thứ tự
  **HPE-Li → WiFlow → MetaFi++**. Một worker PiW, không can thiệp job MM-Fi.
- Attacker PiW v1 đã được người dùng xác nhận đủ100/100 và có trigger.pt.
  Đọc trực tiếp attacker nguồn, không fit lại hoặc chuyển trọng số surrogate
  sang victim. Các hash/plan/poison IDs/doses/source audits vẫn bắt buộc.
- Clean HPE-Li v3 đã qua preflight trên Vân theo output người dùng: final199,
  MPJPE92.370288mm/PA58.454966mm. Đây không phải assistant replay TEST mới.
- Clean không là dependency để train. Thiếu/không tương thích/nhập nhằng khi
  tự dò chỉ ghi **EVALUATION_PENDING_CLEAN**. Nguồn chỉ rõ bằng tham số nhưng
  sai vẫn báo lỗi; không tự chọn bản khác, đổi recipe hoặc train Clean mới.
- Ba victim BackTime độc lập, scratch200/final199, cùng recipe đã khóa bên
  dưới. Zero Clean training. Target vẫn pivot7->[11,13]/hip-axis40d, không
  đổi sang payload trunk Bend. Không sửa bài hoặc artifact lịch sử.

## Chạy một lần trên Vân

Dùng checkout/output **v4 mới trên U:** vì C: đã đầy. Không pull vào checkout
đang chạy hoặc dùng output v1/v2/v3 đã khóa. Chỉ dùng lệnh clone sau khi bản v4
đã được xác nhận push; cập nhật local chưa làm thay đổi GitHub/máy Vân.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    New-Item -ItemType Directory -Path 'U:\backdooranalog\code' -Force | Out-Null
    New-Item -ItemType Directory -Path 'U:\backdooranalog\setup_tmp' -Force | Out-Null
    $env:TEMP = 'U:\backdooranalog\setup_tmp'
    $env:TMP = 'U:\backdooranalog\setup_tmp'
    $piwRepo = 'U:\backdooranalog\code\PiWTrainFirstV4_' + [guid]::NewGuid().ToString('N').Substring(0,8)
    git clone --depth 1 --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git $piwRepo
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed; stop here.' }
    Write-Output "PiW v4 checkout: $piwRepo"
    & "$piwRepo\remote_tools\Start-PiWBackTimeThreeBackbones.ps1" -Mode Start `
        -DataHome 'C:\Users\Van\backdooranalog' `
        -AttackerRun 'U:\backdooranalog\runs\piw_backtime_three_backbones_s42_v1' `
        -CleanHPELi 'C:\Users\Van\backdooranalog\runs\piw_hpeli_body_axis_s42_lr001_v3'
}
```

**Start là lệnh chạy train rõ ràng**, không phải Inspect: CPU/data/source audit
→ disposable GPU capacity smoke nếu chưa có receipt → reuse frozen attacker
→ HPE-Li200 → WiFlow200 → MetaFi++200. Giữ terminal mở và Windows không sleep.
Không retry OOM, giảm batch, kill job khác hoặc fit lại khi attacker import lỗi.
Capacity smoke không phải một run Clean200/attacker100 mới. Receipt đã có được
kiểm binding/execution, không tự chạy lại. Inspect vẫn là mặc định không train.

Output mặc định: `U:\backdooranalog\runs\piw_backtime_three_backbones_train_first_s42_v4`.
Resume dùng cùng checkout và options, gọi lại **Start**: final199 hợp lệ được
skip trước khi dựng model/G/dataloader; partial checkpoint resume model,
optimizer và RNG đúng epoch kế tiếp. Không migrate hoặc relabel victim v1/v3
thành v4. Không xóa lock để vượt guard; stale lock của attacker v1 giữ nguyên.

## Đánh giá và thêm Clean sau — không train lại victim

Giữ nguyên deferred-TEST: chỉ mở prediction arrays/chấm TEST sau **cả ba victim
mới final199**. Có Clean nào đã audit thì đánh giá backbone đó. Ví dụ chỉ có
Clean HPE-Li: vẫn train đủ ba, evaluate HPE-Li, báo hai mục còn chờ và
`ALL_THREE_TRAINED_EVALUATION_PARTIAL`, không gọi là full comparison COMPLETE.
`ALL_THREE_COMPLETE` chỉ in khi đủ ba kết quả đã replay.

Khi tìm được Clean WiFlow/MetaFi++ tương thích, dùng helper **cùng checkout**:

```powershell
# $piwRepo là checkout v4 đã in ở trên; RUN_DIR là thư mục run Clean gốc.
& "$piwRepo\remote_tools\Start-PiWBackTimeThreeBackbones.ps1" -Mode Evaluate `
    -DataHome 'C:\Users\Van\backdooranalog' `
    -CleanWiFlow 'RUN_DIR_CLEAN_WIFLOW' -CleanMetaFi 'RUN_DIR_CLEAN_METAFI'
```

Evaluate chỉ load/evaluate, không gọi fit/train. Late refs audit cùng full
inputs/source/recipe/final199/weights rồi lưu sidecar riêng
`evaluation_references/<backbone>.json` gắn hash training plan; không sửa plan,
victim hoặc kết quả đã đánh giá. Không thay thế một reference đã freeze hoặc
tự chọn lại nguồn theo TEST score. Thiếu nguồn vẫn PENDING, không tự train bù.
Verify replay các kết quả có Clean, giữ rõ mục thiếu. Status chỉ đọc metadata.

## Kiểm thử v4

**189 passed**, exit0,259.04s: 98PiW +91hồi quy BackTime/MM-Fi/transfer;
bốn cảnh báo CPU pin_memory dự kiến. Gồm ba backbone thật CPU backward,
native toy attacker100 reuse, chỉ có Clean HPE-Li vẫn train ba toy200,
không refit/train Clean, skip final199 trước model/G/data, exact resume
epoch73 (model/optimizer/RNG), thêm Clean sau mà không đổi plan/victim/kết quả,
và từ chối binding/tamper. PowerShell AST, Inspect không ghi output, và guard
Start không có AttackerRun đạt. Không phải full-data/CUDA/efficacy trên Vân;
Start vẫn phải audit attacker nguồn và kiểm tra capacity thật. MM-Fi core và
manuscript giữ nguyên.

<details>
<summary>Lịch sử v3 bên dưới — không dùng lệnh cũ để chạy v4</summary>

## PiW BackTime v3 — ghi chú lịch sử, đã được v4 thay thế về Clean dependency

## Material Passport

- Yêu cầu: chuẩn bị BackTime cho ba backbone trên Person-in-WiFi-3D và kiểm tra code.
- Phạm vi: code/config/test; người dùng đã yêu cầu push GitHub và pull vào checkout PiW riêng sau test. Không kill job hay sửa manuscript.
- Trạng thái: exploratory digital CSI/HPE adaptation, một seed; không phải full T1/T6, original-author reproduction hoặc kết quả efficacy.
- Giữ nguyên scientific source MM-Fi và mọi artifact/checkpoint lịch sử.
- Sửa theo yêu cầu mới: **không train lại Clean**. Ba Clean nguồn bắt buộc phải được audit; thiếu/lệch thì dừng, không tự train bù. Người dùng đã yêu cầu publish; chưa cập nhật/launch trên máy Vân.

## Thứ tự chạy đã chốt với người dùng

**Nếu job v1 đang chạy:** không pull/sửa checkout, không đổi `plan.json`, không
xóa lock và không chạy v3 cùng output. Push GitHub không đổi queue trong process
v1. Không tự dừng job. User đã xác nhận dừng riêng PID26052 ngày2026-11-10;
attacker của run đó đã lưu100/100 và có `trigger.pt`. v3 chỉ hỗ trợ dùng lại
**attacker đã hoàn tất** read-only, không chuyển optimizer/RNG hoặc victim v1.

**Một process PiW duy nhất**, chạy tuần tự HPE-Li → WiFlow → MetaFi++,
song song với **job WiFlow/MM-Fi đã có**. Không phải ba PiW jobs cùng lúc.
Sau khi fit attacker PiW100 **hoặc import attacker100 đã hoàn tất**, queue cố định là:

1. HPE-Li BackTime200, dùng Clean HPE-Li đã có.
2. WiFlow BackTime200, dùng Clean WiFlow đã có.
3. MetaFi++ BackTime200, dùng Clean MetaFi++ đã có.

Mỗi victim khởi tạo riêng từ scratch. Clean và attacker không phải model
BackTime victim. HPE-Li-only không phải scope hiện hành. Khi một cell lỗi,
runner dừng; không retry hoặc chuyển qua model tiếp theo. Evaluation vẫn
đợi cả ba victim mới đạt final199, không dùng kết quả TEST để thay đổi các cell còn lại.
Không stop/suspend/restart/pull vào checkout của WiFlow/MM-Fi. GPU chia sẻ
có thể chậm hơn; ảnh nvidia-smi ít VRAM chưa chứng minh đủ capacity.

## Cấu hình đã khai báo trước khi chạy

PiW one-person: official TRAIN sau next-frame filter **27,876**, TEST **2,586**;
input **3x180x20**, output **1x14x3**, tọa độ mét. Native loader normalize
amplitude/phase một lần. Không đổi split hoặc cắt subset để lấy kết quả chính.

Một attacker HPE-Li được fit một lần bằng TRAIN (ở v1 hoặc fresh), sau đó dùng
chung frozen trigger, adaptive poison IDs và paired doses cho ba victim độc lập.
Không mang attacker MM-Fi/17 khớp hoặc surrogate/victim weights sang PiW. Đây là **shared-surrogate
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

Ba **Clean controls lịch sử, read-only** + ba **BackTime victims mới từ scratch**,
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
mẫu của effective group cuối. Ba victim mới train tuần tự trên một owned GPU
worker; không tự thay đổi capacity hoặc can thiệp các job ngoài.

Clean không được train, fine-tune, sửa file hay truyền optimizer/RNG/weights sang
victim. Chỉ load weights để inference. Mặc định tự dò **metadata**, sâu một cấp
trong `DataHome/runs` và thư mục cha của output (mặc định U:/backdooranalog/runs),
chỉ các thư mục PiW; không quét toàn ổ đĩa, không đọc weights/TEST khi FindClean.
Preflight chỉ dùng nguồn duy nhất khớp; nhiều nguồn cùng khớp thì liệt kê và dừng.
Không chọn theo điểm TEST, tên mới nhất hoặc epoch tốt nhất. Có thể chỉ rõ
**thư mục run nguồn** (không chỉ một file `.pt`) để giải quyết nhập nhằng.
Preflight kiểm manifest checksum,
full input/split/order fingerprints, seed42/final199, optimizer/LR/batch/accumulation,
drop_last/AMP, source của model và preprocessing, checkpoint/evaluation/NPZ hashes.
Original runtime/GPU được giữ trong provenance; không giả thành training mới.

Sau khi cả ba victim xong, chạy lại Clean + trigger BackTime mới ở sáu dose.
Replay GT/IDs và clean metrics nguồn, đối chiếu clean prediction mới với cache
nguồn (tối đa 0.05 mm mỗi tọa độ). Sai khác vượt ngưỡng cần kiểm tra backend/source,
không tự train lại để vượt gate. TEST arrays nguồn không mở ở preflight.

Layouts được hỗ trợ: HPE-Li `piw_lr_control.resolved.json` (v3), HPE-Li
`piw_hpeli_clean.resolved.json` và các PiW two-backbone `plan.json` với cell
`attacks/<backbone>/seed42/clean`. Có layout không có nghĩa recipe khớp:
HPE-Li Clean v1 dùng drop_lastFalse, bị từ chối với victim hiện dùng True;
MetaFi++ SGD hoặc model/pretrained khác cũng bị từ chối. Không tự đổi victim recipe.

TEST chỉ mở sau khi cả ba victim mới đạt final199. Sáu dose0,.2,.4,.6,.8,1;
T-error/PCK endpoint1; Target-PCK .2*cleanGT distance(6,4); DCE mean5 positive
doses, clean MPJPE/PA all14, same-victim no-trigger controls, matched Clean+trigger,
triggered-GT errors, nontarget drift. Save full XYZ/GT/targets/IDs/doses/displacement
residuals; replay metrics và hashes trước COMPLETE. Không coi Target-PCK là pose ASR.

## File và lệnh trên Vân (chưa chạy tại đây)

### Chuyển từ v1 đã dừng, giữ attacker100 đã chạy

Giữ nguyên cả checkout và output v1. Dùng folder **mới** (không cần xóa stale
lock v1). Chỉ thực hiện sau khi bản cập nhật được xác nhận đã push:

```powershell
git clone --branch main --single-branch https://github.com/toan31072006/backdoor-analog.git C:\Users\Van\PiWBackTimeReuseV3_20261110
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; stop here. Do not overwrite an existing folder.' }
$piwHelper = 'C:\Users\Van\PiWBackTimeReuseV3_20261110\remote_tools\Start-PiWBackTimeThreeBackbones.ps1'
$piwOptions = @{
    DataHome = 'C:\Users\Van\backdooranalog'
    PythonExe = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
    AttackerRun = 'U:\backdooranalog\runs\piw_backtime_three_backbones_s42_v1'
    Outdir = 'U:\backdooranalog\runs\piw_backtime_three_backbones_reuse_clean_attacker_s42_v3'
}
& $piwHelper -Mode FindClean @piwOptions
```

Gửi output FindClean để xác nhận ba Clean nguồn. Lệnh này chỉ đọc metadata,
không dùng CUDA/TEST hoặc train. Không tự chuyển thẳng sang Launch.
Sau khi kiểm tra danh sách nguồn (và chỉ rõ `CleanHPELi`, `CleanWiFlow`,
`CleanMetaFi` nếu nhiều nguồn khớp), giữ cùng `piwOptions` cho mọi bước:

```powershell
& $piwHelper -Mode CheckOnly @piwOptions
if ($LASTEXITCODE -ne 0) { throw 'Audit failed; no Prepare or Launch.' }
# Chỉ sau khi CheckOnly pass và người vận hành quyết định thử capacity:
& $piwHelper -Mode Prepare @piwOptions
if ($LASTEXITCODE -ne 0) { throw 'Capacity failed; no Launch.' }
# Launch là quyết định riêng, sau khi xem cả hai gate:
# & $piwHelper -Mode Launch @piwOptions
```

CheckOnly xác minh source v1 đã publish, source khoa học không đổi, full
input/split/order fingerprints, seed/rho/target/config, plan/capacity receipt,
frozen-state checksum và provenance, đủ100 epoch/updates,81 biến, poison IDs
và doses. Nguồn đang chạy hoặc thiếu/sai sẽ dừng; **không tự fit lại**.
Plan/evaluation v3 lưu original runtime/execution và mọi source hash. Trigger
được đọc trực tiếp ở run v1; không tạo `attacker/trigger.pt` giả mang hash v3.
Không import `fit_checkpoint.pt`, optimizer, RNG, surrogate hoặc victim weights.
Stale lock v1 được đọc kiểm tra PID rồi giữ nguyên; không xóa.

Với `AttackerRun`, chỉ còn **3×200 epoch victim BackTime**, không100 attacker
và không3 Clean200. Các batch GPU smoke khi Prepare vẫn là model tạm, không
phải train một Clean đối chứng hoặc fit lại attacker100. Thiếu/lệch Clean nguồn
vẫn dừng, không tự đổi recipe để nhận checkpoint cũ.

### Trường hợp fresh attacker (không phải lượt chuyển của người dùng hiện tại)

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
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode FindClean -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Inspect -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode CheckOnly -PythonExe $python
.\remote_tools\Start-PiWBackTimeThreeBackbones.ps1 -Mode Prepare -PythonExe $python
```

Không cần nhớ đường dẫn Clean để bắt đầu: FindClean chỉ liệt kê metadata,
CheckOnly mới audit đầy đủ và freeze nguồn. Nếu có nhiều nguồn khớp, dùng
`-CleanHPELi RUN_DIR`, `-CleanWiFlow RUN_DIR`, `-CleanMetaFi RUN_DIR` theo danh sách
đã tìm được; giữ cùng options khi Prepare/Launch/Verify.
Python tương đương: lặp `--clean-reference hpeli=RUN_DIR`,
`--clean-reference wiflow=RUN_DIR`, `--clean-reference metafiplusplus=RUN_DIR`.
Dùng **output v3 mới**, không sửa/ghi đè plan v1/v2 đã Prepare. Nguồn đã freeze
không được tự chọn lại khi xuất hiện run mới hoặc resume.

`Inspect` không đọc dataset/CUDA hoặc ghi output. `CheckOnly` full-data CPU audit,
khóa source/runtime/input/recipe vào plan. `Prepare` thêm disposable GPU smokes:
surrogateB64, generatorB1, từng modelB32 (WiFlow2 microbatches), cả clean/poison.
Không save smoke weights thành checkpoint train. Prepare không phải Launch.
Chạy Prepare trong lúc WiFlow/MM-Fi vẫn chạy để thử cùng điều kiện chia sẻ GPU.
Nếu fail/OOM, không Launch, không tự giảm batch hay dừng job MM-Fi để vượt gate.

Sau khi xem gate và chủ động quyết định chạy **100 attacker + 3x200 BackTime epochs**:

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

Output mặc định `U:\backdooranalog\runs\piw_backtime_three_backbones_reuse_clean_attacker_s42_v3`:
`plan.json`, `capacity.json`, `attacker/fit_checkpoint.pt`, `attacker/trigger.pt`,
`attacker/provenance.json`; `victims/<backbone>/backtime/checkpoint.pt`,
và mỗi BackTime cell có `predictions.npz`/`evaluation.json`.
Không tạo `victims/<backbone>/clean`. `plan.json` chứa original reference paths,
hashes, source profile/plan/runtime/execution. Khi import attacker, không tạo
attacker local; `attacker_reference` trỏ file v1 với original binding, không ghi
đè hoặc relabel. Evaluation chứa audit/replay nguồn;
không đổi provenance của Clean thành provenance của victim mới.

## Verification boundaries

Local synthetic/CPU unit tests không chứng nhận full official data trên Vân,
CUDA/VRAM, tốc độ hội tụ hoặc attack efficacy. Cần chạy CheckOnly/Prepare thật
trên checkout mới trước Launch. Không có kết quả TEST mới hoặc số paper nào
được tạo bởi việc setup này. Các recipe/target từng được xem TEST ở giai đoạn
phát triển; không gọi đây là blind evaluation đầu tiên.

Gate v1 lịch sử 2026-11-10 trước sửa reuse: **125 tests passed**, exit0 (34 PiW +91 BackTime
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

Hai test dispatch v1 trước đây kiểm tra exact HPE-Li/Clean → HPE-Li/BackTime → WiFlow/Clean
→ WiFlow/BackTime → MetaFi++/Clean → MetaFi++/BackTime và fail-stop không retry/
mở TEST/chạy model tiếp. Capacity receipt từ chối đổi thứ tự, config binding
từ chối đảo queue. Đây là kiểm tra điều phối local, không phải launch thật.
SSH BatchMode tới máy Vân bị từ chối xác thực; chưa chạy Prepare/Launch từ đây.

Gate v2 reuse kiểm tra queue chỉ ba BackTime, hard rejection của mọi lệnh train
Clean, missing-reference fail trước data/attacker, read-only source hashes,
recipe/partial/poison/wrong-model rejection, deferred TEST arrays và native
attacker100 + ba toy victims200 với saved-metric/reference replay. Xem kết quả
test mới trong amendment của experiment plan; không suy CPU toy thành full PiW/CUDA.
Gate v2: **150 passed**, exit0,105.35s (59PiW +91regression), bốn CPU pin_memory
warnings như dự kiến. PowerShell AST/Inspect/FindClean pass, bao gồm forwarding
paths có dấu cách, không ghi output. Gate này không phải audit artifact/CUDA thật trên Vân.

Gate v3 kiểm tra import completed attacker100 read-only, strict original-source/
plan/capacity/config/full-input/provenance/frozen-hash binding, đủ100 epoch và
counters/IDs/doses, source PID active/stale-lock, source tamper/partial/foreign
rejection, no-refit/no-GPU/no-TEST import, new signed-plan origin/resume,
PowerShell forwarding AttackerRun có dấu cách và metadata-only FindClean.
Native CPU toy attacker100 được dùng lại cho ba toy BackTime victims200;
generator/IDs/doses giữ nguyên và source files không đổi. Không phải audit
checkpoint thật trên Vân hoặc xác nhận compatibility của ba Clean thật.
Kết quả cuối: **178 passed**, exit0,135.29s (87PiW +91hồi quy), bốn CPU
pin_memory warnings dự kiến. PowerShell AST/Inspect có AttackerRun chứa dấu
cách và FindClean pass, không ghi output. MM-Fi source/manuscript không đổi.

</details>
