# Cross v3: hai vai + khuỷu + cổ tay, 6 target joints

## Material Passport

- academic-research-suite / experiment-agent; user-approved extension, 2026-11-08.
- V2 Geometry trên Vân dừng tại TRAIN21530: tổng hai xương123.273mm nhỏ hơn
  khoảng cách vai-mid-plane124.331mm. Đây là bất khả thi khi vai cố định, không
  phải lỗi CUDA. Người dùng duyệt cho vai chuyển động và mở rộng4->6 target.
- Cả MM-Fi/PiW HPE-Li, seed42/rho.1. Giữ native recipes/trigger/poison/doses/
  metric definitions, Nod v1, hai slot/GPU và immediate evaluation từng job.
- Code/test/push handoff, không tự launch trên Vân. Không sửa/xóa raw labels,
  checkpoint, failed outputs v1/v2, ZIP cũ hoặc paper/PDF. Không claim full T7.

## 1. Target và bone constraints mới

| Dataset | Cross target joints | Fixed anchors | Nod không đổi |
| --- | --- | --- | --- |
| MM-Fi |11,12,13,14,15,16|thorax8|neck9 -> head10,40*d|
| PiW |3,6,10,2,4,8|neck0,hips7/5|neck0 -> head1,40*d|

Cross spec version: mmfi_action_cross_geometry_v3 / piw_action_cross_geometry_v3.
T-MPJPE, Target-PCK, DCE và target-joint clean error dùng đủ6khớp; không ghép
với số Cross4khớp v1/v2 như thể cùng metric/target. Clean MPJPE/PA vẫn17/14
khớp; PCK scale vẫn cleanGT MM-Fi(5,12), PiW(6,4). Full six-dose XYZ/ordered IDs
được giữ. Không sửa baseline/clean-trained controls để làm đẹp bảng.

Body frame, chest goals và crossing/front checks luôn lấy từ CLEAN pose,
không lấy mid-plane mới của hai vai đã di chuyển. Vai vẫn phải ở phía ban đầu;
cổ tay tại d1 vượt sang phía đối diện và nằm phía trước clean torso.

- MM-Fi: mỗi vai quay trên vòng tròn quanh trục body-up qua thorax8, giữ
  khoảng cách tới8. Các khớp thân/đầu/chân không di chuyển.
- PiW: vai3 nối neck0 và hip7; vai2 nối neck0 và hip5. Mỗi vai di chuyển trên
  vòng tròn giao hai mặt cầu tâm neck/hip: quay quanh đường neck-hip. Nhờ vậy
  giữ được CẢ neck-shoulder và shoulder-hip, không làm biến dạng xương thân.
- Orbit hướng tới điểm có forward projection lớn nhất. Các nominal fractions
  cố định .2,.3,...,1; dùng candidate đầu tiên có target hợp lệ. Có thêm cực
  trị lateral giải tích và bisection60bước cho khoảng h mục tiêu để không bỏ
  lỡ vùng reachable hẹp giữa các nominal candidates. Đây là phép giải label
  theo geometry, không phải chọn hyperparameters/checkpoints bằng TEST scores.
- Từ vai mới, two-bone IK đặt elbow/wrist theo tầm với từng tay và strict
  crossing/front semantics. Không kéo dài xương, đổi poison IDs hoặc bỏ pose.
- Ở dose trung gian: vai đi đúng orbit với angle*d; từng hướng xương tay được
  nội suy bằng great-circle rotation. Tất cả bone edges giữ độ dài ở mọi dose.
  Dose0 identity; mọi khớp ngoài6target giữ nguyên.

Vai bị hai anchors pin tại một điểm vẫn có thể không chuyển động; audit ghi
số vai thực sự di chuyển, không giả rằng mọi target joint đều luôn dịch chuyển.
Degenerate bones hoặc không có candidate hợp lệ vẫn là lỗi. Không chứng minh
joint limits, self-collision/contact, anatomical plausibility hoặc action
recognition. Không áp định lý rigid40deg/near-linear/monotonic của payload cũ.

## 2. Audit toàn bộ trước khi train

Mỗi dataset kiểm đủ Cross/Nod và mọi selected poison TRAIN pose tại sáu dose,
float64 cả reference/target, bone threshold1e-8m không đổi. Geometric
ValueError/AssertionError được tổng hợp, không dừng ở lỗi đầu tiên. I/O hoặc
lỗi chương trình bất ngờ vẫn dừng ngay và không được gọi là audit hoàn tất.

Mỗi native output có:

```text
geometry_audit.json       PASS/FAILED, n_checked/n_failed từng action
geometry_failures.json    toàn bộ geometric failures nếu có; không chứa raw XYZ
geometry_failure_cross.json / geometry_failure_nod.json  lỗi đầu để tương thích
geometry.json            CHỈ tồn tại khi mọi pose/dose đều đạt; bind audit SHA
```

Cross reach_adaptation ghi moved_shoulders, max displacement/rotation, số tay
không thể cross với vai cũ, goal adaptations và min crossing/front clearance.
Nontarget/bone validation và poison denominators không bị thay đổi khi có lỗi.

Coordinator cho chạy tiếp PiW CPU audit sau MM-Fi chỉ khi worker của CHÍNH
session hiện tại attests một complete FAILED audit với queue/plan/audit SHA
đúng. Unknown crash, stale/foreign/partial report, interrupt hoặc disk error
vẫn chặn việc mở stage sau. Đây chỉ là thu thập hai CPU audit, KHÔNG cho phép
tiếp tục Prepare/training sau lỗi. Nếu còn bất kỳ failure nào: ghi root
geometry_audit_failures.json rồi dừng, không có success/prepared receipt.

## 3. Pull và chạy Geometry trên Vân

Không pull nếu một training job dùng cùng checkout vẫn đang chạy. Copy cả
khối dưới; dấu chấm ở đầu giúp giữ biến trong terminal cho Prepare/Launch sau.
Output v3 phải mới; không xóa/đổi tên/chép file từ v1/v2.

```powershell
. {
    $ErrorActionPreference = 'Stop'
    Set-Location -LiteralPath 'C:\Users\Van\PiWMetaFiTwoExtra4090'
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Pull failed; preserve local changes' }
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    $CrossNodRunner = '.\remote_tools\action_targets_parallel\Run-CrossNodTwoJobs.ps1'
    $CrossNodArgs = @{
        DataHome = 'C:\Users\Van\backdooranalog'
        Python = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
        Outdir = 'U:\backdooranalog\runs\cross_nod_mmfi_piw_s42_2jobs_v3'
        ApproveActionLabels = $true
    }
    & $CrossNodRunner -Mode Geometry @CrossNodArgs
}
```

Hai native folders đều dùng suffix v3 dưới OUT\runs. Xem bốn ảnh TRAIN
geometry_cross.png/geometry_nod.png sau GEOMETRY_ONLY. Nếu failed, gửi report
hoặc child log; không tự retry, xóa lock/output hoặc bỏ mẫu. Sau khi đã xem
đúng cả hai dataset mới chạy Prepare -ApproveGeometry; chỉ sau
PARALLEL_PREPARED_ONLY mới Launch. Lịch vẫn MM-Fi Cross+Nod rồi PiW Cross+Nod,
hai slot gồm cả training/evaluation; evaluate ngay từng final49/final199.

## 4. Verification limits

Gate cuối:165passed/4deselected, exit0,267.77s; gồm complete-audit plumbing và
36coordinator tests. Bốn native HPE-Li B32 CPU cases không chạy lại vì giới hạn
bộ nhớ local; không tính là pass. Regression riêng144passed/1deselected cho
adapter/split/recipe (thêm một native HPE-Li CPU case không chạy); tổng hai gate
309passed/5deselected, chưa thay CUDA Prepare. Core initial gate126passed/5deselected trước
complete-audit không thay gate cuối; xem VALIDATION.md. Pasted real TRAIN9092
được kiểm local/in-memory: sáu dose đạt, max bone delta2.22e-16m, Nod bitwise
như95d7c50; vai dịch29.99/45.31mm. Raw pose không được đưa lên Git. Real XYZ
của TRAIN21530 chưa được cung cấp: chỉ synthetic analogue với xương107.6mm/
15.7mm và mid-plane130mm đã được kiểm, không gọi đó là replay frame21530.
Full poison TRAIN audit, geometry inspection và CUDA B32/concurrent Prepare
trên Vân vẫn bắt buộc. Chưa có kết quả train/TEST/qualitative empirical mới.
