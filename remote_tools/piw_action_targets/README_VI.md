# PiW3D: hai target mới Cross và Nod — HPE-Li, seed42

**Cross reach v2 (08/11):** Git dùng cùng quy tắc goal thích nghi như MM-Fi,
giữ vai/xương, Nod và recipe. Dùng output mới v2, không sửa plan/output v1;
ZIP07/11 giữ nguyên. Xem [quy tắc và lệnh khôi phục](../action_targets_parallel/CROSS_REACH_V2_VI.md).

**Cập nhật 2026-11-08:** để chạy CẢ MM-Fi/PiW với2job/1GPU và evaluate ngay từng
job, dùng [coordinator mới](../action_targets_parallel/README_VI.md). File này
giữ hướng dẫn serial của gói ZIP cũ, không trộn hai launcher/output.

## Material Passport

- Phạm vi: chuẩn bị code cho hai target hành động mới trên Person-in-WiFi-3D
  one-person; không phải bảng thay motion prior, full T7 hay kết quả đã chạy.
- Nguồn: checkout handoff `backtime_4090_setup_20261106`; không thay source
  khoa học PiW/MM-Fi cũ, checkpoint cũ hoặc bản ZIP MM-Fi đã giao.
- Trạng thái: xem `VALIDATION.md` cho kiểm thử local. Chưa xác minh geometry
  trên toàn dữ liệu thật, VRAM CUDA, convergence hoặc efficacy Cross/Nod.
- Không cấp quyền tự launch, dừng job, push Git hay đưa kết quả giả vào paper.

## 1. Những gì khác MM-Fi

Giữ official lists: TRAIN one-person28121 còn27876 sau next-frame filter;
TEST2586 không lọc next-frame. Input `[3,180,20]`, label `[1,14,3]`, đơn vị mét.
Không tạo split mới, không gọi đây là cross-environment vì capture prefixes
có thể chung giữa TRAIN/TEST, dù frame identities không trùng.

Recipe **PiW HPE-Li v3**: hai victim scratch độc lập, chạy tuần tự Cross rồi Nod,
**mỗi victim200epoch/final199**, AdamW lr.001/wd.01/betas(.9,.999)/eps1e-8,
batch32, workers0, drop_last=True, FP32, all-joint MPJPE, fixed LR. Không copy
SGD50epoch của MM-Fi hoặc lr.01 của PiW v2. Không warm-start Bend/Clean.
TRAIN27876 giữ nguyên trong dataset; mỗi epoch shuffle tiêu thụ27872 mẫu do
drop_last=True, không tự thay batch hoặc lọc thêm frame.

Poison10% =2787 TRAIN frames, native RNG42 chọn uniform không hoàn lại,
paired doses U(.2,1) giữ cố định. Native A1 eps.185, pattern `[3,30,20]` lặp
trên từng block30 trong90 hàng amplitude. Normalize amplitude/phase đúng một
lần bằng loader cũ, inject sau đó;90 hàng phase giữ nguyên. Không coi180 hay90
hàng feature là số physical subcarriers. Đây là digital poisoning, chưa over-air.

## 2. Target mới, không chỉ đổi trigger action

Layout PiW: neck0/head1; left shoulder2/elbow4/hand8/hip5/knee9/ankle12;
right shoulder3/elbow6/hand10/hip7/knee11/ankle13.

- **Cross:** hai tay bắt chéo phía trước ngực; vai3/2 cố định, target joints
  `[6,10,4,8]`. Cùng tham số tương đối như MM-Fi, nhưng torso/frame lấy từ
  midpoint hai vai và midpoint hai hông; đây là điểm ảo hình học, không phải
  khớp annotation bổ sung. Hai-bone IK tìm endpoint, great-circle interpolation
  từng hướng xương tạo target tại dose trung gian. Cross v2 giữ nominal goal
  khi reachable; nếu không, giữ lateral crossing trước rồi chỉ giới hạn bán
  kính height/front theo tầm với từng tay. Vẫn báo lỗi nếu không thể qua
  midline/phía trước ngực; không bỏ frame hoặc tune trên TEST.
- **Nod:** chỉ head1 xoay quanh neck0, góc40*d, theo trục ngang hai vai đã bỏ
  thành phần dọc torso. Mọi khớp khác giữ nguyên.
- Forward=`cross(lateral,up)` là quy ước cố định, phải xem geometry TRAIN trước
  khi duyệt. Không suy hướng từ TEST metric. Nếu không đúng ý, dừng trước
  Prepare; sửa geometry có version mới/output mới, không reuse plan cũ.
- Cross thay góc khuỷu tay, **không phải** payload xoay rigid một subtree của
  main paper; không áp định lý near-linear/monotonic từ payload cũ cho Cross.
  Bảo toàn độ dài xương không chứng minh joint limits/collision/contact/action
  recognition. Nod40degree cũng là định nghĩa target, không bảo đảm victim học được.

Old Bend/Cross/Nod motion-prior results chỉ đổi CSI trigger nhưng vẫn xoay
chân. Đừng dùng chúng làm qualitative của ba action-target. Old Bend cũng
chưa tự thành target cúi thân. Đây là hai thí nghiệm bổ sung riêng.

## 3. Giải nén riêng; CheckOnly không chạy train

Giải nén ZIP PiW vào **thư mục source mới**, không đè checkout/job đang chạy.
Ví dụ PowerShell trên máy Vân:

```powershell
Expand-Archive -LiteralPath 'C:\Users\Van\Downloads\PiWCrossNodTargets_s42_code_20261107.zip' -DestinationPath 'C:\Users\Van\Downloads\PiWCrossNodTargets_s42_code_20261107'
Set-Location 'C:\Users\Van\Downloads\PiWCrossNodTargets_s42_code_20261107'

$taskPiwArgs = @{
    DataHome = 'C:\Users\Van\backdooranalog'
    Python = 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe'
}
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode CheckOnly @taskPiwArgs
```

Nếu env Python hoặc dataset ở chỗ khác, sửa `Python`/thêm `DatasetRoot` vào
hashtable trên. Dataset mặc định `DataHome\datasets\Person-in-WiFi-3D_processed_v1`.
Phải có `train_data/test_data` với `csi_ap/keypoint` và official lists, không
thay bằng ZIP raw khác schema. Cross/Nod priors mặc định
`DataHome\actions\data_cross.npy`, `data_nod.npy`; dùng CrossAction/NodAction
nếu ở đường dẫn khác. Không tạo prior giả hay copy Bend đổi tên.

## 4. Geometry TRAIN-only, rồi Prepare tách biệt

Nếu chưa có sidecar `.npy.provenance.json` hợp lệ, **chỉ sau khi bạn xác nhận
hai file thật đúng action Cross/Nod**, thêm:

```powershell
$taskPiwArgs.ApproveActionLabels = $true
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode Geometry @taskPiwArgs
```

Geometry đọc TRAIN labels cho toàn bộ2787poison IDs/sáu dose, kiểm bone lengths,
non-target exact, lưu số bước giảm requested displacement (không lọc/tune chúng).
Chọn first/middle/last sorted poison IDs để xem ảnh, không chọn theo score.
Ảnh PNG/PDF và NPZ nằm tại:

```text
DataHome\runs\piw_action_targets_cross_nod_s42_v2\geometry_cross.png
DataHome\runs\piw_action_targets_cross_nod_s42_v2\geometry_nod.png
```

Đây chỉ là geometry GT/target, chưa phải qualitative của model. Sau khi xem
đúng hai tay bắt chéo trước ngực và đúng hướng cúi đầu, chạy:

```powershell
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode Prepare -ApproveGeometry @taskPiwArgs
```

Prepare cần4090 rảnh: full-content audit official lists và mọi one-person CSI/
pose, kiểm geometry không đổi, real CUDA B32 TRAIN smoke8poison+24clean/action.
Smoke weights bỏ đi, không làm checkpoint; **Prepare không launch training**.
Không kill/pause hoặc bypass GPU job khác. Nếu fail, giữ log/artifact và xem
lỗi, không tự retry/giảm batch/đổi AMP. Dùng output v2/v3 mới khi cần đổi plan.

## 5. Chỉ launch khi bạn muốn chạy thật

Lệnh dưới khởi động hai job tuần tự, tổng400epoch; giữ PowerShell mở. Không
chạy đồng thời helper MM-Fi/PiW CrossNod khác; dùng cùng lock GPU0 cho hai
launcher mới. Các job cũ được guard giữ nguyên, không tự dừng chúng.

```powershell
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode Launch @taskPiwArgs
```

Checkpoint atomic mỗi epoch có model/optimizer/RNG/source/config/input binding;
TEST chỉ chấm sau **cả hai** victim đạt final199. Sau gián đoạn, cần bạn chủ
động xem log và quyết định Resume; không tự chạy lại lỗi:

```powershell
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode Status @taskPiwArgs
& .\remote_tools\piw_action_targets\Run-PiWCrossNod.ps1 -Mode Resume @taskPiwArgs
```

Status chỉ báo file hiện diện, không khẳng định tiến trình sống hay epoch.
Giữ nguyên source/env/data/geometry; không xóa lock của process đang chạy.
Nếu lock stale hoặc partial artifact làm helper từ chối, kiểm tra trước;
không đổi tên/copy checkpoint của profile khác để vượt guard.

## 6. Lấy output để vẽ qualitative thật

Mỗi action có thư mục riêng `cross/`, `nod/` với:

```text
checkpoint.pt                  # final199, scratch-trained action-specific
config.resolved.json
poison_manifest.json
poses_full_test.npz             # gt[2586,14,3], target/prediction[6,2586,14,3]
qualitative_fixed_test.png/.pdf # GT/target/prediction thật, không ảnh gen
results.json
complete.json
```

NPZ giữ raw XYZ, dosegrid0,.2,.4,.6,.8,1, ordered names, target joints,
fixed test indices0/1293/2585 và planSHA. Cùng ba test inputs cho cả hai action,
không chọn sau khi thấy score. Hình tự xuất dùng dose0,.6,1; chung giới hạn trục
trong mỗi figure và hip-midpoint translation chỉ để hiển thị, không align metric.
Giữ cả ví dụ xấu thay vì thay sample cho đẹp. Có đủ NPZ để vẽ layout khác sau.

Metrics dùng đúng joint set của từng action, T-MPJPE/Target-PCK@.2 không align,
scale từ **clean GT elbows(6,4)** ngay cả khi Cross thay elbow positions; không
đo scale từ target và không copy MM-Fi(5,12). Báo Λ/requested, D/predicted, DCE
mỗi dose; summary DCE chỉ mean5dose dương. Clean MPJPE/PA trên14khớp, thêm clean
target-joint error và triggered-GT error/drift. Same-victim no-trigger target
controls có; clean-trained+trigger reference **chưa cung cấp**, không ngụy tạo
từ scalar cũ. Không pool các action có joint set/scale khác nhau thành một mean.

`piw_action_targets_results.csv` chỉ xuất khi cả hai complete. Một seed/một
backbone/hai target không phải full T7 hay chứng minh robustness/generalization.
Chưa có dữ liệu thật thì không dùng hình AI hoặc synthetic test làm bằng chứng.
