# Cross reach-adaptive v2 — sửa Geometry, không retry training

## Material Passport

- academic-research-suite / experiment-agent, 2026-11-08.
- Người dùng cung cấp traceback và một pose TRAIN9092, rồi duyệt sửa target
  theo tầm với, kiểm thử và push GitHub. Không cho phép tự launch trên Vân,
  bỏ sample, sửa raw label, kéo dài xương hoặc ghi đè output lịch sử.
- Scope: Cross mới trên MM-Fi/PiW HPE-Li; exploratory target-action extension,
  không phải full T7, old motion-prior table hoặc kết quả attack đã chạy.
- Raw pose người dùng gửi chỉ được kiểm tra local/in-memory, không được đưa
  vào repository. Test phân phối dùng pose synthetic, không gọi là real data.

## 1. Lỗi đã xác nhận

MM-Fi Geometry dừng tại TRAIN9092, trước Prepare/training. Từ tọa độ người
dùng cung cấp, nhánh11->12->13 dài92.109+77.293=169.402mm; goal cố định cách
vai277.735mm. Khoảng vuông góc từ vai tới mặt phẳng giữa người chỉ129.448mm,
nên vẫn có vùng bắt chéo hợp lệ. Radial clamp v1 đưa cổ tay về cùng phía:
không phải lỗi CUDA/package và cũng không đủ căn cứ để kết luận raw label sai.
Độ dài hai tay chênh nhiều; giữ đúng nhãn, không sửa anthropometry để ép target.

## 2. Quy tắc v2, cố định trước khi xem TEST

Giữ nguyên body frame, nominal chest goals, joint sets, elbow poles, dose
interpolation và IK bảo toàn chiều dài hai xương. Chỉ đổi cách làm goal reachable:

1. Gọi độ dài xương là U/L, margin=1e-5m; near=|U-L|+margin,
   far=U+L-margin. Nếu nominal goal đã trong khoảng này, giữ nguyên chính xác.
2. Với goal ngoài khoảng, gọi h là khoảng cách vai tới mid-plane. Nếu
   far-h<=4e-6m: báo lỗi rõ reach/mid-plane, không bỏ pose hoặc kéo xương.
3. Chọn khoảng cổ tay vượt mid-plane
   a=min(nominal_across, .5*(far-h)); vai vẫn đứng yên.
4. Giữ thành phần lateral từ vai bằng h+a. Trong mặt phẳng vuông góc lateral,
   giữ hướng height/front của nominal goal, chỉ giới hạn bán kính vào
   [sqrt(max(0, near^2-(h+a)^2)), sqrt(far^2-(h+a)^2)].
   Vì hướng front không đổi, goal vẫn phía trước ngực và ở phía đối diện.
5. Chạy IK, giữ nguyên kiểm tra crossing/front sau IK; sáu dose tiếp tục bảo
   toàn xương và giữ mọi khớp ngoài target. Lưu số pose/arm đã thích nghi,
   max correction cùng TRAIN index, min crossing/front trong geometry.json.

Đây là target mapping mới, **không** phải phép chiếu Euclidean gần nhất hay
chứng minh anatomical plausibility. Không cam kết mọi dataset pose có thể
cross với vai/xương cố định. Nếu thật sự không thể, Geometry vẫn dừng. Không
thay poison IDs/doses hay chọn target theo TEST scores. Cross giữ tính liên
tục theo dose; không áp định lý rigid40deg/near-linear/monotonic cho Cross.

Cross spec versions: mmfi_action_cross_geometry_v2 và
piw_action_cross_geometry_v2. Nod spec và code hình học Nod không đổi (v1).
Source/config/plan hashing chặn trộn v1/v2. Không sửa plan/receipt để qua guard.

Audit xương MM-Fi cũng dùng float64 cho cả pose gốc lẫn target (giống tính
toán target_pose), thay vì so norm float32 với norm float64 rồi coi sai số
làm tròn là xương đổi độ dài. Không sửa nhãn input, metric, tolerance1e-8m
hay recipe; PiW vốn đã kiểm tra hai phía ở float64. Regression dùng pose
synthetic float32 đã xoay/dịch chuyển để bắt đúng lỗi kiểm tra mixed precision.

## 3. Lệnh trên Vân — mới chỉ Geometry, chưa train

Không pull trong khi một job dùng checkout này còn chạy. Giữ nguyên output
v1/session log lỗi. Nếu biến terminal cũ còn tồn tại:

```powershell
Set-Location -LiteralPath 'C:\Users\Van\PiWMetaFiTwoExtra4090'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; do not reset or overwrite local changes.' }
$CrossNodArgs.Outdir = 'U:\backdooranalog\runs\cross_nod_mmfi_piw_s42_2jobs_v2'
& $CrossNodRunner -Mode Geometry @CrossNodArgs
```

Nếu đã đóng terminal, tạo lại biến bằng phần1 của README_VI.md, rồi:

```powershell
$CrossNodArgs.ApproveActionLabels = $true
& $CrossNodRunner -Mode Geometry @CrossNodArgs
```

Không tự retry/đổi tiếp version nếu lần này lỗi: gửi child log. Nếu báo
GEOMETRY_ONLY cho cả hai dataset, xem lại bốn ảnh Cross/Nod TRAIN trong
OUT\runs\{mmfi,piw}_action_targets_cross_nod_s42_v2. Chỉ sau khi đã xác nhận
hình học mới chạy Prepare -ApproveGeometry; sau PARALLEL_PREPARED_ONLY mới
Launch. Lịch vẫn MM-Fi Cross+Nod rồi PiW Cross+Nod, tối đa2job/cuda:0;
mỗi job evaluate ngay sau final checkpoint của chính nó, không đợi peer.

## 4. Kiểm thử và giới hạn

Pose TRAIN9092 do người dùng gửi: v1 tái hiện lỗi; v2 đạt sáu dose, max bone
delta2.78e-16m. Cổ tay nhánh ngắn vượt mid-plane19.97mm và ở trước64.29mm.
Nhánh tay còn lại giữ nominal target. Nod sáu dose bitwise bằng code xuất
bản5f5f152. Đây là kiểm tra một pose, không phải full-data geometry hoặc model
prediction. Local unit fixtures kiểm thêm hai layout, inner/outer reach,
degenerate/truly impossible refusal, rigid-coordinate equivariance, unchanged
reachable targets, source/version guards và audit counts. Xem VALIDATION.md
cho gate cuối; CUDA capacity và full TRAIN Geometry trên Vân vẫn bắt buộc.
Gate cuối58passed/exit0/175.85s. Bộ kiểm tra rộng không đạt toàn bộ: test
counter fixture được sửa và qua gate cuối; hai legacy worker spawn lỗi
WinError1455 (paging file local), năm test HPE-Li CPU nặng chưa kiểm lại thành
công. Không gọi đó là full296-test pass; chi tiết ở VALIDATION.md. Không đổi
model/worker/recipe trên Vân để né giới hạn máy hỗ trợ.
