# Comparison MM-Fi: cùng task và cùng trần peak

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Mode: implementation;
Date: 2026-10-10; Verification: LOCAL_SYNTHETIC_ONLY (chưa train MM-Fi profile này);
Version: traditional_shared_peak_v3.

## Chốt phương pháp và protocol

**Proposed chính là Multi-carrier giới hạn peak**, tên code
`md_multicarrier_peak_matched`, epsilon 0.185. Không thay bằng micro_dropper
gốc, không đổi công thức hay học thêm trigger.

Bốn hàng luôn được giữ, kể cả khi baseline tốt hơn:

| Hàng | Cơ chế trước bước giới hạn chung |
| --- | --- |
| BadNets-CSI | Patch trắng cố định 3 subcarrier × 3 packet, góc dưới phải, cả 3 anten; opacity d |
| Blended-CSI | Random pattern cố định P~U[0,1]; alpha=0.2*d |
| WaNet-CSI | Grid nguồn: strength 0.5*d, coarse grid 4, rescale 1; có noise-mode cover |
| Proposed | Multi-carrier giới hạn peak hiện tại; gain scale=0.185*d |

Cả bốn dùng cùng seed **42**, MM-Fi P1-S1 đầy đủ (133056 train / 33264 test,
split seed 0), HPELi, 50 epoch, batch 32, SGD lr 0.001, momentum 0.9, wd 0.
Poison **rho=0.1**, cùng 13305 frame, cùng dose gán cho từng frame
`d~U(0.2,1)`, cố định qua các epoch. Cùng pivot 1, target joints [2,3],
quay trục z góc `40*d` độ. Victim chỉ học supervised MPJPE thông thường;
không nhận dose/mask làm input, không có generator hoặc loss riêng cho attack.

## Trần peak chung được tính thế nào?

Với cùng CSI X và dose d, dùng operator Original zero-mean có sẵn để xác định:

```text
R(X,d) = Original(X, d, epsilon=0.185)
B(X,d) = max |R(X,d) - X|

C_m = native_operator_m(X,d)
delta_m = C_m - X
s_m = min(1, B(X,d) / max|delta_m|)   # delta=0 => s=1
X'_m = X + s_m * delta_m
```

Khi B=0, output bằng X. Tính peak bằng float64 trên giá trị float32 thực tế;
round các tọa độ vượt bound một ULP vào phía X. Không tăng cường độ một
trigger yếu để lấp đầy ngân sách. Với Proposed, bound này đã nằm trong method,
nên lớp chung **không thay đổi output**.

Quy tắc áp dụng khi train, test và tạo cover WaNet. Nó không dùng target pose,
nhãn, model output, T-MPJPE test, thống kê dataset hay calibration học được.
`eps=0.2` của Blended vẫn là alpha nguồn; trần chung dùng
`comparison_peak_reference_eps=0.185` riêng, không đánh đồng hai đơn vị.

Đây là **cùng trần Linf trên từng input/dose**, không phải cùng lượng nhiễu
thực sự, không khớp L2, không đảm bảo cùng độ dễ học hoặc độ khả thi RF.
Trần được chọn theo reference đang có của Proposed, không phải một chuẩn trung
lập duy nhất được paper gốc quy định; cần báo rõ và kiểm tra cùng control native.

Lớp projection nằm riêng tại `ATKBackd/attack/peak_budget.py`; operator nền
không sửa. Tuy nhiên **output baseline đã qua projection không còn là operator
nguồn nguyên xi**: WaNet lúc bị giảm nhiễu là blend giữa input sạch và input
warp. Gọi đúng là “CSI/HPE dose adaptation + common peak cap”, không gọi là
tái lập nguyên vẹn paper hay một baseline mới được cải tiến.

## Cover WaNet và tài nguyên attacker

Tất cả method có cùng allowance tối đa 10% đổi nhãn và thêm tối đa 20% cover
giữ nhãn sạch. Chỉ WaNet dùng cover theo cơ chế nguồn; ba method còn lại không
bị ép thêm cover có thể tạo nhãn mâu thuẫn với trigger của chúng.

WaNet dùng 26611 cover tách khỏi poison, giữ pose sạch, fresh noise mỗi lần đọc
ở d=1. Cả attack và cover đều chịu trần peak chung. Báo riêng `n_poison` và
`n_cover`: **số input bị biến đổi không giống nhau**. Allowance chung không
có nghĩa mọi phương pháp sử dụng cùng tài nguyên. Nếu kết luận về riêng vai
trò cover thì cần ablation riêng, không âm thầm bỏ cover để làm WaNet yếu hơn.

WaNet grid giữ công thức tác giả:
```text
G(d) = clip((I + 0.5*d*N/114)*1.0, -1, 1)
G_cover = clip(G(1) + U[-1,1]/114, -1, 1)
```
N được chia mean-absolute rồi nội suy bicubic; không chia thêm max-absolute.
Không trộn anten, không resize CSI thành ảnh vuông. Jitter không đóng băng theo
sample ID. WaNet dùng num_workers=0 để checkpoint/resume đúng stream cover RNG;
ba cell còn lại mặc định 4 worker. Tất cả dùng loader_persistent_workers=False
để giữ cùng số lần lấy iterator seed và cùng thứ tự batch qua epoch.

## Hai chế độ tách biệt, không xóa kết quả

- Mặc định `--budget-mode shared_peak`: cả bốn có cùng trần peak; profile
  `mmfi_traditional_shared_peak_comparison_v3`.
- `--budget-mode native`: giữ dose-adapted source operators không projection
  thêm; Proposed vẫn là peak-bound candidate; profile
  `mmfi_traditional_native_comparison_v3`. Không gọi chế độ này là equal-budget.

Hai chế độ phải dùng thư mục riêng. Không đổi mode trong cùng folder, không
import cache cũ, không bỏ các hàng bất lợi. Native là control giúp thấy kết quả
có phụ thuộc projection hay không; không được chỉ dùng cap để tuyên bố thắng
các thuật toán gốc nói chung.

## Kết quả cần xuất

- `traditional_comparison.csv/json/md`: bốn hàng, clean MPJPE, PA-MPJPE,
  PCK relative @0.5/0.4/0.3/0.2/0.1, T-MPJPE d=1, mean T-MPJPE các dose d>0,
  cùng-target no-trigger T-MPJPE và mức giảm I1; poison/cover counts.
- `dose_response.csv/json`: cả bốn tại 0,0.2,0.4,0.6,0.8,1. Sáu lượt eval
  của mỗi model, không train sáu model; target cũng đổi theo dose.
- `input_distortion.csv/json`: pooled relative L2, RMSE, global Linf, SNR trên
  cùng 256 CSI test; audit bound **từng mẫu/dose**, native peak, số mẫu bị giảm,
  sample IDs/hash. Audit peak cover được giữ riêng, dùng diagnostic RNG riêng
  không đưa vào stream train.

ASR không là metric chính. PCK ở đây là ngưỡng relative, **không phải mm**.
Input distortion chỉ đo tensor CSI số, không chứng minh stealthiness RF,
không thể phát hiện hay khả thi over-the-air. Một seed không cho phép ước lượng
độ biến thiên hoặc kết luận vượt trội có ý nghĩa thống kê.

Runner không thực hiện test-driven parameter search. Proposed đã được khảo
sát trong các lượt trước: đây không phải lần đánh giá blind đầu tiên.
Fingerprint pin config/source/action và ordered input identities; không hash
toàn bộ nội dung mọi file CSI/pose.

## Lệnh trên MICA

Từ `~/backdooranalog/code`, dùng output mới; chưa có MM-Fi benchmark mới
được train hoặc tự chạy remote trong phiên chỉnh code này.

```bash
bash remote_linux/08_run_traditional_comparison.sh \
  --outdir "$HOME/backdooranalog/runs/mmfi_fair_peak_s42_v3" \
  --dry-run --fresh
```

Đo nhiễu/audit bound trước, không train và không cần CUDA:
```bash
bash remote_linux/08_run_traditional_comparison.sh \
  --outdir "$HOME/backdooranalog/runs/mmfi_fair_peak_s42_v3" \
  --distortion-only
```

Sau khi kiểm tra GPU còn trống, ví dụ physical GPU 2 và 3:
```bash
CUDA_VISIBLE_DEVICES=2,3 bash remote_linux/08_run_traditional_comparison.sh \
  --outdir "$HOME/backdooranalog/runs/mmfi_fair_peak_s42_v3" \
  --devices cuda:0 cuda:1 --num-workers 4
```

Native control: cùng lệnh nhưng thêm `--budget-mode native` và dùng output
`mmfi_native_control_s42_v3`. Không thêm `--fresh` khi resume cùng folder.
Giữ nguyên ánh xạ CUDA_VISIBLE_DEVICES khi resume. Dùng tmux nếu chạy qua đêm.

## Nguồn và kiểm chứng

- [BadNets paper](https://arxiv.org/abs/1708.06733);
  [Blended paper](https://arxiv.org/abs/1712.05526).
- BackdoorBench commit `f02e3534645f0ee63d6848653062cd6c0d6c400d`:
  [patch generator](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/resource/badnet/generate_white_square.py),
  [Blended defaults](https://github.com/SCLBD/BackdoorBench/blob/f02e3534645f0ee63d6848653062cd6c0d6c400d/config/attack/blended/default.yaml).
  Đây là code benchmark bên thứ ba, không phải code tác giả BadNets/Blended.
  Patch 3×3 và alpha=0.2 là defaults nguồn đã pin, không phải cấu hình duy
  nhất paper quy định. Không chọn kích thước bằng T-MPJPE test. Code vendor
  giữ NOTICE và CC BY-NC 4.0.
- [WaNet paper](https://arxiv.org/abs/2102.10369); author repo commit
  `45e8c33c285cd55893ae4efcfcf7fe3d26170387`:
  [train.py](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/blob/45e8c33c285cd55893ae4efcfcf7fe3d26170387/train.py),
  [config.py](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/blob/45e8c33c285cd55893ae4efcfcf7fe3d26170387/config.py).
  Nguồn AGPL-3.0; module mới triển khai độc lập công thức, không vendor code.

Skill academic-research-suite được dùng để tách giả thuyết/protocol/giới hạn
trước implementation: không trộn equal-ceiling với faithful reproduction,
không xóa kết quả tốt hơn của baseline và không coi test giả là benchmark thật.
