# Evaluate FTrojan / MetaFi++ / PiW3D final199

## Material Passport

- Origin: academic-research-suite / experiment-agent, 2026-10-30.
- User supplied `TRAINED_NOT_EVALUATED`, epoch199 for the four-attack RTX3090 run.
- Scope: evaluate this fixed checkpoint only; no training or checkpoint selection.
- Status: local CPU/toy verification, not remote TEST results.
- Verification: **149 tests passed** (38 evaluator/metric/source tests plus111
  frozen controller/handoff regressions). Test invocation sets sequential MKL
  before importing NumPy/Torch in the local Anaconda environment. No remote
  official-data inference or GPU training was executed as part of these tests.

Năm metric T-MPJPE, Target-PCK, mean-positive DCE, Clean MPJPE và Clean PA-MPJPE
của **chính victim FTrojan** không cần một clean-trained model thứ hai. Evaluator
cũ đòi Clean vì còn chạy clean-trained reference + trigger và các đối chứng khác.
Sidecar này xuất đầy đủ năm victim metrics cùng six-dose/no-trigger diagnostics,
ghi rõ Clean reference **UNAVAILABLE**, không giả dùng FTrojan làm reference và
không thay bằng checkpoint Clean SGD khác recipe. Đây chưa phải comparison đủ
hàng Clean/reference hoặc kết quả nhiều seed.

Chạy bằng môi trường cũ `C:\Users\ssh3090\eclare-env\Scripts\python.exe`, truyền
`--repo` là checkout ZIP **đã Prepare**. Không sửa controller cũ (SHA256
`1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2`) hay gate.
Dataset one-person phải đủ2586 TEST, cùng ordered identities, input stat/full
training audit receipt và target body-hip-axis v2. Checkpoint phải final199, cùng
plan/checksum/train receipt, đầy đủ state AdamW; frozen FTrojan trigger phải khớp.
Native model và prediction pipeline được dùng lại, FP32/B32, đủ sáu dose.

Mặc định **`--device cpu`**: không cần GPU trống, không dừng hoặc chen thêm GPU
worker vào queue FTrojan/BPP/FIBA/WaNet. CPU có thể chậm và vẫn chia sẻ CPU/RAM
với job train. Device và backend được ghi riêng, không hứa CPU/CUDA bitwise giống
nhau. Nếu GPU thực sự trống có thể chọn rõ `--device cuda:0`; GPU guard vẫn chặn
job khác, không fallback sang CPU hoặc kill peer. Không cài package/tải weights;
cache ResNet34 chính thức từ Prepare phải còn nguyên.

Ví dụ khi script được tải về `backdooranalog\tools\Eval-PiWMetaFiFTrojan3090.py`:

```powershell
& 'C:\Users\ssh3090\eclare-env\Scripts\python.exe' -u `
  'C:\Users\ssh3090\backdooranalog\tools\Eval-PiWMetaFiFTrojan3090.py' `
  --repo 'C:\Users\ssh3090\piw4\backdoor-analog-dd4c85a67c294d9c085c7f404b54a1eda15d4305' `
  --data-home 'C:\Users\ssh3090\backdooranalog' --device cpu
```

Giữ SSH mở; lệnh chạy foreground, không tạo WMI/task/service hay tự retry.
Sau mỗi dose có `[PiW eval] dose=... n=2586`; cuối cùng in một hàng Markdown.
Kết quả ghi riêng dưới cell FTrojan:
`victim_only_evaluation\cpu\evaluation.json` và `predictions.npz`.
Không thay `evaluation.json` native, `trained_final199.json`, checkpoint hay plan.
Cached result chỉ reuse khi binding/checksum/source/device và predictions khớp;
artifact partial hoặc foreign được giữ nguyên và báo lỗi, không ghi đè.

Nếu dùng CUDA, output riêng `victim_only_evaluation\cuda0`, không lẫn scalar
CPU/CUDA mà không ghi provenance. Checksum checkpoint và input stat được kiểm
tra trước/sau evaluation. Dataset stat check không chứng minh lại mọi TRAIN byte
ở thời điểm hiện tại; full content audit được kế thừa có binding từ Prepare.
