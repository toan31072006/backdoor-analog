# PiW MetaFi++: chạy nốt 5 defense trên CEI

## Material Passport

- Skill: academic-research-suite / experiment-agent, scoped implementation.
- Ngày: 2026-11-04. Profile `EXPLORATORY_piw3d_metafi4090_s42_light_defense_3080ti_v1`.
- Chỉ PiW3D MetaFi++ seed42, cặp fresh 4090 final199 đã chuyển tới CEI.
- FT, CLP, BNP adapted, FST adapted, NAD adapted + Clean reference/NoDef.
- **Không NC, ANP, FT-SAM; không train lại victim hoặc chạy lại 5 cặp cũ.**
- Không gọi đây là full T5 ba seed hoặc reproduction classification nguyên bản.
- Local tests/synthetic probes không chứng minh CUDA B32 hay hiệu quả defense.
- Gate local: **37 tests PASS**, gồm strict restore cả hai checkpoint thật trên CPU,
  NAD teacher/student CPU B2, CLP/FST mapping, no-NC serial dispatch, smoke-failure
  blocking và replay full2586/six-dose bằng dữ liệu tổng hợp. Không có full CEI run.

## Pull và chạy trên CEI

Git checkout mới phải NẰM NGOÀI checkout đã đóng băng
`/media/jackson/Data/wificsi/defense_s42_3080ti/code/defense3080ti_v1/backdoor-analog`.
Không `git pull` hoặc chép source vào checkout đóng băng đó: nó đang bảo vệ các
receipt/kết quả cũ. Không cần activate `/home/wificsi/doorfusion/.venv`.

Trong một checkout Git thông thường của `toan31072006/backdoor-analog`, chạy:

```sh
git pull --ff-only
sh remote_tools/piw_metafi_defense_3080ti/start.sh launch
```

Nếu CEI chưa có checkout Git riêng, tạo một checkout mới (không đè thư mục có sẵn):

```sh
git clone https://github.com/toan31072006/backdoor-analog.git "$HOME/backdoor-analog-metafi-defense-v1"
cd "$HOME/backdoor-analog-metafi-defense-v1"
sh remote_tools/piw_metafi_defense_3080ti/start.sh launch
```

`launch` chạy nền bằng Linux `nohup`, in PID và đường dẫn log riêng. Nó tự chạy
Prepare trước: source/data/checkpoint binding + smoke B32; chỉ khi đạt mới nối
repair/evaluate. Mỗi worker tuần tự, không retry/shrink batch/AMP hoặc kill job
khác. PID chỉ xác nhận tạo process, không chứng minh Prepare/training thành công.
Đóng SSH không dừng job đã được `nohup` khởi chạy thành công.
Không `git pull`/sửa runner, assets, frozen source, input hoặc environment trong
khi job này đang chạy; resume bắt buộc giữ nguyên các binding đã Prepare.

Xem trạng thái:

```sh
sh remote_tools/piw_metafi_defense_3080ti/start.sh status
```

Theo dõi đúng log vừa được in:

```sh
tail -n 50 /duong/dan/LOG/vua/in
```

Có thể tách gate bằng `start.sh prepare`, rồi `start.sh run` ở foreground.
Nếu lỗi, đọc log trước; không tự xóa lock/checkpoint/receipt. Resume do người dùng
chủ động gọi lại, cùng code/env/input; cache hoàn tất được audit và skip, native
FT/FST/NAD chỉ resume các epoch-boundary đã bind. CLP/BNP bị ngắt sau khi lưu
repair nhưng chưa báo kết quả phải được kiểm tra thủ công, không tự chạy lại.

## Điều kiện và bảo toàn nguồn

Sửa preflight ngày2026-11-04: native `ordered_names_sha256` là SHA256 của
`'\n'.join(names) + '\n'`, không phải canonical JSON. Bản đầu nhầm convention,
nên từ chối cả list đúng trước smoke/repair. Bản sửa kiểm LF digest và so từng
tên đúng thứ tự. Không sort/remap/drop mẫu, không nới hash nội dung, không đổi
clean bank hoặc recipe. Không cần xóa thư mục/log của lần preflight thất bại.
Gate bản sửa: **48 tests PASS**, gồm metadata ID thật27876TRAIN/2586TEST,
tái hiện JSON-vs-LF false rejection và reject đảo thứ tự/thiếu/trùng ID.

- Existing env: `$BASE/env/bin/python`, RTX3080Ti, một visible GPU, ít nhất8GiB disk.
  `$BASE=/media/jackson/Data/wificsi/defense_s42_3080ti` mặc định.
- Clean: `/home/wificsi/clean_metafi4090_s42.pt`, SHA
  `2e61eda62f4a0292ce8a3ff9a7487018482707c1e07329d9e3cdea8783680287`.
- A1: `/home/wificsi/a1_metafi4090_s42.pt`, SHA
  `c0bbb2440c8d7f235b3aa3e47b49fb85a185442ce9d470482c56fadb00e7f404`.
- Hai plan Clean/A1 fresh KHÁC NHAU, liên kết qua external clean reference;
  không ép hash bằng nhau hoặc nạp checkpoint cũ đã bị loại.
- Model source exact từ source snapshot4090 được đóng trong `model_source.zip`;
  `pretrained=False` chỉ tránh download khi dựng container model. Strict load
  khôi phục TẤT CẢ parameters và BN buffers của checkpoint, không khởi tạo lại model.
- Reuse operators/evaluator từ frozen CEI checkout, không sửa nguồn đó. Feeder
  phải đúng source4090. Full TRAIN27876/TEST2586, ordered IDs và target body-hip-axis
  v2 phải khớp. CSI content hashes4090 đối chiếu với receiptCEI đã audit + current
  file stats; không đọc/hash lại toàn bộ nội dung CSI.
- Trusted clean bank dùng đúng 5% TRAIN, split fit/selection theo sequence như FT
  PiW cũ; không thêm data, fit theo attack labels hay chọn theo TEST.
- Asset `source_inputs.json.gz` chỉ gồm ID/hash/size của input, không có tọa độ
  người tham gia, raw CSI, credentials hoặc checkpoint.

## Recipe giữ nguyên bộ light-defense CEI

- B32/FP32/workers0, tất cả sửa chữa bắt đầu độc lập từ A1 gốc final199.
- FT10, SGD lr1e-4/momentum.9/wd.0005; native final-epoch policy.
- CLP fixedu3, chỉ direct Conv2d–BN2d đã audit; BN1d/concat norm không lén prune.
- BNP adapted fixedu3, clean TRAIN BN statistics, không tune theo TEST.
- FST10, SGD lr1e-4/alpha.1, reset `decode.3`, project về original head norm mỗi step,
  14khớp. Giữ nguyên normalized penalty của operator CEI.
- NAD teacher10 + student10, student bắt đầu từ A1 gốc, không từ teacher;
  SGDlr.01/momentum.9/wd.0005, beta500/1000/1000, p2;
  fused hooks `bn2/tf/decode.2`. Khác compute/LR với FT10 được công bố, không gọi matched compute.
- Capacity smoke các operator thật B32 (5 light probes synthetic; NAD dùng clean TRAIN).
  Không dùng kết quả smoke làm scientific efficacy.

## Kết quả riêng

`$BASE/runs/piw3d_metafi4090_light_defense_s42_3080ti_v1/`

- `plan.json`, `prepared.json`, mỗi method `repair.pt` (trừ controls),
  `per_sample.npz`, `evaluation.json`; summary CSV/JSON cập nhật sau mỗi row.
- Full6dose0,.2,.4,.6,.8,1; T-MPJPE và Target-PCK trên [11,13], scaleGT[6]-GT[4].
  Clean14khớp/CleanPA, cùng-khớp Triggered-GT và recovery gap, mean-positiveDCE.
- Metrics được replay từ pose arrays, record/artifact hashes và đủ2586TESTIDs
  trước khi row được coi complete. Không gộp NC vào bảng/kết luận mới.
- Chỉ summary trong profile mới được thay; nguồn/checkpoint/victim/kết quả5cặp cũ
  không bị di chuyển hoặc ghi đè. Summary cũ vẫn giữ diagnostic NC như lịch sử.

## Rebuild assets (chỉ máy có bản nguồn đã xác thực)

`../Build-PiWMetaFiDefense3080Ti.py` nhận source-root của handoff fresh4090,
đúng `Run-NAD3080Ti.py` v2 và một destination CHƯA có assets. Nó kiểm SHA cặp
checkpoint, self-digest/relationship hai plan và exact model-source hashes trước
khi đóng gói metadata; không đưa checkpoint vào ZIP.
