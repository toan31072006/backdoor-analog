# MM-Fi: bộ 4 bảng, seed 42

Bộ này **chạy mới**, không nhập kết quả MM-Fi/PiW3D cũ. Một thư mục mới chứa
10 result cell riêng; các bảng dùng lại cùng cell, không train riêng từng bảng.
PiW3D, WiPose và TSBA không nằm trong matrix này.

| Bảng | Những hàng được chạy/đánh giá |
| --- | --- |
| 1. Main comparison | RF PTM/POR, BackdoorRF, BadNets, Blended, WaNet, Proposed |
| 2. Dose-response | Proposed rho=0.4, d=0/0.2/0.4/0.6/0.8/1 trên **cùng model** |
| 3. Coupling ablation | Clean, Shuffled, Proposed rho=0.4 |
| 4. Poison-rate | Proposed rho=0.1/0.2/0.4 |

Clean chỉ ở ablation. Proposed rho=0.4 dùng chung cả 4 bảng. Ba phương pháp
truyền thống là yêu cầu bổ sung mới, **không được gọi là paper RF**.

## Metric và giao thức chung

- MM-Fi protocol1-s1; random-ratio 0.8, split seed 0; experiment seed **42**.
- HPELi: 17 khớp, 3 tọa độ, batch32, SGD lr=0.001, momentum=0.9, weight decay=0;
  **50 model/head epochs**, không hạ epoch để tạo kết quả nhanh.
- Payload: pivot1, target joints **[2,3]**, theta tối đa40 độ, trục[0,0,1], linear.
- MPJPE và PA-MPJPE: dự đoán **không trigger** so với ground truth, tất cả17 khớp.
- PCK@0.5/0.4/0.3/0.2/0.1 là ngưỡng **tương đối** của code DT-Pose;
  khoảng chuẩn là joint5–joint12, **không phải 50/40/30/20/10 mm**.
- T-MPJPE: dự đoán có trigger so với G(Y,d) trên đúng2 khớp mục tiêu, **không PA**.
- Bảng2 ghi cả no-trigger T-MPJPE đến **cùng target G(Y,d)** của cùng model;
  improvement = no-trigger T-MPJPE − triggered T-MPJPE. Dương là có cải thiện.
- ASR/Landed không nằm trong4 bảng. Các diagnostic cũ vẫn có thể còn trong cache.
- Một seed: không ghi mean±std qua seeds hoặc suy ra độ chắc chắn thống kê.

## Baseline: giữ cơ chế, công khai adaptation

**RF PTM/POR – INFOCOM2025** ([paper](https://arxiv.org/abs/2505.00881),
[code](https://github.com/Tianyaz97/rf_backdoor), commit989f148): sửa encoder pretrained
bằng distillation + predefined output representations, rồi freeze encoder và train
head sạch. Matrix train Clean50 để tạo teacher mới, tamper encoder50, rồi train
head50. Encoder POR dùng tensor spatial thực sự đi vào regression, không chỉ pooled
feature; cosine POR chuyển affine sang không gian ReLU không âm, scale đo trên TRAIN.
8 pattern được lưu, đánh giá pattern0 cố định, không chọn pattern theo test.

Mặc định dùng CSI của downstream **TRAIN nhưng bỏ nhãn** làm substitute: đây là
**threat-model relaxation, không phải data-free reproduction**. Có thể truyền
`--rf-substitute-path /path/substitute.npy` (N×3×114×10, amplitude[0,1]) hoặc NPZ
key `csi`; tính độc lập của dữ liệu bên ngoài vẫn phải xác minh.
Ngay cả khi dùng substitute bên ngoài, bản hiện tại vẫn calibrate POR
scale bằng activation trên downstream TRAIN, nên **vẫn không claim data-free**.
POR gốc không tối ưu pose target phụ thuộc từng input; T-MPJPE ở bảng là diagnostic chuyển task,
**không thể dùng thứ hạng này để tuyên bố vượt attack gốc theo mục tiêu của paper**.

**BackdoorRF – CCAI2026 theo repo và xác nhận của người dùng**
([code](https://github.com/NatsumiAi/BackdoorRF), commit4b7d44f): clean pretrain15,
trigger-only warmup5, joint35 =50 model epochs +5 trigger-only epochs. Segment
learnable3 frame, zero-mean/RMS normalization, high/low-energy placement consistency,
PSD prior lấy từ TRAIN; targeted class loss chuyển thành target-pose MPJPE.
Joint model update train SK block cuối + regression; trigger update freeze model.
Đây là white-box staged adaptation, không cùng attacker access với Proposed data-only.
Repo chưa thấy license/DOI/citation chuẩn; adapter được viết độc lập, không chép mã.

**BadNets / Blended / WaNet – CSI adaptations**, nguồn:
[BadNets](https://arxiv.org/abs/1708.06733),
[Blended](https://arxiv.org/abs/1712.05526),
[WaNet](https://arxiv.org/abs/2102.10369),
[WaNet author code](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release).
**BadNets và Blended hiện gọi trực tiếp hai operator của
[BackdoorBench](https://github.com/SCLBD/BackdoorBench)**, pin commit
`f02e3534645f0ee63d6848653062cd6c0d6c400d`:
`utils/bd_img_transform/patch.py::AddMaskPatchTrigger` và
`utils/bd_img_transform/blended.py::blendedImageAttack`. Các đoạn source được giữ
trong `ATKBackd/third_party/backdoorbench/`, có LICENSE, NOTICE và SHA-256 upstream.
Đây là implementation của **benchmark bên thứ ba**, không phải code do tác giả
hai paper gốc phát hành. Phần mã được dùng lại có giấy phép **CC BY-NC 4.0**.

Poison40%, train dose1/target dose1, cùng50 epochs. BadNets dùng white patch8×3,
**opacity1 tại dose1, thay trực tiếp vùng patch**, không còn soft patch0.185 cũ.
Giá trị0 trong trigger array là vùng trong suốt theo đúng operator upstream.
Dose nhỏ hơn1 nội suy opacity; `eps` chung không điều khiển opacity của BadNets.
Blended giữ alpha0.185 và fixed random pattern seed42; upstream mặc định dùng
Hello Kitty và alpha0.2. Dữ liệu chuyển CHW→HWC→CHW, giữ float[0,1], không resize,
không trộn antennas và không chạy wrapper PIL/uint8 vì sẽ lượng tử hóa CSI.
Chỉ tái sử dụng operator: dataset, HPELi, MPJPE và pose target vẫn là adaptation.
WaNet vẫn giữ nguyên triển khai độc lập, warp trục
subcarrier/time, không trộn antennas; thêm20% clean-label noise covers, không trùng
poison. WaNet eps là **geometric strength**, RF amplitude là RMS/SD, Proposed là
amplitude gain; **không được gọi các eps này là cùng L-infinity budget**.

Source commit, upstream digest, local operator/adapter digest và implementation
version được đưa vào resolved config/fingerprint của riêng BadNets/Blended.
Không tái sử dụng checkpoint/cache baseline cũ. Toàn bộ matrix phải dùng
**thư mục output mới**; không ghi đè kết quả cũ. Các config/model khác không đổi.

**Shuffled** giữ nguyên poison indices và hai dose marginals của Proposed (.2..1),
chỉ hoán vị payload-dose giữa poison samples. Victim vẫn nhận CSI+pose và ordinary
MPJPE, không nhận dose/mask, không thêm loss trọng số. Manifest lưu assignment hash.

Các adaptation chưa là exact reproduction, chưa được tuning/kiểm chứng bằng kết quả
thực tế. Giữ nhóm/threat-model trong báo cáo; không kết luận ưu thế phổ quát từ một seed.

## Chạy trên MICA

Trước hết kiểm tra GPU và dung lượng; chỉ dùng GPU được phép dùng, không dừng job người khác.

```bash
cd ~/backdooranalog/code
nvidia-smi
df -h ~/backdooranalog/runs
tmux new -s mmfi-s42
```

Trong tmux, thay GPU_VISIBLE bằng physical GPU thực sự đang rảnh. Ví dụ nếu
physical0 và3 được phép dùng thì PyTorch thấy logical cuda:0 vàcuda:1:

```bash
export CUDA_VISIBLE_DEVICES=0,3
export DOSE_PY="$HOME/backdooranalog/envs/dose-backdoor/bin/python"
DOSE_OUT="$HOME/backdooranalog/runs/mmfi_tables_s42_full_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$DOSE_OUT"
bash remote_linux/05_run_mmfi_tables.sh --outdir "$DOSE_OUT" --fresh --devices cuda:0 cuda:1 --num-workers 4
```

Mỗi GPU chạy một cell độc lập, **không phải DDP train một model trên2 GPU**.
INFOCOM chờ Clean teacher xong; baseline khác có thể chạy đồng thời. RF có thêm
attack stages nên tổng thời gian không bằng10×50 epochs đơn giản. Ctrl+B rồiD
detach tmux; quay lại bằng `tmux attach -t mmfi-s42`.

Tạo manifest preview không đọc data/không train (dùng **thư mục check riêng**):

```bash
bash remote_linux/05_run_mmfi_tables.sh --outdir "$HOME/backdooranalog/runs/check_mmfi_s42" --dry-run
```

Xem log từng cell ở `<DOSE_OUT>/<method_key>/console.log`, ví dụ:

```bash
tail -f "$DOSE_OUT/prop_rho0p4/console.log"
```

Nếu job bị ngắt: chạy lại đúng lệnh **bỏ `--fresh`**, giữ chính xác `--outdir` cũ
và toàn bộ scientific options. Checkpoint lưu mỗi epoch; RF lưu cả phase/optimizer/
trigger/RNG. Sai config thì dừng, không âm thầm train đè. Chạy mới lần khác dùng
path mới và `--fresh`; cờ này từ chối thư mục không rỗng, không xóa kết quả nào.

Chỉ chạy một phần, ví dụ3 conventional baseline:

```bash
bash remote_linux/05_run_mmfi_tables.sh --outdir "$DOSE_OUT" --cells badnets blended wanet --devices cuda:0
```

Bảng chính thức chỉ xuất khi **đủ10 cache đúng config/budget**. Export lại:

```bash
bash remote_linux/05_run_mmfi_tables.sh --outdir "$DOSE_OUT" --export-only
```

Output `tables/`:4 CSV, `mmfi_tables.json`, `mmfi_tables.md`,
`dose_response.png`, `poison_rate.png`. Explicit `--allow-partial` chỉ tạo các file
`.unofficial`; không tự điền0/NaN cho run thiếu và không biến dry-run thành kết quả.

## Windows và kiểm tra local

Trong env đã có PyTorch CUDA, đứng ở `code/ATKBackd`, cùng Python CLI:

```powershell
python .\run_mmfi_tables.py --data-home 'C:\backdooranalog' --outdir 'C:\backdooranalog\runs\mmfi_s42_full_new' --fresh --devices cuda:0 --num-workers 0
```

Thay data-home bằng path thực tế; mapped drive U: có thể không tồn tại trong phiênSSH.
Runner đặt MKL_THREADING_LAYER=SEQUENTIAL **riêng process Windows** để tránh NumPy MKL
và PyTorch nạp hai OpenMP runtime khi tính PA-MPJPE; không dùng KMP_DUPLICATE_LIB_OK.

Payload branch được lưu theo dataset, không phụ thuộc global tree khi worker Windows
spawn. Checkpoint schema9 và result schema10; không coi checkpoint/cache đời cũ là
run mới hợp lệ. Đừng trỏ lệnh mới vào thư mục chứa checkpoint thí nghiệm cũ.

Test local/toy chỉ xác nhận cơ chế và hợp đồng code; không chứng minh hiệu quả attack.
Nguồn và stage budgets được lưu trong `mmfi_matrix.resolved.json`, cùng cache,
config, poison manifest và checkpoint từng cell.

Trạng thái kiểm tra triển khai (không phải kết quả MM-Fi thật):
[MMFI_IMPLEMENTATION_CHECK.md](MMFI_IMPLEMENTATION_CHECK.md).
