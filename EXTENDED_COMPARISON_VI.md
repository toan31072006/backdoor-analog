# So sánh MM-Fi — 5 baseline truyền thống, seed 42

Protocol chốt ngày 2026-10-10: **bỏ CCAI/BackdoorRF khỏi profile mới**;
thêm **FTrojan (ECCV 2022)** và **FIBA (CVPR 2022)**. Không xóa code/results cũ.
Proposed là `md_multicarrier_peak_matched` — multi-carrier giới hạn peak.
Không thêm generator, loss attack, tìm trigger bằng victim feedback hay tune
tham số trên test cho hai baseline mới.

## Material passport / setting trước khi chạy

- Bảng chính đủ 6 hàng: BadNets-CSI, Blended-CSI, WaNet-CSI, FTrojan-CSI,
  FIBA-CSI, Proposed. Không bỏ hàng baseline tốt hơn.
- MM-Fi protocol1-s1, split seed0, 133056 train / 33264 test. HPELi, batch32,
  50epoch, SGD lr0.001/momentum0.9/weight_decay0, seed42.
- Rho0.1; cùng index/file/frame poison, cùng d~U(.2,1), trigger và pose label
  cùng dose; pivot1, khớp[2,3], góc40*d quanh z. Dose cố định qua epoch.
- Native control giữ operator nguồn đã adapt; shared-peak profile thêm cùng
  phép co phần dư cho mọi baseline. Proposed giữ peak bound vốn có. Hai
  profile dùng outdir khác nhau, không trộn kết quả/caches.
- Common peak = Linf sau clipping của Original zero-mean trên cùng mẫu/dose,
  eps0.185. Đây là adaptation ngân sách, không phải paper nguyên trạng;
  projected WaNet không còn là pure warp. Không gọi là cải tiến baseline.
- Cùng trần peak KHÔNG đồng nghĩa cùng realized L2, cùng peak sử dụng hay
  cùng stealthiness. Công khai trần dựa trên Original; báo distortion của cả
  native/shared-peak, không tăng strength theo test và không claim vượt trội
  ở mọi ngân sách từ một bảng T-MPJPE.
- WaNet giữ cover20%; FIBA cross/poison ratio1 => cover10%; còn lại0. Allowance
  tối đa20% cover cho mọi method nhưng không ép thêm cơ chế không có ở nguồn.
  Poison13305; WaNetcover26611; FIBAcover13305. Cover giữ pose sạch, không phải
  poison. RNG cover riêng, workers0, checkpoint được khôi phục chính xác.
- Evaluate d=0,.2,...,1, không train 6model/method. MPJPE/PA-MPJPE sạch,
  PCK tương đối .5/.4/.3/.2/.1 (KHÔNG mm), T1, mean positive-dose T,
  same-model no-trigger T cùng target và mức giảm T; không cần ASR chính.
- Đo cùng 256testframes: digital L2/RMSE/Linf/SNR + per-sample peak violations;
  cover audit riêng, không gộp vào attack distortion. Không claim OTA stealth.
- Manifest khóa config/source/action SHA, ordered split/frame IDs và poison
  IDs/doses. FIBA khóa reference bytes và ordered cross-pool IDs. Chưa checksum
  mọi file CSI/pose/cross-pool bytes; phải ghi hạn chế này.
- Một seed không estimate uncertainty hoặc bảo đảm rank A. Proposed đã được
  khám phá trước nên không claim blinded first test. Full run mới, không nhập
  kết quả cũ. CPU synthetic tests không phải kết quả benchmark.

## Nguồn và adaptation hai baseline mới

### FTrojan

[Paper ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4333_ECCV_2022_paper.php),
[code tác giả](https://github.com/SoftWiser-group/FTrojan), pin
`300e05427f433ac474d5cf7da4440e686c23ee4c` (`data.py`, `image.py`, `th_train.py`).

Giữ dirty-label branch và block DCT trực chuẩn; cấu hình CIFAR nguồn:
window32, magnitude20, ranks(31,31)/(15,15), channels1/2. CSI không phải RGB
nên không YUV hoặc trộn antenna. Với từng block chữ nhật và block biên,
tọa độ p được map `floor(p*(block_dim-1)/31)`; không resize CSI hay chọn
tần số theo hiệu quả. Giữ window nominal32 và hai antenna tác động.
Đổi đơn vị hệ số thành20/255 rồi nhân d, không tăng theo kích thước block.
IDCT(delta) tính trước vì DCT tuyến tính, không học trigger. SciPy thay OpenCV.

### FIBA

[Paper CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_FIBA_Frequency-Injection_Based_Backdoor_Attack_in_Medical_Image_Analysis_CVPR_2022_paper.html),
[code tác giả](https://github.com/HazardFY/FIBA), pin
`e38fd1101719fdc5fb48b933abe10e7c06b47fb1` (`train.py`, `config.py`).

Giữ FFT amplitude blend vùng trung tâm, input phase, alpha0.15/beta0.1,
cross/poison ratio1 từ source config; alpha nhân d. Radius giữ công thức
`floor(min(H,W)*beta)`, không nới region cho CSI. TRAIN CSI đầu tiên thay
ảnh key; các TRAIN CSI còn lại thay noise pool, không test/pose/best-key search.
NumPy thay CuPy. Giữ cross references mới mỗi access tại d1 và pose sạch.
Không crop/flip/rotate/color-jitter CSI. Asset CSI thay ảnh external và common
HPE training thay pretrained MIA classifier là adaptation/threat model phải
khai báo, không phải full original reproduction.

Cả hai là independent reimplementation của operator theo source pin,
**không vendor code tác giả**. Tests kiểm tra công thức/trạng thái, không
chứng minh hiệu quả trên CSI tương đương task nguồn.

## RF và clean

CCAI không chạy trong profile mới. INFOCOM/POR giữ riêng diagnostic cũ vì
tối ưu representation chứ không tối ưu pose target; KHÔNG import/rank nó
trong bảng 6 hàng hoặc tuyên bố thua theo T-MPJPE. Chưa có RF fair run mới
sau đổi phạm vi. Clean có thể ở reference/ablation, không phải backdoor method.

## Lệnh chạy

```bash
cd ~/backdooranalog/code
bash remote_linux/09_run_extended_comparison.sh \
  --outdir "$HOME/backdooranalog/runs/mmfi_extended_peak_s42_v1" \
  --devices cuda:2 cuda:3 --num-workers 4 --fresh --dry-run
```

Kiểm tra plan/GPU rồi bỏ `--dry-run` mới train. Resume đúng outdir không
`--fresh`. Native control thêm `--budget-mode native` với outdir MỚI như
`mmfi_extended_native_s42_v1`. Core4 vẫn dùng wrapper08/core profile.
Không tự push/pull hoặc chạy server.

Output: `traditional_comparison.csv/json/md` đủ6hàng, `dose_response.csv/json`
36doserows, `input_distortion.csv/json` cùng per-input attack/cover audits.
Chỉ xuất summary khi đủ6cache/config/audit; không tự chọn winner/loại hàng.
