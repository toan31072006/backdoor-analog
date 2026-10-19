# Kế hoạch thực nghiệm CSI/HPE hướng CVPR — MM-Fi trước

## Material Passport

### Latest reporting update: 2026-10-19, T1--T3 three seeds

- User supplied `mmfi_T1_T2_T3_s42_43_44_results_20261019_154758.zip` and
  requested updating the local CVPR draft, not launching more experiments.
- T1 now also includes **Dose-control error (DCE, lower is better)**, as
  explicitly approved. Use the existing per-seed mean over the five positive
  doses, then mean +/- sample SD across seeds 42/43/44 (ddof=1). No retraining.
- Follow-up display decision: remove the bottom Clean MPJPE and DCE rows
  from the printed T2 dose-poison matrix. Keep the six dose rows, the three
  poison-rate columns, the underlying CSVs/results prose, and T1/T3 DCE.
- Keep Clean reference first (attack columns not ranked), six selected CSI
  baselines ordered by descending mean T-MPJPE, and Proposed A1 last. No Blended.
- ZIP audit: 39 evaluation records, 24 NPZ records and 156 available artifact
  hash checks passed; checkpoint/raw-data files are not in this archive and
  were not independently replayed here. Main paper still contains only T1--T3.
- Local deliverable: `E:/wifi/deliverables/mmfi_cvpr_three_seeds_20261019`.
  Existing historical sections below remain as provenance, not current status.
- Latest manuscript instruction: preserve the original ICLR prose and section
  progression wherever correct, retaining the CVPR working format. Only update
  actual A1 method/results and unsupported old empirical statements; reuse the
  original conceptual leg-payload figure panel for later author editing.
  Separate deliverable: `E:/wifi/deliverables/mmfi_cvpr_iclr_preserved_20261019`.
  Scientific table sources remain identical to the approved three-seed v2;
  original ICLR files and prior PDFs/ZIPs are not overwritten.

- Origin Skill: academic-research-suite / deep-research + experiment-agent.
- Origin Mode: khảo sát có mục tiêu và lập kế hoạch; không phải systematic review.
- Origin Date: 2026-10-15, Asia/Bangkok.
- Verification Status: PLAN; chưa có kết quả mới cho toàn bộ kế hoạch này.
- Version Label: mmfi_cvpr_experiment_plan_v1.
- Phạm vi được phép ở lượt này: khảo sát nguồn công khai và ghi note local. Không push, không khởi chạy MICA, không sửa baseline/runner trong lượt này.
- Cập nhật quyền ngày 2026-10-16: người dùng đồng ý triển khai, kiểm thử và push evaluator tính nốt T1 từ checkpoint; người dùng tự pull và chạy trên MICA. Phạm vi cũ ở dòng trên là lịch sử lúc soạn note.
- AI hỗ trợ tìm nguồn, đối chiếu code và soạn kế hoạch. Không giả định tác giả đã tự đọc tất cả nguồn; cần tự kiểm tra trước khi trích vào manuscript.

## 0. Checklist bắt buộc trước mỗi lượt chạy (người dùng xác nhận lại 2026-10-16)

### Chốt hiện hành 2026-10-20: T5 FT / ANP / NAD / FT-SAM, ba seed

Sau bộ ba ANP/NAD/FT-SAM, người dùng yêu cầu chọn thêm một hoặc hai defense.
Bổ sung **FT** làm đối chứng clean fine-tuning cơ bản: bộ hiện hành là **FT,
ANP (HPE-adapted), NAD, FT-SAM**, **seeds 42, 43, 44**, kèm **No defense A1**
ghép cặp trong từng seed. Không chạy thêm Fine-Pruning trong batch mới; không
xóa, đổi tên hoặc sửa kết quả FP đã có. Các danh sách cũ bên dưới giữ làm lịch sử.

- Đây là quyết định điều chỉnh **sau khi xem pilot seed 42**, không phải bộ
  defense được định trước khi nhìn kết quả. Giữ và ghi nhận đầy đủ kết quả
  FT/FP đã đánh giá khi báo cáo; không dùng thay đổi này để suy ra kháng mọi
  defense hoặc che kết quả Fine-Pruning bất lợi.
- Chỉ repair **Proposed A1/rho=.1** trên full MM-Fi protocol1-s1; reuse đúng
  checkpoint victim 50 epoch của từng seed trong
  `mmfi_three_tables_4090_win_fresh_v1`. Không train lại victim, đổi trigger,
  thêm baseline attack, Blended hoặc POR trong lượt này.
- Mục tiêu là **15 record**: 3 seed x (No defense + FT + ANP + NAD + FT-SAM), gồm
  **12 repair pipelines**, không có nghĩa tất cả đều phải train mới nếu reuse
  artifact hợp lệ. Chỉ báo mean +/- sample SD (ddof=1) khi đủ cả ba seed
  đã audit cho từng hàng; một seed không phải bảng ba seed hoàn chỉnh.
- No-defense/FT/NAD/FT-SAM seed 42 đã được người dùng cung cấp trong bảng pilot.
  Chỉ reuse artifact sau khi kiểm tra checkpoint, recipe, clean-bank, split,
  input identities và provenance; không coi bảng terminal là full artifact
  replay mới. Không chạy lại các kết quả hợp lệ chỉ vì đổi danh sách defense.
- Giữ clean bank chung tối đa 5% original clean TRAIN/seed, tách khoảng 80/20
  **theo sequence** thành repair-fit/repair-selection; không dùng official TEST
  để chọn pruning/mask/strength/checkpoint. Giữ nguyên recipe NAD và FT-SAM
  đã khóa (kể cả FT); không tune theo target score của pilot.
- ANP chỉ thích nghi loss MPJPE và lớp neuron/BatchNorm cần thiết cho HPELi,
  không cải tiến defense và không gọi là reproduction CIFAR nguyên bản. Phải
  pin nguồn/defaults, khóa tham số và quy tắc chọn bằng clean-selection, test
  gradient/mask/pruning và identity trước khi chạy GPU. Chưa suy ra ANP mạnh
  hay yếu hơn Fine-Pruning khi chưa có kết quả.
- Metric full TEST: T-MPJPE và Target-PCK trên target joints [2,3], dose 1,
  không alignment; Target-PCK threshold .2, scale từ clean GT joints (5,12).
  Clean MPJPE dùng **toàn bộ 17 khớp**; Triggered-GT MPJPE dùng **[2,3]** so với
  pose thật. Hướng defender: T-MPJPE tăng, Target-PCK giảm, hai lỗi GT giảm.
  Giữ sáu-dose/DCE diagnostic; không dùng DCE giảm làm bằng chứng defense tốt.
- CSV/JSON bổ sung Clean PA-MPJPE, Clean MPJPE trên **cùng [2,3]** của chính
  repaired model, no-trigger T-MPJPE/Target-PCK với target d=1, và gap
  `Triggered-GT - clean-target`. Gap âm được giữ nguyên, không clip. Không
  so trực tiếp lỗi triggered hai khớp với clean toàn bộ 17 khớp để kết luận
  recovery; đây cũng không phải clean-trained floor độc lập. Báo paired
  `repair - No defense` trong **cùng seed**, rồi tính mean/sample SD của ba
  delta. Chênh lệch PCK dùng percentage points, không trừ hai SD.
- Giữ runtime Windows workers=0, tối đa **2 job/GPU** như đã chốt. Bổ sung ANP
  phải dùng manifest/output phiên bản mới hoặc sidecar tách biệt; không sửa
  manifest/checkpoint/cache T5 v1 hay T1--T3. Reuse kết quả cũ phải có binding
  tới artifact nguồn, không gán hash kế hoạch mới cho một record cũ.
- Phiên bản riêng: `run_a1_defenses_v2.py`, manifest
  `a1_defenses_v2.resolved.json`, output mặc định
  `mmfi_a1_defenses_ft_anp_nad_sam_s42_43_44_v2`; helper Windows
  `Start-A1DefenseTaskV2.ps1`. V1 không bị sửa scientific sources. Reuse v1
  bằng copy artifact đã audit, kèm original binding/record/hash riêng; không
  giả record cũ thành lần train/evaluate mới. Partial checkpoint v1 chưa
  complete không được gán sang một scientific binding khác.
- ANP: pin official mechanism và scalar defaults của BackdoorBench; 2000
  mask iterations, lr=.01, batch256, eps=.4, alpha=.2, 1 noise step; grid
  threshold 0:.05:.90. Chọn threshold lớn nhất có clean-selection MPJPE
  tăng không quá 10%, identity fallback được ghi rõ; **không chọn theo TEST
  ASR** như nhánh selection của BackdoorBench. Không thêm post-pruning FT.
- **Trạng thái implementation:** đã đạt **308 test local theo nhóm** (60 ANP,
  137 metric/reuse/runner/helper/worker, 111 regression v1); gồm một iteration
  HPELi CPU bằng tensor tổng hợp và NPZ identity fixture đủ 33264 hàng, không
  phải full MM-Fi run. Lượt gộp bị kẹt trong môi trường local không được tính
  là pass; chỉ tính các nhóm đã thoát exit code 0. Người dùng đồng ý
  commit/push sau khi test. Hướng dẫn: `A1_DEFENSES_V2_VI.md`.
  Chưa launch GPU từ workspace này, chưa có kết quả full MM-Fi ANP và chưa
  xác nhận hoàn tất bảng T5 mới. Config HPE là adaptation đã khai báo, không
  phải reproduction CIFAR hoặc một cải tiến defense.

### Cập nhật 2026-10-19: T5 seed 42, hai job runtime trên máy Vân

Người dùng **báo** đã hoàn tất T1–T3 seeds 42/43/44 trên máy Vân; chưa coi đây
là audit mới của assistant khi chưa nhận ZIP/checker. Người dùng yêu cầu pilot
defense seed 42, sau đó chốt **2 job song song** và đồng ý push sau khi test.
Các chỉ đạo “defense để sau” ở mục lịch sử bên dưới không còn chặn pilot này.

- Vẫn chỉ Proposed A1/rho=.1, full MM-Fi protocol1-s1, cùng checkpoint 50 epoch
  trong `mmfi_three_tables_4090_win_fresh_v1`; không retrain victim/đổi trigger.
  No defense + FT/FP/NAD/FT-SAM; mọi recipe, bank, metric và workers=0 giữ nguyên.
- Sidecar runtime `run_a1_defenses_parallel.py` có 1 hoặc 2 slot trên một device,
  không sửa scientific sources/hash/manifest cũ. Clean bank chuẩn bị trước khi
  spawn; completed record audit mới được skip; epoch/stage checkpoint được reuse.
  Chỉ một controller giữ khóa root; phải dừng owner/orphan cũ trước khi chuyển.
- Windows helper `remote_tools/Start-A1DefenseTask.ps1` chạy qua Task Scheduler;
  `-StopExisting` chỉ dừng đúng T5 output đã chọn, không kill Python/YOLO khác,
  không xóa checkpoint/lock. Không auto-retry, không đổi batch để che lỗi OOM.
- Hai job không bảo đảm nhanh hơn hay đủ VRAM/RAM. Test local/toy subprocess
  không thay full MM-Fi/GPU run; chưa báo kết quả defense hay độ kháng defense.
  Một seed đủ 5 record là pilot; vẫn partial so với bảng T5 ba seed/15 record.

### Bổ sung mới nhất: T5 local, không đổi batch T1–T3 (2026-10-18)

Người dùng đã đồng ý **tạo code, kiểm thử và push GitHub** bộ defense **FT, FP, NAD,
FT-SAM**, kèm No defense. Chỉ đạo hiện hành: kiểm tra seed 42/43 của T1–T3 trước;
ưu tiên hoàn tất seed 43 với tối đa 2 job nếu còn thiếu, **defense để sau**.
Chưa khởi chạy thí nghiệm mới trên máy Vân/MICA từ workspace hiện tại.
Đây là batch T5 riêng, không bổ sung vào hoặc sửa manifest/source/checkpoint của
T1–T3 đang chạy. Các mục lịch sử dưới đây vẫn giữ nguyên để truy vết.

- Theo chỉ đạo bổ sung cùng ngày: **chỉ adapt defense và chỉ đánh giá trên Proposed
  A1**, không cải tiến defense, không chạy lại baseline truyền thống. Reuse A1
  checkpoint 50 epoch/rho=.1 của từng seed; không train victim lại hoặc sửa trigger.
- Pilot seed 42 trước; code hỗ trợ 43/44 sau. Mỗi seed có 5 hàng No-defense/repair,
  4 repair pipelines; đủ ba seed (15 hàng) mới báo mean ± sample SD. Clean MPJPE
  trên A1 vẫn bắt buộc; không repair thêm clean model. Không thêm Blended/POR.
- Chung tối đa 5% original clean TRAIN/seed, tách khoảng 80/20 theo sequence
  thành repair-fit/repair-selection. Selection chỉ held out khỏi repair fitting;
  original victim có thể đã thấy dữ liệu ấy. Không dùng `validation` hiện tại vì
  nó trùng official TEST; không chọn defense bằng attack score/TEST.
- T5 dùng full TEST, T-MPJPE, Target-PCK, Clean MPJPE và Triggered GT-MPJPE
  trên joints [2,3]. Defense tốt làm target matching kém đi nhưng lấy lại pose thật;
  model hỏng không được tính là defense thành công. Sáu-dose DCE giữ diagnostic.
- Profile **mmfi_a1_defenses_source_default_adapt_v1** giữ số mặc định/source
  schedule: FT/FP/SAM 100 epochs, NAD 10+10, SGD .01/.9/WD=.0005, batch256,
  NAD betas [500,1000,1000], SAM radius2.0/adaptive. Chỉ adapt pose loss/layer,
  bank dữ liệu, FP clean-selection và Windows workers0. Mọi nhánh giữ final
  checkpoint, không clean-best fallback. Không gọi đây là source classification
  reproduction hoặc config tối ưu MM-Fi. Pin code/default-YAML hashes và giữ
  output T5 riêng; không đổi worker/manifest T1–T3.
- Runner mới `ATKBackd/run_a1_defenses.py`; hướng dẫn `A1_DEFENSES_VI.md`;
  nguồn `ATKBackd/defend/a1_defense_sources.json`. Code/test local không phải
  kết quả full MM-Fi/GPU hay thí nghiệm paper đã hoàn tất. Chưa chạy T4/T6/T7.
- `check_a1_three_tables.py --outdir EXISTING_RUN --seeds 42 43` chỉ đọc và
  kiểm tra các artifact đã có; không train, không sửa checkpoint/manifest,
  không launch defense hay tự chuyển sang seed 44. Đếm đủ file không tự là
  kết quả đã verify; checker phải xác minh binding/cache/checkpoint/inputs.

### Cập nhật hiện hành: ba bảng chính, ba seed (2026-10-16)

Người dùng đã chốt triển khai **chỉ T1, T2, T3 trên MM-Fi**, seeds **42, 43, 44**;
đồng ý push sau khi kiểm thử. Người dùng tự pull và khởi chạy trên MICA; chưa có
kết quả seed 43/44 chỉ vì code được chuẩn bị. Các trạng thái thiếu triển khai
ở các mục lịch sử bên dưới không thay thế quyết định mới nhất này.

- **T1:** Clean reference + sáu baseline native của comparison v3 + Proposed A1.
  Không thêm Blended/POR hay đổi tham số baseline. Cột chính: **T-MPJPE ↓,
  Target-PCK ↑, Clean MPJPE ↓, Clean PA-MPJPE ↓**. Dose 1 và threshold 0.2
  được định nghĩa trong caption/protocol, không nối vào tên cột.
- **T2:** ma trận sáu hàng dose, ba cột rho=.1/.2/.4; mỗi ô T-MPJPE.
  Kèm summary Clean MPJPE và mean-positive DCE cho từng rho. Không train riêng
  theo từng dose; mỗi victim được đánh giá đủ sáu dose.
- **T3:** A1--Full (auxiliary ✓, coupling ✓), A2 (✓, ×), A3 (×, ✓),
  A4 (×, ×). Ba cột kết quả: **T-MPJPE ↓, DCE ↓, Clean MPJPE ↓**.
  Giữ đúng bốn nhánh factorial đã chạy, không tối ưu Proposed tiếp.
- Mỗi ô tính metric trên full TEST trong từng seed, rồi **mean ± sample SD
  (ddof=1)** trên ba giá trị seed. DCE summary lấy năm dose dương trong từng
  seed trước; không đưa d=0 vào mean hoặc dùng các frame làm lần lặp seed.
- Dùng chung đúng A1 giữa T1/T2(rho=.1)/T3 trong mỗi seed. Tổng cộng 13 scientific
  victims/seed. Mặc định reuse 13 seed-42 victims sau kiểm tra config/cache/
  checkpoint/poison/input identities; chỉ train **26 victims mới** cho 43/44.
  Không sửa thư mục lịch sử, không tự retrain seed 42 khi reference thiếu.
- Launcher mới: `23_run_a1_three_tables.sh`; hướng dẫn:
  `A1_THREE_TABLES_VI.md`; output riêng:
  `runs/mmfi_a1_three_tables_s42_43_44_v1`. Chỉ xuất complete sau đủ 39 cells.
- Khóa target joints [2,3], PCK scale từ clean GT (5,12), threshold .2, full
  protocol1-s1/split seed 0, SGD recipe 50 epochs đã chốt. Native noise/cover
  không được gọi equal-budget; method vẫn là development-selected.
- T4 giữ cho supplementary; **không chạy thêm T4/T5/T6/T7 trong batch này**.
  Diagnostics/no-trigger controls vẫn được giữ, không thêm cột metric linh tinh.

**Mở note này trước khi định chạy, sửa runner, đóng gói hoặc gọi một bảng là hoàn chỉnh.** Bảng ở Mục 5 là kế hoạch thực nghiệm người dùng đã chốt; note không tự cấp quyền khởi chạy hay push. Trước mỗi job, đối chiếu và ghi rõ: ID bảng; các phương pháp và clean/control reference; dataset/split; seed/rho/dose/epochs; định nghĩa và hướng của từng metric; code đánh giá metric đã có chưa; checkpoint/artifact nào được reuse và fingerprint có khớp không. Nếu thiếu hàng hoặc metric bắt buộc, dừng để thống nhất thay đổi hoặc gắn nhãn **exploratory/incomplete**, không gọi là bảng chính của bài.

**Sửa danh sách T1 theo chỉ đạo mới của người dùng (2026-10-16): không chạy Blended.** T1 hiện gồm **6 baseline** BadNets-CSI, WaNet-CSI, FTrojan-CSI, FIBA-CSI, SIG-CSI, BppAttack-CSI; thêm Proposed và **Clean reference**. Bốn metric chính vẫn là **T-MPJPE@d=1 ↓, Target-PCK@0.2 tại d=1 ↑, Clean MPJPE ↓, Clean PA-MPJPE ↓**. Clean PCK không phải Target-PCK; không thay thế một cột bằng cột kia. Kết quả Blended cũ được giữ riêng, nhưng không thuộc T1 hiện tại và không tuyên bố Proposed hơn Blended. Nếu người dùng đổi danh sách tiếp, ghi phiên bản kế hoạch mới trước khi chạy.

**Đối chiếu lượt `mmfi_a1_native_comparison_s42_v3` (2026-10-16):** run seed 42/50 epochs/rho=.1 đã hoàn thành, gồm đúng 6 baseline hiện chọn và A1 reused. **Không chạy thêm Blended.** T1 vẫn thiếu **Clean reference** và `Target-PCK`; ba scalar T-MPJPE, Clean MPJPE, Clean PA-MPJPE đã có. Vì vậy run này **chưa phải T1 hoàn chỉnh**. Giữ nguyên kết quả và provenance; không xóa, sửa số, hoặc suy ra Target-PCK từ clean PCK/aggregate T-MPJPE.

**Bổ sung T1 từ checkpoint (2026-10-16):** người dùng đã xác nhận 8/8 checkpoint còn trên MICA. Evaluator `eval_a1_target_pck.py` / launcher `19_eval_a1_target_pck.sh` đọc 6 baseline v3 + A1 `full` + Clean cũ, evaluate full TEST tại d=1, tính Target-PCK và no-trigger controls. Chỉ xuất complete sau khi tái tính T-MPJPE/Clean MPJPE/PA-MPJPE khớp cache trong 0.05 mm. Output riêng `target_pck_eval/`, giữ cache/checkpoint cũ. Clean cũ cùng recipe nhưng thiếu pair-ID hash tại thời điểm train; evaluator kiểm tra recipe, ordered split hiện tại và replay utility, đồng thời công bố giới hạn này. Xem `A1_TARGET_PCK_EVAL_VI.md`. Chưa có kết quả TEST mới cho đến khi người dùng chạy evaluator trên MICA.

**T1 đã hoàn tất và kiểm tra ZIP (2026-10-16):** 8/8 replay đạt; CSV/JSON và Target-PCK tái tính từ 33264 mẫu/8 NPZ khớp. A1: T-MPJPE@1=155.397937 mm, Target-PCK@.2=51.937530%, Clean MPJPE=174.684435 mm, Clean PA-MPJPE=104.599647 mm. Bảng trình bày Clean reference đầu, targeted methods theo T-MPJPE giảm dần. Chỉ seed 42, không suy thành mean/SD ba seed.

**Lượt tiếp theo được người dùng yêu cầu (2026-10-16): T2 và T4, seed 42.** T2 reuse A1 rho=.1 và coupling control từ factorial; chỉ train hai victim mới rho=.2/.4 theo cùng A1 recipe, đủ 50 epochs/full split. Theo bổ sung mới nhất của người dùng, bảng T2 chính hiển thị **đủ cả sáu dose**, thêm Clean MPJPE và mean-positive DCE cuối mỗi hàng. T4 reuse toàn bộ 7 targeted checkpoints T1; đo full TEST paired distortion và hai detector fit trên clean TRAIN, calibration theo sequence tách biệt. Hướng dẫn/lệnh trong `A1_T2_T4_VI.md`; chưa có số T2/T4 mới cho đến khi chạy trên MICA.

## 1. Quyết định chính

Không chốt chất lượng bài bằng số bảng. Kế hoạch gồm **7 nhóm thực nghiệm có câu hỏi riêng**, có thể trình bày thành 7 bảng, hoặc gộp một phần vào figure/phụ lục khi viết bài. Đây là thiết kế đề xuất cho bài này, không phải một quota hay bảo đảm được nhận ở CVPR.

1. Main comparison: hiệu quả tấn công so với chất lượng pose sạch.
2. Dose control × poison rate: điều khiển liên tục và đánh đổi với lượng poison.
3. Component ablation: coupling và carrier phụ thực sự đóng góp gì.
4. Stealthiness: nhiễu đầu vào và khả năng bị detector phát hiện.
5. Defense resistance: backdoor còn tác dụng sau sửa model/tiền xử lý không.
6. Generalization: backbone và split người/môi trường khác trong MM-Fi.
7. Target generality: không chỉ hoạt động với một nhánh skeleton/một hướng xoay.

**Giai đoạn A: hoàn thiện trên MM-Fi trước.** Chạy seed 42 để kiểm tra pipeline và tính nhất quán; sau khi khóa cấu hình chạy thêm 43, 44. Tất cả victim của thực nghiệm chính dùng full split và đủ lịch train, không lấy kết quả screening 15 epoch thay cho kết quả chính.

**Giai đoạn B: WiPose hoặc Person-in-WiFi-3D sau.** Mở rộng các nhóm comparison/dose/ablation quan trọng sang benchmark thứ hai. Chưa quyết định chọn dataset chỉ vì có kết quả thuận lợi. Với 2D và 3D, ghi riêng đơn vị, skeleton và chuẩn hóa; không trung bình sai số pixel với mm.

CVPR hướng dẫn reviewer xem thiết kế thí nghiệm, insight và đóng góp, không chỉ bảng SOTA. Số dataset/bảng không phải điều kiện đủ: [CVPR 2026 Reviewer Training Material](https://cvpr.thecvf.com/Conferences/2026/ReviewerTrainingMaterial).

## 2. Gate quan trọng trước khi chạy comparison mới

### 2.1 Không trộn setting cũ

Kiểm tra code local tại thời điểm soạn note:

- `configs/mmfi/{badnets,blended,wanet_source,ftrojan,fiba}.yaml`: rho=0.1 và có `comparison_peak_budget: original_postclip_linf_v1`.
- `run_proposed_ablation.py --no-peak`: Proposed native `md_multicarrier`, rho=0.4, không có peak projection.
- Vì vậy **không ghép Proposed không peak/rho=.4 với baseline có cap/rho=.1 rồi gọi là controlled comparison**. Đổi cách trình bày metric không khắc phục được khác biệt này.
- Những kết quả đã chạy vẫn giữ nguyên, không sửa số hay xóa đối thủ tốt hơn; gắn nhãn historical/development và đúng protocol.

### 2.2 Protocol đích đề xuất

- Main comparison, ablation, stealthiness và defense: rho=0.1 cho tất cả targeted-pose methods; đây là mức baseline truyền thống đã dùng. rho=.2 và .4 thuộc poison-rate study.
- Proposed được chọn: **multi-carrier không peak bound**. Không đưa lại peak bound vào method hoặc thêm hàng `w/o Peak` khi Full vốn đã không có thành phần này.
- Baseline truyền thống chỉ dùng các adaptation đã khai báo, không tối ưu thêm loss/trigger/poison selection để giúp hay làm yếu một bên.
- Muốn protocol native không cap cho comparison, cần **được người dùng đồng ý trước** khi tạo bản config baseline mới bỏ lớp comparison-cap bên ngoài. Giữ nguyên công thức trigger và cấu hình gốc, không ghi đè YAML/kết quả cũ. Note này **chưa cho phép và chưa thực hiện thay đổi đó**.
- Nếu baseline phải đóng băng cả cap, giữ kết quả ấy thành panel khác ngân sách; không gọi đó là cùng-budget superiority của Proposed không cap.
- Ngay cả native operators cũng không có cùng L2/Linf thực tế chỉ vì cùng training schedule. Phải công bố nhiễu thực tế. Nếu muốn claim equal-budget superiority, cần thêm một protocol budget-matched đã duyệt, hiệu chỉnh bằng TRAIN/validation và chạy cho mọi bên; không chỉnh trên TEST hoặc âm thầm đổi baseline.

**Trạng thái gate: cần chốt phiên bản ngân sách baseline trước khi triển khai batch comparison mới.** Chưa có lệnh duy nhất chạy đủ 7 nhóm trong repo.

## 3. Setting chung phải khóa và lưu manifest

- MM-Fi hiện tại: protocol1-s1, 133056 train / 33264 evaluation; giữ split seed 0. Không tự gọi split này là unseen-person/cross-environment nếu chưa kiểm tra global subject/session identities.
- Victim chính: HPELi, đúng pipeline DT-Pose đang dùng; SGD lr=0.001, momentum=0.9, weight decay=0, batch size 32, 50 epochs, khởi tạo mới, ordinary ERM.
- Seed victim/poison: 42, 43, 44; chạy 42 trước. Báo mean ± sample SD theo **seed**, không coi các frame liền kề là các lần lặp độc lập.
- Training doses: U(0.2,1), paired trigger/payload cho mọi targeted baseline; chỉ shuffled ablation phá coupling.
- Evaluation doses: 0, .2, .4, .6, .8, 1. Mỗi victim train một lần rồi evaluate nhiều dose.
- Target chính: pivot 1, target joints [2,3], xoay quanh Z với góc cực đại 40 độ theo dose; giữ skeleton topology đúng DT-Pose.
- Poison identities và trigger-dose draws giống nhau giữa methods trong mỗi seed/rho; giữ native cover của WaNet/FIBA nhưng ghi số cover/modified inputs riêng, không gọi toàn bộ số mẫu bị sửa là bằng nhau.
- Không cùng nghĩa epsilon giữa patch, blend, warp, DCT, FFT và multiplicative carrier. Ghi semantics và tham số từng operator, không chỉ ghi một con số epsilon.
- `method_carrier_seed=42` chỉ cố định RNG carrier phụ. Pattern Original vẫn theo experiment seed; không gọi toàn bộ trigger là bất biến xuyên seed nếu code chưa hỗ trợ điều đó.
- Official TEST đã được sử dụng ở các vòng phát triển trước: phải mô tả trung thực là development-selected method, không gọi đây là lần blind test đầu tiên. Từ kế hoạch này trở đi, khóa hyperparameter bằng TRAIN/validation và không chọn seed/target/threshold từ TEST.

## 4. Từ điển metric — không đổi tên số cũ thành metric mới

### T-MPJPE(d), mm, thấp hơn tốt hơn

Sai số dự đoán trên **các khớp mục tiêu** của input có trigger so với attacker target tại dose d. Comparison dùng d=1, không Procrustes-align trước khi tính metric tấn công này.

### Target-PCK@0.2(d), %, cao hơn tốt hơn

- Chỉ các khớp mục tiêu J; main MM-Fi J=[2,3].
- Ground truth để chấm là **attacker target**, không phải pose thật chưa tấn công.
- Chuẩn hóa bằng cùng khoảng cách từ **pose thật sạch** của từng mẫu: s_i = ||y_i[5]-y_i[12]||. Không lấy scale từ prediction và không fit theo từng method.
- Một khớp đạt nếu ||pred_trigger(i,j,d)-target(i,j,d)|| <= .2*s_i. Lấy trung bình trên mẫu và khớp, nhân 100.
- Comparison chấm d=1. Không tính cả 17 khớp vì 15 khớp không đổi sẽ che lấp thất bại ở nhánh mục tiêu.
- **Không gọi Target-PCK là pose-level ASR**. ASR tổng hợp cũ trong code cũng không phải metric này.
- Evaluate thêm same-victim **no-trigger Target-PCK với chính target d=1** và clean-trained victim + trigger. Lưu vào diagnostics/phụ lục để tránh điểm cao chỉ vì target gần pose thật hoặc ngưỡng dễ.
- Không thể suy Target-PCK mới từ các scalar Clean PCK/T-MPJPE hiện có. Nếu còn checkpoint và dataset, chỉ cần evaluate lại, không nhất thiết train lại.

### Clean MPJPE / Clean PA-MPJPE, mm, thấp hơn tốt hơn

Input không trigger, so với pose thật. PA-MPJPE chỉ dùng căn chỉnh cho **độ chính xác sạch**. Bảng comparison giữ hai metric này; không giữ Clean PCK làm cột PCK thứ hai.

### Dose-control error (DCE), mm, thấp hơn tốt hơn

Theo từng mẫu và khớp mục tiêu, đo norm của:

`(pred_d - pred_0) - (target_d - true_0)`.

Trung bình theo mẫu/khớp; cột summary trung bình **5 dose dương**. Không dùng dose 0 bằng 0 để làm đẹp trung bình. Spearman/schedule MAD chỉ là diagnostics, không đủ chứng minh điều khiển chính xác: model có thể biến đổi rất ít nhưng vẫn monotonic.

### Stealthiness metrics

- Relative L2: mức biến đổi CSI ở domain đầu vào model, mẫu clean/trigger paired; công bố aggregation và thêm raw-domain nếu có. Không đặt vào main comparison.
- Detector AUROC: label positive là triggered; báo riêng input-statistics detector và HPE-feature detector. Gần .5 nghĩa detector đã định nghĩa phân biệt kém; không chứng minh mọi detector đều thất bại. Báo giá trị thô; không gọi AUROC < .5 là bằng chứng stealthiness mạnh hơn mà bỏ qua khả năng đảo score.
- TPR tại ngưỡng được hiệu chỉnh **5% FPR trên clean calibration**: metric phụ quan trọng; báo cả FPR thật trên TEST. Không chọn threshold trên official TEST.
- Linf và signal-to-perturbation SNR lưu CSV/phụ lục. SNR và relative L2 trùng nhiều thông tin năng lượng, không cần cả hai làm cột chính.
- Không mượn LPIPS/SSIM/PSNR của ảnh rồi khẳng định cảm nhận hay RF stealthiness trên CSI khi chưa có validation tương ứng.

## 5. Bảy bảng/nhóm thực nghiệm cụ thể

| ID | Câu hỏi | Hàng / biến can thiệp | Metric hiển thị chính |
| --- | --- | --- | --- |
| T1 Comparison | Method đạt pose attacker mà giữ utility sạch tốt đến đâu? | BadNets-CSI, WaNet-CSI, FTrojan-CSI, FIBA-CSI, SIG-CSI, BppAttack-CSI, Proposed A1; Clean reference không xếp hạng attack. Không có Blended. | T-MPJPE ↓; Target-PCK ↑; Clean MPJPE ↓; Clean PA-MPJPE ↓; định nghĩa dose/ngưỡng ở Mục 4 |
| T2 Dose × Poison | Điều khiển dose và đánh đổi poison có ổn định không? | Sáu hàng d=0,.2,.4,.6,.8,1; ba cột rho=.1,.2,.4 | Mỗi ô T-MPJPE ↓; kèm Clean MPJPE và mean-positive DCE theo rho; full curves lưu artifact |
| T3 Ablation | Coupling và carrier phụ đóng góp độc lập/tương tác thế nào? | A1--Full (✓,✓); A2 (✓,×); A3 (×,✓); A4 (×,×) theo thứ tự auxiliary/coupling | T-MPJPE ↓; mean-positive DCE ↓; Clean MPJPE ↓ |
| T4 Stealthiness | Có ít nhiễu và khó phân biệt clean/trigger không? | Cùng targeted methods, cùng d=1/rho và cùng input identities | Relative L2 ↓; AUROC-input và AUROC-feature (gần .5 khó phân biệt); TPR@calibrated-5%-FPR ở panel/phụ lục |
| T5 Defense resistance | Sau defense model có trở lại dự đoán đúng thay vì chỉ bị phá hỏng không? | Hiện chốt 2026-10-20: No defense A1 + FT, ANP (HPE-adapted), NAD, FT-SAM; seeds 42/43/44. Giữ kết quả FP cũ; v2 local test không phải full run. | T-MPJPE; Target-PCK; Clean MPJPE (17 khớp); Triggered-GT MPJPE ([2,3]); cùng-khớp clean và paired delta trong CSV/JSON; dose/ngưỡng ở protocol |
| T6 MM-Fi generalization | Kết quả có phụ thuộc một victim/split dễ không? | Backbone chính + backbone thứ hai đã reproduce sạch; split chuẩn + held-out subject/environment đã audit | T-MPJPE@1 ↓; Target-PCK ↑; Clean MPJPE ↓, trình bày từng setting riêng |
| T7 Target generality | Có điều khiển được ngoài một nhánh/hướng target mặc định không? | Nhánh chân A, chân B, nhánh tay và một hướng xoay đảo dấu được định trước | T-MPJPE@1 ↓; Target-PCK ↑; mean-positive DCE ↓; Clean MPJPE ↓ |

Các metric target ở T5 nhìn từ attacker vẫn có hướng thấp T / cao Target-PCK. **Defense hiệu quả** làm điều ngược lại, đồng thời giữ/lấy lại accuracy với pose thật; caption phải nói rõ để không tô đậm sai hướng.

### T1 — phạm vi đối thủ và clean reference

Giữ nguyên 6 baseline hiện chọn (không Blended) và kết quả lịch sử của các phương pháp bị loại ở hồ sơ riêng. Đây là các **CSI/HPE dose adaptations**, không phải pipeline phân loại nguyên bản. Không tuyên bố hơn phương pháp không có trong T1.

INFOCOM/POR hiện có là staged encoder-tampering/representation attack, rho/dose/objective khác targeted pose. Giữ diagnostic riêng, không dùng T-MPJPE để xếp hạng nó như thể nó được tối ưu cùng mục tiêu. Không phát minh một adaptation mới rồi gọi là reproduction gốc.

Nếu reuse checkpoint cũ: kiểm tra khớp split/target/training/poison/operator/budget/seed/source, evaluate metric mới và giữ provenance. Một kết quả cũ có cap hoặc rho khác không trở thành cùng protocol chỉ bằng việc chép vào bảng.

### T2 — bảng dạng grid, không train theo từng dose

- Mỗi rho là một victim riêng cho mỗi seed; không cần train 3 hoặc 6 victim chỉ vì bảng có nhiều cột dose.
- Dùng đủ 6 điểm cho plot; d=0 là identity control, không tính trong positive-dose DCE summary.
- Vẽ DCE(d) và T-MPJPE(d), ít nhất Full và w/o Coupling ở rho chính; giữ full curves của mọi method T1 trong artifact. Không thêm Blended vào lượt chạy mới.
- Không dùng T-MPJPE ở các target dose khác nhau để dựng một frontier equal-budget như thể cùng endpoint. Nếu vẽ efficacy–distortion, dùng cùng target d=1 cho các điểm được so sánh.

### T3 — ablation đúng thành phần hiện tại

| Carrier | Coupling | Tên hàng |
| --- | --- | --- |
| Multi-carrier native | Paired | Full |
| Multi-carrier native | Shuffled | w/o Coupling |
| Original unit-RMS | Paired | w/o Auxiliary |
| Original unit-RMS | Shuffled | w/o Both |

Giữ tập poison, trigger doses, payload-dose marginal, victim/optimizer cố định trong mỗi seed. Original unit-RMS không có cùng L2 thực tế với multi-carrier; phải công bố chứ không nói removal là equal-L2 ablation.

`w/o Both` là thiết kế factorial được **đề xuất mới**, chưa có trong launcher 16. Nếu manuscript muốn claim motion-derived prior/physics prior là đóng góp, thêm một random-prior unit-RMS control được định trước, hoặc thu hẹp claim; không tự thêm nhiều module chỉ để tăng số hàng.

### T4 — đánh giá phát hiện chứ không chỉ đo nhiễu nhỏ

Đề xuất hai detector kiểm tra đơn giản, khai báo rõ không phải chứng minh kháng toàn bộ SOTA:

1. **Input-statistics detector:** Isolation Forest trên thống kê CSI theo link/subcarrier/packet; fit trên trusted clean TRAIN bank.
2. **Feature detector:** score Mahalanobis với covariance có regularization trên feature HPE cố định; fit mean/covariance trên trusted clean TRAIN bank của từng victim.

Cả hai không dùng trigger labels để fit. Chia fit/calibration theo subject/session/sequence khi khả thi; giữ clean và triggered phiên bản của cùng frame trong cùng partition. Hyperparameter, chiều score và threshold khóa bằng TRAIN/calibration, không chọn trên TEST. Ghi rõ defender được phép có trusted clean data/model-feature access đến đâu. Đây là input-level detection, không phải model-level detector AUC trên 3 model/3 seed.

Đo paired distortion trên cùng evaluation identities, ghi domain raw/normalized và n thật. Linf vẫn có thể được đo dù method không áp peak bound; **đo peak không phải đưa peak cap trở lại**.

### T5 — ba loại defense phải phân biệt

- Clean fine-tuning: trước mắt định trước 5% trusted clean TRAIN, 10 epochs, lr=1e-4, cùng protocol cho mọi victim; nếu cần chọn một setting khác, chỉ dùng validation và ghi lại trước khi mở TEST.
- Fine-pruning: chọn channel/layer có thể prune đúng trong HPELi; xếp activation trên clean bank; audit shape/forward và utility sạch. Không bê class-logit inversion hoặc STRIP entropy phân loại sang regression rồi gọi nguyên bản.
- Preprocessing CSI: lựa chọn nhỏ có cơ sở như smoothing nhẹ; kèm cùng biến đổi trên clean inputs. Noise/packet loss/gain drift là stress tests riêng; kiểm tra không phá alignment CSI–pose.
- Trước/sau defense dùng chính checkpoint bị cài backdoor, không train lại một victim dễ hơn. Báo cả target error và triggered-to-true error, tránh việc model hỏng được đếm là defense thành công.
- Giữ sweep strength/utility curve trong phụ lục, không chọn chỉ một mức defense thuận lợi sau khi nhìn TEST.

### T6 — MM-Fi có thể kiểm tra thêm trước khi tải dataset thứ hai

1. Audit split identities theo người/sequence; thêm một protocol held-out environment hoặc subject chính thức, cố định trước khi chạy. Không suy từ random per-action split thành cross-subject generalization.
2. Thêm backbone CSI/HPE thứ hai từ code tác giả sau khi reproduce clean utility hợp lý; ưu tiên xem DT-Pose chính thức. Giữ cùng attack protocol giữa methods trong mỗi backbone, dùng đúng training recipe của backbone đó.
3. Local `models/factory.py` hiện chỉ hỗ trợ HPELi; MetaFi++ và GraphPose-Fi đang bị chặn vì output/topology. Vì vậy backbone thứ hai là **việc cần triển khai/kiểm chứng**, không phải đổi tên config rồi có ngay.

Cross-backbone retraining là mỗi backbone có victim độc lập. Nếu không train trên backbone mới thì phải gọi đúng là trigger transfer, không gọi là kết quả đã huấn luyện tương đương.

### T7 — target set định trước, không chọn target dễ nhất

Đề xuất 4 setting: target chính pivot 1 → [2,3], nhánh đối diện pivot 4 → [5,6], nhánh tay pivot 11 → [12,13], và target chính với góc xoay Z đảo dấu. Kiểm tra topology và plausibility trên TRAIN trước khi freeze.

Train victim riêng khi thay đổi target mapping; không lấy model chỉ học target A rồi gọi evaluate target B là một attack đã được huấn luyện cho B. Giữ protocol cho Proposed, một baseline không-Blended được định trước và single-carrier control; không chỉ chọn baseline thất bại. So sánh từng target riêng, không gộp sai số các nhánh có scale khác nhau thành một con số không giải thích.

## 6. Figure và sanity checks cần đi cùng bảng

1. **Dose calibration:** curve DCE(d), T-MPJPE(d), mean ± SD theo seed. Không chỉ vẽ displacement/Spearman.
2. **Pose overlays theo dose:** cùng input, pose thật, no-trigger prediction, attacker target và triggered prediction; có target joints và drift của nontarget joints. Chọn ví dụ theo quy tắc cố định/quantile, có cả failure cases.
3. **CSI residual/spectrum và efficacy–distortion scatter:** cùng color scale và sample IDs; không tự normalize từng ảnh để che nhiễu. Scatter so tại cùng target d=1; norm nhỏ không tự chứng minh RF stealthiness.
4. **Controls học-backdoor:** clean-trained victim + đúng trigger; poisoned victim không trigger; poisoned victim + đúng key; poisoned victim + wrong/random key có năng lượng đo được tương đương. Không dùng chung tên ASR cho các tiêu chí khác nhau.

Nontarget drift, bone-length plausibility, direction/gain và legacy ASR giữ diagnostics/phụ lục, không ép thành một dãy cột chính. Chúng vẫn cần để kiểm tra kết luận không dựa trên lỗi metric.

## 7. Thứ tự chạy và tránh lãng phí

1. Chốt gate baseline budget/rho; tạo manifest mới, giữ snapshot cũ. Bổ sung Target-PCK tests trước.
2. Seed 42: clean reference + 7 targeted methods (6 baseline không-Blended + Proposed) tại rho=.1; kiểm tra training/eval, per-sample errors, Target-PCK sanity và digital distortion.
3. Sau khi metric/protocol khóa: chạy thêm seeds 43/44. Không chọn best seed. Nếu đổi method/config sau đó, phân biệt lại development với final runs.
4. Reuse đúng cùng scientific victim cho T1/T2(rho=.1)/T3 Full/T4/T5; rho=.2/.4 thêm victim, không nhân số victim theo số dose.
5. Hoàn thiện T2/T3 và detectors T4; T5 dùng các checkpoint đã có. Triển khai T6/T7 như batch riêng, không làm hỏng pipeline main.
6. Sau MM-Fi mới chuyển WiPose/PiW3D; khóa adapter/normalization/units/skeleton và clean sanity trước attack study.

Ví dụ số lượng **scientific victim configurations**, không phải ETA: 7 targeted methods + 1 clean reference + 2 rho bổ sung cho Proposed + 3 factorial ablations ngoài Full = 13 config/seed, tức 39 full victim trainings cho 3 seed nếu không reuse. Không bao gồm T6/T7, random-prior control, detector fitting hay defense fine-tuning. Checkpoint reuse chỉ hợp lệ sau khi audit config/input/budget, không chép cache khác profile.

Artifact cần giữ: resolved configs, source/plan/input/poison fingerprints, checkpoint, logs, six-dose metrics, per-sample joint/pose errors hoặc predictions + identities để tính Target-PCK/CI lại. Cache chỉ có aggregate không đủ tái tính metric mới. Thu thập training GPU-hours, trigger-injection latency, peak memory trong một bảng cost phụ lục nếu claim nhẹ/nhanh; đo trên cùng máy với warm-up và số lặp công bố.

## 8. Khảo sát nguồn: điều bài đã làm và điều mình đề xuất

Phạm vi tìm: nguồn primary tại CVF/ECVA, OpenReview/arXiv author manuscripts, IEEE/website tác giả và venue policy; từ khóa backdoor + evaluation/stealthiness/defense + pose/regression/RF. Đây là targeted narrative survey, không phải PRISMA search đầy đủ. Nhóm papers ảnh giúp thiết kế thí nghiệm; không tự chứng minh adaptation CSI hợp lệ.

| Nguồn | Bằng chứng đã đối chiếu | Áp dụng cho kế hoạch — đây là suy luận thiết kế, không phải số liệu của bài mình |
| --- | --- | --- |
| **INACTIVE, CVPR 2025** — [venue](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Invisible_Backdoor_Attack_against_Self-supervised_Learning_CVPR_2025_paper.html), [full text](https://arxiv.org/html/2405.14672v2) | §5.2 effectiveness; §5.3 stealthiness; §5.4 detection/robustness; §5.5 và Appendix generalization/ablation | Tách comparison, stealth, defense, generalization; không chỉ có một bảng thắng ASR |
| **WaNet, ICLR 2021** — [paper](https://arxiv.org/pdf/2102.10369) | §4 nhiều dataset/model, defense, noise-mode ablation và sensitivity; Appendix Table 5 nhiều warp ngẫu nhiên | Có ablation và biến thiên seed; physical screen-recapture của ảnh không phải RF OTA |
| **Refool, ECCV 2020** — [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550188.pdf) | Tables 2–3, Fig.3, §4.4: comparison, poison rate, trigger types và fine-tuning/pruning | Dose/rho và defense là câu hỏi riêng; venue đúng là ECCV, không phải CVPR |
| **SIBA, TIFS 2024**, DOI 10.1109/TIFS.2024.3411936 — [paper](https://arxiv.org/html/2306.06209v3), [code tác giả](https://github.com/YinghuaGao/SIBA) | §IV-A BA/ASR và distortion; IV-C sensitivity; IV-D STRIP/ABL/FP/NC/Scale-Up; IV-E architecture/data access | Không thay stealthiness bằng chỉ một norm; không chuyển entropy classification nguyên xi sang pose |
| **Color Space Perspective, TDSC 2025**, DOI 10.1109/TDSC.2024.3521942 — [author PDF](https://tianweiz07.github.io/Papers/25-tdsc.pdf) | §VI comparison/sensitivity/defenses; §VII-A ablation naturalness; §VIII adaptive defenses | Bỏ bound là lựa chọn method nhưng phải đo đánh đổi nhiễu; không tự gọi là thắng cùng stealthiness |
| **IRBA, TIFS 2024**, DOI 10.1109/TIFS.2023.3333687 — [paper](https://arxiv.org/pdf/2208.08052) | §V-A clean/attack/Chamfer; V-C preprocessing; V-D perceptual evidence; V-E sensitivity | Robustness phải theo modality; Chamfer/LPIPS của modality khác không được bê sang CSI không giải thích |
| **6DAttack, AAAI 2026** — [paper](https://ojs.aaai.org/index.php/AAAI/article/download/40855/44816) | §5.2 tách clean accuracy, target-pose accuracy và disruption ASR; §5.3 nhiều model/dataset và đoạn Defense/Fig.5 fine-tuning | Target-PCK của mình là adaptation đánh giá khớp; không đồng nhất ASR gây sai với đến đúng target |
| **SecureGaze, SenSys 2025** — [paper](https://arxiv.org/html/2502.20306v1), DOI 10.1145/3715014.3722071 | §3.2 benign/backdoored model controls; §5.1.6 attack/defended errors; §5.3 mitigation; §5.4 overhead | Defense phải báo utility và lỗi với target/GT; model-level AUC của bài không phải input-level AUROC của mình |
| **RF PTM/POR, INFOCOM 2025** — [paper](https://arxiv.org/html/2505.00881v1), [code tác giả](https://github.com/Tianyaz97/rf_backdoor) | §VI-B–E UASR/TR, SNR/ΔL2, Isolation Forest/STRIP, fine-tuning | Có precedent detection RF, nhưng objective khác targeted pose; không đưa vào cùng ranking T-MPJPE |

### Venue/ranking và giới hạn xác minh

- [ICORE portal CVPR](https://portal.core.edu.au/conf-ranks/604/) và [ICLR](https://portal.core.edu.au/conf-ranks/2273/) ghi A* trong ICORE2026. Rank venue không bảo đảm giá trị của mọi setup thí nghiệm trong từng paper.
- TIFS và TDSC được ghi **Q1 theo JCR 2024** trong các category tương ứng ở [bản JCR do UEFISCDI công bố tháng 6/2025](https://uefiscdi.gov.ro/resource-865584-JCR_2024.iunie2025.pdf): TIFS — Computer Science, Theory & Methods; TDSC — Computer Science, Hardware & Architecture. Đây không phải tuyên bố đã kiểm tra quartile 2026.
- Dùng bài HPE *Invisibility Cloak* như tài liệu liên quan nếu cần, nhưng hiện chỉ xác minh là [author arXiv preprint](https://air-ml.org/publication/2024/zhang2024invisibility/), không gọi nó là ICLR 2025.
- Nguồn chính thức [CVPR Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines) không cung cấp công thức số bảng/dataset để bảo đảm acceptance. Thiết kế này nhằm giảm các lỗ hổng bằng chứng, không phải dự đoán kết quả review.

## 9. Trạng thái implementation tại lúc soạn note (2026-10-15) — tránh hiểu note là đã chạy xong

- Đã có local profile no-peak **3 arms**, seed 42, rho=.4, 50 epochs; 404 code tests đã đạt ở lượt trước. Không có full MM-Fi run mới cho profile này do assistant khởi chạy.
- Note này đề xuất main rho=.1 và factorial 4 arms; chưa sửa launcher 16 để thành full matrix đó.
- Target-PCK mới, detector/calibration, defense suite, backbone thứ hai và target-generalization batch **chưa triển khai đầy đủ**. Clean PCK trong cache cũ không được đổi nhãn thành Target-PCK.
- Chỉ HPELi currently available trong model factory; không báo kết quả nhiều backbone trước khi reproduce sạch và kiểm tra skeleton.
- Không có claim RF stealthiness vật lý/OTA từ phép sửa tensor CSI. Nếu manuscript muốn claim vật lý, cần channel/hardware model được validate hoặc đo OTA có kiểm soát; nếu chưa có, giới hạn threat model ở **digital CSI training-data poisoning**.
- Những nhóm phụ có thể đưa vào appendix khi dàn trang, nhưng không bỏ một đối thủ hoặc failure case chỉ vì nó làm Proposed kém đẹp.

**Kết luận kế hoạch:** hoàn thiện bằng chứng theo 7 câu hỏi, chạy MM-Fi trước và mở rộng dataset sau; không cam kết CVPR chỉ vì đủ bảng hoặc Proposed đứng đầu một metric.
