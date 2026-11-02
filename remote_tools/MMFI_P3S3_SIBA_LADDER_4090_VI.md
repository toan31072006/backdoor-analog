# SIBA / BackTime adapted; LADDER thiếu source — MM-Fi P3S3

## Material Passport

- Origin: academic-research-suite, source-fidelity audit, 2026-11-02 (Asia/Bangkok).
- Nguồn: [SIBA / TIFS 2024](https://arxiv.org/abs/2306.06209), [GitHub SIBA](https://github.com/YinghuaGao/SIBA/tree/2968e76070c532ac103b1571de77009c004ad04b); [BackTime / NeurIPS 2024](https://arxiv.org/abs/2410.02195), [GitHub BackTime](https://github.com/xiaolin-cs/BackTime/tree/9e0a1f9deecbacfdfea9159e6e8892148b44da9f); [LADDER / NDSS 2025](https://gsmaragd.github.io/publications/NDSS2025/NDSS2025.pdf), [GitHub LADDER](https://github.com/dzhliu/LADDER/tree/a752cf48805283fcf7becd88e4382199e40de2c1).
- Quyết định hiện hành: **chỉ thích nghi sang CSI → pose, không tự cải tiến phương pháp**. Giữ cơ chế/tối ưu/schedule tác giả khi nguồn đủ rõ; khai báo riêng thay đổi task và protocol so sánh.
- Scope: exploratory extension T6, **HPE-Li / MM-Fi Protocol 3 Setting 3 / seed42 / tối đa1job trên cuda:0**. Không tự chạy P1S1, backbone/seed khác, A1, BadNets hoặc defense.
- Không sửa checkpoint/manifest/source khóa/kết quả lịch sử. Bản fidelity sửa lại dùng profile/output **v2 riêng**, không resume vào v1 đã bind thuật toán/budget khác.
- Chưa có kết quả mới trên full MM-Fi/CUDA4090; local tests không xác nhận efficacy, hội tụ, tốc độ hoặc VRAM. Một seed không phải mean ± SD.
- Người dùng xác nhận **test rồi commit/push SIBA + BackTime** ngày 2026-11-03. Trạng thái publish phải đối chiếu commit/remote thực tế; chưa launch trên máy Vân. Quyền publish không cho phép kill job khác, thay recipe hoặc tự bỏ method.

## Trạng thái từng phương pháp

| Phương pháp | Giữ theo nguồn | Trạng thái |
| --- | --- | --- |
| SIBA | Sparse support chung các channel, sign-gradient/Linf projection; full TRAIN,200epoch, attacker B128, support refresh mỗi5epoch, chọn trigger theo mean TRAIN loss từng epoch | Đã sửa và kiểm local v2; vẫn cần full-input/CUDA gate máy đích trước launch |
| BackTime | GNN input-dependent, historical-anchor replacement, alternating surrogate/generator, attacker100epoch/B64, warmup epoch0..10, error-driven recipient selection | Chỉ **adaptive**, không có uniform variant trong run được duyệt |
| LADDER | Public source chỉ có inject/train với trigger đã được tìm từ trước | **BLOCKED**: thiếu author trigger-search source/artifact; không tự viết thuật toán thay thế rồi gọi adapted |

Queue mặc định giữ các method đã yêu cầu. Nếu có LADDER, **CheckOnly/Launch phải dừng trước GPU/full training**; không âm thầm bỏ LADDER để báo complete. Chỉ khi người dùng chủ động chọn **-MethodSet SibaBackTime** mới chạy riêng hai method còn lại. Method set được ghi trong plan/output, không phải fallback tự động.

### Vì sao LADDER bị chặn?

Tại commit đã pin, [train_freq_attack.py](https://github.com/dzhliu/LADDER/blob/a752cf48805283fcf7becd88e4382199e40de2c1/train_freq_attack.py#L188) đọc ./saved_vars/3objs_vgg11_cifar10_selected.pt; file này và chương trình search tạo nó không có trong repository. Guard tiếp theo còn yêu cầu có vars sau khi chạy trigger search. [data_poison.py](https://github.com/dzhliu/LADDER/blob/a752cf48805283fcf7becd88e4382199e40de2c1/data_poison.py#L297) chỉ thực hiện DCT injection từ tọa độ/biên độ đã cung cấp; giá trị FTrojan hardcode không phải trigger LADDER.

Paper có mô tả MOEA, nhưng bản search tự viết trước đây còn đưa vào selection/default riêng. Theo yêu cầu **chỉ dùng phương pháp/code tác giả**, không dispatch engine đó dưới tên LADDER. Không thay bằng random DCT/FTrojan hoặc tải nhầm trigger CIFAR rồi nhận là CSI-trained. Muốn mở gate cần source/artifact phù hợp từ tác giả và audit lại; không sửa hash/bypass guard.

## Cấu hình chung không đổi

- Full official P3S3: **240570 TRAIN / 80190 TEST**; không cắt TEST hoặc chọn kết quả theo TEST.
- Scientific victim HPE-Li: scratch **50epoch/final49**, ordinary MPJPE, physical B32, workers0, SGD **lr.001/momentum.9/wd0**, FP32. Không pretrained, scheduler, AMP, early stop, best-TEST, tự giảm batch hoặc retry.
- Rho.1: **24057 TRAIN frames**. SIBA dùng uniform; BackTime giữ error-driven selection nên có thể khác recipients. Không nhận hai method có cùng poison identities khi thực tế khác.
- Target chung pivot1 → joints[2,3], Z40°×dose; TRAIN paired U(.2,1), TEST doses0,.2,.4,.6,.8,1. Đây là **dose/payload extension của benchmark**, không phải tính năng native của các paper.
- T-MPJPE/Target-PCK@.2 chỉ hai target joints, so attacker target; scale clean GT(5,12). Clean MPJPE/PA-MPJPE chấm17khớp; DCE mean5dose dương, không lấy dose0 làm đẹp mean.
- Lưu same-victim no-trigger, Clean-model+trigger controls, full ordered IDs/predictions/GT/targets, input/source/poison/checkpoint hashes. Không chọn seed/trigger/checkpoint sau khi nhìn official TEST.

## Những phần thích nghi task phải công bố

### SIBA

Đổi ảnh RGB thành normalized amplitude CSI **3×114×10**; classifier thành HPE-Li, class-label objective thành target-pose regression. Mask H×W chung3antenna: **k100vị trí = tối đa300tọa độ**, epsilon8/255, sign step epsilon×.8. Sửa phép trải chỉ số mask cho input chữ nhật, không resize CSI thành ảnh vuông.

Trigger học trên **toàn bộ TRAIN/200epoch**, attacker B128, refresh support ở batch đầu mỗi5epoch, cap loss10 theo giá trị main của source; chọn snapshot theo **mean TRAIN loss từng epoch**, không theo từng minibatch hoặc TEST. Không còn bank1024/1000bước/refresh160bước. Cap chuyển sang pose-loss là khác biệt task cần ghi rõ; không tuyên bố cùng ý nghĩa số học với CE classification.

### BackTime

Đổi342biến antenna×subcarrier và10packet thành giao diện CSI→single-pose. Giữ history6/trigger4, learned graph/GCN + constant branch, historical-anchor replacement, DC penaltyλ.05, generator Adam.005/surrogate Adam.0001. Học **surrogate riêng từ scratch100epoch/B64**, warmup epoch0..10 rồi alternating poison/surrogate/generator. Không thay warmup bằng checkpoint Clean50 hoặc chỉ fit20epoch trên bank nhỏ. Victim sau đó vẫn scratchSGD50/B32 như mọi method.

Giữ error-driven selection bằng **online per-sample SmoothL1** từ warmup, không đổi sang static MPJPE ranking của Clean đã train xong. Chỉ adaptive được dùng; uniform variant bỏ khỏi plan chạy. Giữ chế độ train/BatchNorm của surrogate trong bước học generator theo source, không âm thầm eval/freeze BN để thay thuật toán.

Normalization generator dùng mean/std tính từ TRAIN. FFT feature giữ thao tác source trên đủ10packet bins theo từng window rồi aggregate TRAIN; không tự chọn3low bins. Các frame MM-Fi là cửa sổ độc lập: **không nối thành chuỗi liên tục giả**. Bỏ điều kiện greedy temporal-spacing của forecasting là thay đổi data/task bắt buộc công bố, không nhận đã tái lập temporal-overlap behavior gốc.

Single pose thay future-variable pattern; không bịa nhãn future CSI. Dose nội suy segment gốc→learned replacement, dose0 identity. **Epsilon.2 ở đơn vị TRAIN-standardized**: bound so historical anchor sau denormalize là `.2 × TRAINstd`, không phải actual residual Linf≤.2 so segment đầu vào. Không claim equal-noise/equal-epsilon với SIBA/A1 hoặc RF/over-air.

## Chi phí và thứ tự nếu chọn hai method khả dụng

-MethodSet SibaBackTime chọn rõ **Clean reference mới → SIBA → BackTime**:

1. Clean reference scratch50epoch/final49; đồng thời là frozen surrogate cho SIBA.
2. SIBA full-TRAIN optimization200epoch/B128 → đóng băng trigger → learned-trigger B32 CUDA smoke → victim mới scratch50epoch → full evaluation/controls.
3. BackTime **surrogate riêng scratch100epoch/B64 gồm warmup/alternating fitting**, không mượn trọng số Clean50 → đóng băng generator → learned-trigger B32 CUDA smoke → victim mới scratch50epoch → full evaluation/controls.

Có **3 full scientific victim/reference trainings +2 attacker optimization schedules riêng**, không chỉ ba job ngắn. B128/64 là attacker batch, không thay victimB32. Một worker tuần tự, không ghép job. Không bảo đảm ETA/VRAM; OOM dừng, không tự shrink budget/batch.

SIBA có `200 × ceil(240570/128) = 376000` bước gradient. BackTime có `100 × ceil(240570/64) = 375900` bước surrogate và `89 × 24057 = 2141073` bước generator từng recipient, ngoài ba lịch train50epoch. Không âm thầm batch/rút gọn bước generator để làm nhanh rồi gọi là cùng lịch nguồn.

**Prepare chỉ audit full inputs +native Clean B32 CUDA smoke2updates**; không full train/reference/search. Learned-trigger smoke thật (4poison+28clean,2updates trên victim tạm rồi bỏ) chỉ có sau khi học trigger trong Launch. Synthetic tests không thay gateCUDA.

## Lệnh handoff trên máy Vân

**Chỉ sau khi code đã thật sự publish và tải vào checkout mới.** Không pull checkout đang train, clone đè thư mục hoặc dùng SHA giả. Launcher không cài package/download data/update Git. Python mặc định tìm envs\dose-backdoor\python.exe rồi envs\mmfi-p3\Scripts\python.exe trong DataHome; nếu truyền -PythonExe khác phải dùng nhất quán mọi mode.

Handoff hiện tại chọn rõ **SIBA + BackTime**, không phải bộ ba có LADDER.
Tải vào một checkout mới, chẳng hạn `C:\Users\Van\code_siba_backtime_p3s3`;
pin commit đã được xác nhận publish trước khi Prepare và giữ nguyên checkout
giữa các pha/resume. Không cài lại env nếu env MM-Fi hiện có vẫn dùng được.

Người dùng đã gửi log `COMPLETE` của Adap-Patch P3S3 và P1S1. Hai run này
không phải dependencies hoặc checkpoint warm-start của batch mới. Nếu GPU
vẫn dùng nhiều VRAM hoặc còn job khác, chưa Prepare/Launch và không tự kill.
Helper không có cơ chế chờ GPU rảnh hay hàng đợi nền sau SSH: khi máy thực sự
sẵn sàng, gọi Prepare riêng rồi xem receipt trước khi gọi Launch. Trong Launch,
Clean -> SIBA -> BackTime đã là một hàng đợi tuần tự, tối đa một worker GPU.

~~~powershell
$S = 'C:\Users\Van\code_siba_backtime_p3s3\remote_tools\Start-MMFiP3S3SibaLadder4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect -MethodSet SibaBackTime
~~~

Xem yêu cầu đủ3method và lý do LADDER bị chặn (chỉ đọc plan, không train):

~~~powershell
$S = 'C:\Users\Van\code_siba_ladder_p3s3\remote_tools\Start-MMFiP3S3SibaLadder4090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect -BackTimePolicy adaptive
~~~

**Chỉ nếu người dùng chọn SIBA+BackTime trước**, dùng cùng -MethodSet SibaBackTime trong mọi mode; không bỏ cờ khi resume:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Inspect -MethodSet SibaBackTime
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Preflight -MethodSet SibaBackTime
~~~

Khi GPU không còn compute job khác, prepare rồi **dừng đọc receipt**:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare -MethodSet SibaBackTime
~~~

Sau gate đạt và người dùng quyết định full training, gọi riêng:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly -MethodSet SibaBackTime
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -MethodSet SibaBackTime
~~~

Không gộp Prepare+Launch trong loop. Launch foreground: **giữ SSH mở**, không có tmux/TaskScheduler/WMI bảo đảm sống sau disconnect. Không kill job khác/tự relaunch khi lỗi. Inspect in outputv2 thực tế; binding có method set. Không dùng runv1 hoặc đổi method/policy để resume.

Đọc status từ SSH khác:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status -MethodSet SibaBackTime
~~~

GPU0%, có checkpoint/PID sống không chứng minh complete. Phải đọc epoch/phase, evaluation artifacts và receipts. Nếu lỗi giữ nguyên lock/checkpoint/gate, gửi traceback/log; không sửa hash, xóa lock hoặc overwrite partial .saving để vượt gate.

## Trạng thái kiểm chứng

**Báo cáo291testpass trước đây thuộc implementationv1 đã đổi schedule/selection/bank; không chứng minh fidelityv2 hiện tại đúng.** Hai smoke HPE-Li cũ dùng input/label giảCPU, không phải MM-Fi/CUDA4090. Giữ báo cáo cũ làm lịch sử, không cộng lại thành gate mới.

Gatev2 phải kiểm lại source-fidelity, full-TRAIN streaming, epoch/warmup/batch semantics, adaptive selection, checkpoint/RNG resume, PS method-set forwarding và **LADDER fail-closed trước training**. Báo kết quả mới riêng sau khi lệnh thực sự hoàn tất. Chưa có full-data/CUDA efficacy, xác nhận đã chạy trên Vân hoặc bảng paper hoàn tất.

### Gate local v2 đã thực hiện ngày 2026-11-02

**330 passed, 10 warnings** trong79.23s. Warnings là `pin_memory` trên CPU không có accelerator. Bao gồm parity các bước SIBA với reference nguồn, BackTime warmup/online SmoothL1/adaptive selection và cập nhật generator từng recipient, exact resume optimizer/RNG, dispatch không dùng tensor bank nhỏ, fresh initialization, CLI PowerShell thật và regression runner lịch sử.

Hai smoke dùng kiến trúc HPE-Li thật với input/label **giả trên CPU**, có gradient/optimizer update; integration dùng schedule/data nhỏ được đánh dấu riêng. Không phải đã chạy200/100epoch trên MM-Fi thật, không xác nhận CUDA4090/VRAM/ETA/efficacy. Bài kiểm chặn LADDER chứng minh guard hoạt động, **không** chứng minh LADDER đã được tái lập.

```powershell
python -m pytest tests/test_siba_ladder_mmfi.py tests/test_backtime_mmfi.py tests/test_mmfi_p3s3_siba_ladder_runner.py tests/test_mmfi_p3s3_backtime_runner.py tests/test_mmfi_p3s3_siba_ladder_launcher.py tests/test_mmfi_p3s3_siba_ladder_integration.py tests/test_mmfi_p3s3_siba_ladder_model_smoke.py tests/test_mmfi_p3s3_source_adapted_dispatch.py ATKBackd/tests/test_mmfi_p3s3_three_attacks.py ATKBackd/tests/test_trojannn_mmfi.py -q
```

### Kiểm thử lại trước publish — 2026-11-03

Chạy lại nguyên bộ lệnh trên với output test mới, không sửa thuật toán:
**330 passed, 10 warnings, exit0 trong83.58s**. Output local:
`E:/wifi/deliverables/siba_backtime_publish_tests_20261103_a`.
Các warnings vẫn chỉ là `pin_memory` khi không có accelerator. Bao gồm kiểm
tra CLI PowerShell, full-TRAIN dispatch không dùng bank nhỏ, nguyên trạng
surrogate, fresh victim, optimizer/RNG resume, metrics và HPE-Li CPU/synthetic.
Không cộng hai lượt330 thành660 tests riêng biệt; không coi đây là full-data
hoặc CUDA4090 validation. Chưa có số TEST mới và chưa có benchmark thời gian
GPU cho200/100epoch optimization.
