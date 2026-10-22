# Model pipeline — flow tracker

> Cập nhật từ các file output lúc 2026-10-22 11:34:39. "Đã chạy" nghĩa là các file bằng chứng tồn tại; nó **không** có nghĩa model đã được chọn hay final test đã đạt.

~~~mermaid
flowchart TD
    A[Gold audit / 21 Core] --> B[Temporal split / 4 folds]
    B --> C[Dummy + LR]
    C --> D[Bỏ Year, Month, Week / 18 features]
    D --> E[RF và XGB]
    E --> F[Tune theo H1 và H2]
    F --> G[Ablation / F1 = RF17]
    G --> H[Thử calibration]
    H --> I[Error analysis]
    I --> J{Chốt model?}
    J --> K[Final refit]
    K --> L[Final test OOT]
    L --> M[Package / handoff]
    classDef done fill:#d9f4e4,stroke:#29845a,color:#12372a
    classDef current fill:#fff0bf,stroke:#b87700,color:#453000
    classDef pending fill:#eceff3,stroke:#8b98a5,color:#26323d
    class A done
    class B done
    class C done
    class D done
    class E done
    class F done
    class G done
    class H done
    class I done
    class J current
    class K pending
    class L pending
    class M pending
~~~

## Đang đứng ở đâu

- **Focus:** Chốt ứng viên RF17 (F1) với xác suất raw cho H1 và H2 trước khi mở final test.
- **Vấn đề cần giải:** Lợi thế validation khá nhỏ, nhất là H1. Cần quyết định tiêu chí chấp nhận và xác nhận 17 input có thể tạo đúng tại T-14.
- **Bước kế:** Ghi cấu hình cuối cùng, refit trên các development rows đủ điều kiện tại 2017-04-18, rồi đánh giá final test đúng một lần.
- **Quyết định gần nhất:** Calibration sigmoid/isotonic chưa cải thiện mean Log Loss hoặc Brier của F1 trên folds 2-4; raw hiện là ứng viên dẫn đầu.

## Trạng thái tự đọc từ artifacts

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| Gold audit & feature design | đã chạy | [output](gold_eda.ipynb) |
| Temporal split | đã chạy | [output](splits/split_v1/config.json) |
| Dummy + LR baseline | đã chạy | [output](reports/baseline_v1/) |
| Calendar ablation → 18 features | đã chạy | [output](reports/lr_calendar_ablation_v1/) |
| RF + XGB initial | đã chạy | [output](reports/rf_initial_v1/) |
| Tuning LR/RF/XGB | đã chạy | [output](reports/tuning_round1_v1/) |
| Feature ablation → F1 / RF17 | đã chạy | [output](reports/feature_ablation_round1_v1/) |
| Calibration thử nghiệm | đã chạy | [output](reports/calibration_round1_v1/) |
| RF17 error analysis | đã chạy | [output](reports/rf17_error_analysis_v1/) |

## Các gate cần tự xác nhận

| Gate | Trạng thái |
|---|---|
| Chốt model / feature / xác suất | chưa làm |
| Final refit | chưa làm |
| Final test OOT | chưa làm |
| Package & handoff | chưa làm |

## Kết quả để định vị vấn đề

| So sánh | H1 Log Loss | H2 Log Loss | Ý nghĩa |
|---|---:|---:|---|
| RF17 F1, 4 folds | 0.2649 | 0.3575 | Ứng viên dẫn đầu sau feature ablation; thấp hơn là tốt hơn. |
| RF17 F1 raw, folds 2–4 | 0.2751 | 0.3754 | Mốc so calibration. |
| RF17 F1 sigmoid, folds 2–4 | 0.2791 | 0.3782 | So cùng 3 folds với raw; không so trực tiếp với dòng 4 folds. |

**Đọc flow:** nếu một bước có output nhưng kết quả chưa thuyết phục, ghi câu hỏi ở tracker_notes.json; đừng đánh dấu gate cuối là xong chỉ vì đã có file. [Notebook train](model_train_main.ipynb) chứa logic và giải thích chi tiết.

## Cập nhật

Trong thư mục Model pipeline, chạy lệnh dưới đây và mở Markdown Preview của file này. Script quét artifacts và notes mỗi 5 giây; dừng bằng Ctrl+C.

    powershell -NoProfile -ExecutionPolicy Bypass -File .\Update-ModelPipelineTracker.ps1 -Watch

Sửa tracker_notes.json khi m đổi focus, quyết định hoặc trạng thái gate. Nếu chỉ muốn cập nhật một lần, bỏ tham số -Watch.