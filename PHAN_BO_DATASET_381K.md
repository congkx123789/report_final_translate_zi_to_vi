# 📊 BÁO CÁO PHÂN BỔ CHÍNH XÁC TOÀN BỘ TẬP DATASET GỘP (381,013 MẪU)

> **File Dataset:** `dataset/master_finetune/filtered_balanced_with_foreign_names_278k.jsonl.gz`  
> **Dung lượng:** `239 MB` (Nén `.gz`)  
> **Tổng số cặp câu:** **`381,013`** mẫu

---

## 📈 1. Phân Bổ Theo Dải Độ Dài Tokens (Nguồn + Đích)

- **Độ dài Token Nguồn (`prompt_src`) trung bình:** **`162.9`** tokens
- **Độ dài Token Đích (`vi_text`) trung bình:** **`178.8`** tokens
- **Độ dài Tổng trung bình:** **`341.7`** tokens/mẫu

| Dải Độ Dài Token | Số Lượng Mẫu | Tỷ Lệ % | Đặc Điểm Ngữ Cảnh & Huấn Luyện |
| :--- | :---: | :---: | :--- |
| **0 – 50 tokens** | **37,774** | **9.91%** | Giao tiếp ngắn, khẩu ngữ, câu hội thoại nhanh |
| **50 – 100 tokens** | **48,308** | **12.68%** | Câu ghép, miêu tả hành động, câu có gắn tên riêng ngắn |
| **100 – 200 tokens** | **43,113** | **11.32%** | Đoạn văn ngắn 2-3 câu, chiêu thức, miêu tả đồ thị |
| **200 – 300 tokens** | **44,241** | **11.61%** | Đoạn văn vừa, hội thoại nhiều lượt giữa các nhân vật |
| **300 – 400 tokens** | **35,579** | **9.34%** | Đoạn văn dài, công pháp tu tiên & tên riêng quốc tế dài |
| **400 – 500 tokens** | **87,416** | **22.94%** | Khối văn bản truyện lớn, đại cảnh sa trường & đô thị |
| **500 – 600 tokens** | **16,811** | **4.41%** | Trích đoạn tiểu thuyết nhiều đoạn liên tiếp |
| **600 – 700 tokens** | **16,120** | **4.23%** | Trích đoạn tiểu thuyết dài |
| **700 – 800 tokens** | **41,741** | **10.96%** | Trích đoạn tiểu thuyết dài nguyên cảnh |
| **800+ tokens** | **9,910** | **2.60%** | Trích đoạn nguyên chương (~1k – 2.4k tokens) |
| **TỔNG CỘNG** | **381,013** | **100.0%** | **Phân bổ đều đẹp lý tưởng cho mọi dải độ dài** |

---

## 🏷️ 2. Phân Bổ Theo Cơ Chế Thẻ Dẫn Đường (Glossary Guidance)

| Phân Loại | Số Lượng Mẫu | Tỷ Lệ % | Tác Dụng Trong Quá Trình Huấn Luyện |
| :--- | :---: | :---: | :--- |
| **Có thẻ `<glossary>`** | **307,239** | **80.64%** | Huấn luyện cơ chế ép dịch theo từ điển Hán-Việt & Tên riêng tiếng Anh chuẩn |
| **Thuần túy `<src>`** | **73,774** | **19.36%** | Rèn luyện khả năng dịch tự do, mượt mà khi không có từ điển |

---

## ⏱️ 3. Dự Báo Thời Gian & Hiệu Năng Huấn Luyện (Trên RTX 5060 Ti)

- **Tốc độ xử lý:** ~8.9 mẫu/giây (với `batch_size=16`, `accum_steps=2`).
- **Tổng số Steps cho 1 Epoch:** **`11,906 steps`**.
- **Thời gian hoàn thành 1 Epoch:** **`~11.5 – 12.0 giờ`**.
