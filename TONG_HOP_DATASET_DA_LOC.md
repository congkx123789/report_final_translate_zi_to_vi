# 📊 TỔNG HỢP TOÀN BỘ CÁC TẬP DỮ LIỆU ĐÃ LỌC & CÂN BẰNG

## 📁 1. Danh Sách Các Tệp Dữ Liệu Sẵn Sàng Train

| Tên Tệp | Đường Dẫn Tuyệt Đối | Dung Lượng | Tổng Số Mẫu | Chi Tiết Thành Phần |
| :--- | :--- | :---: | :---: | :--- |
| **⭐ Bản 283k (Khuyên Dùng)** | [filtered_balanced_plus_foreign_283k.jsonl.gz](file:///home/alida/Documents/IT/all_model/My_alida_model/dataset/master_finetune/filtered_balanced_plus_foreign_283k.jsonl.gz) | **176 MB** | **283,729** | **Tập 236k** + **100% câu dài $\ge 300$t tên riêng (17.3k)** + **30k câu ngắn tên riêng** |
| **Bản 236k** | [filtered_balanced_dataset_236k.jsonl.gz](file:///home/alida/Documents/IT/all_model/My_alida_model/dataset/master_finetune/filtered_balanced_dataset_236k.jsonl.gz) | 209 MB | 236,368 | Giảm 5 nhóm đầu xuống 1/3 + 100% câu dài (>400t) |
| **Bản 625k** | [filtered_balanced_dataset_627k.jsonl.gz](file:///home/alida/Documents/IT/all_model/My_alida_model/dataset/master_finetune/filtered_balanced_dataset_627k.jsonl.gz) | 335 MB | 670,945 | 300k câu ngắn + 100% câu dài (>200t) |
| **Bản Gốc 5.46M** | [all_master_dataset_5.46M.jsonl.gz](file:///home/alida/Documents/IT/all_model/My_alida_model/dataset/master_finetune/all_master_dataset_5.46M.jsonl.gz) | 879 MB | 5,416,248 | Tập Master nguyên bản |

---

## 📊 2. Phân Bổ Chi Tiết Tập Dữ Liệu Tối Ưu Nhất (283,729 Mẫu)

> **File:** `dataset/master_finetune/filtered_balanced_plus_foreign_283k.jsonl.gz`  
> **Thời gian huấn luyện 1 Epoch dự kiến:** **`~7.5 – 8.5 giờ`** trên GPU RTX 5060 Ti.

| Nhóm Dữ Liệu | Số Lượng Mẫu | Tỷ Lệ Thu Thập | Ý Nghĩa / Mục Tiêu Huấn Luyện |
| :--- | :---: | :---: | :--- |
| **Dữ liệu 236k cân bằng gốc** | **236,368** | Giảm 1/3 câu ngắn, giữ 100% câu siêu dài | Đảm bảo văn phong truyện, tiên hiệp, kiếm hiệp, ngữ pháp tự nhiên |
| **Tên riêng quốc tế dài ($\ge 300$t)** | **17,361** | 🌟 **LẤY TOÀN BỘ 100%** | Bắt chuẩn xác các tên riêng tiếng Anh trong các đoạn văn dài |
| **Tên riêng quốc tế ngắn ($< 300$t)** | **30,000** | Lấy ngẫu nhiên đều | Tăng cường phản xạ dịch tên riêng tiếng Anh / Latinh chuẩn xác |
| **TỔNG CỘNG** | **283,729** | 🎯 **Tối ưu toàn diện** | **Đầy đủ câu ngắn, câu dài, thuật ngữ Hán-Việt & Tên riêng quốc tế** |
