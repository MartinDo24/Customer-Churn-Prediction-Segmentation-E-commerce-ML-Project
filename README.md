# churn-prediction-ml

# Dự đoán và phân nhóm khách hàng rời bỏ bằng Machine Learning

Dự án sử dụng dữ liệu bán hàng để xây dựng mô hình dự đoán khách hàng rời bỏ (churn) và phân nhóm khách hàng đã rời đi nhằm đưa ra chiến lược giữ chân phù hợp.

## 🎯 Mục tiêu
- Dự đoán khả năng khách hàng sẽ rời bỏ (churn)
- Phân tích các yếu tố ảnh hưởng đến churn
- Phân nhóm khách hàng churn bằng unsupervised learning (KMeans)

## ⚙️ Các bước thực hiện
1. Làm sạch và xử lý dữ liệu: xử lý giá trị thiếu, mã hóa, chuẩn hóa
2. Phân tích dữ liệu (EDA): biểu đồ phân phối, heatmap, boxplot, correlation
3. Huấn luyện mô hình Random Forest để dự đoán churn
4. Tối ưu mô hình bằng GridSearchCV, đánh giá bằng F1-score (~82%)
5. Phân cụm khách hàng đã churn bằng KMeans (3 nhóm)
6. Trực quan hóa và đưa ra đề xuất cải thiện

## 🧰 Công cụ sử dụng
- Python, pandas, numpy, seaborn, matplotlib  
- scikit-learn (RandomForest, GridSearchCV, KMeans)

Dataset:'churn_prediction'
