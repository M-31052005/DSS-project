#Import thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH CHẾ ĐỘ ĂN (DSS)")
print("Training Deep Learning Model...")
print("=" * 60)

# Load dataset
try:
    df = pd.read_csv('train_filtered.csv')
    print(f"\n✓ Đã load dataset: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"✓ Các cột: {list(df.columns)}")
except FileNotFoundError:
    print("\n❌ Không tìm thấy file train_filtered.csv!")
    exit(1)

# 2. Định nghĩa các đặc trưng (X) và mục tiêu (y)
# X: Thông tin người dùng (Đầu vào)
features = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
    'Workout_Frequency (days/week)', 'Experience_Level'
]
# y: Nhu cầu dinh dưỡng (Đầu ra)
targets = ['Calories', 'Proteins', 'Fats', 'Carbs']

X = df[features]
y = df[targets]

# 3. Phân chia tập Huấn luyện (Train) và Kiểm tra (Test)
# Chúng ta sẽ dùng 80% dữ liệu để huấn luyện và 20% để kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Xây dựng một "Pipeline" Tiền xử lý
# Mô hình không hiểu 'Gender' = 'Male' -> Cần OneHotEncoder
# Các cột số (Age, BMI) có thang đo khác nhau -> Cần StandardScaler

# Xác định cột nào là số, cột nào là danh mục
numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Workout_Frequency (days/week)', 'Experience_Level']
categorical_features = ['Gender']

# Tạo bộ tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), # Chuẩn hóa cột số
        ('cat', OneHotEncoder(), categorical_features) # Chuyển đổi cột danh mục
    ])

print("--- Quá trình chuẩn bị dữ liệu hoàn tất ---")
print(f"Kích thước tập Huấn luyện (X_train): {X_train.shape}")
print(f"Kích thước tập Kiểm tra (X_test): {X_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Mô hình 1: Hồi quy Tuyến tính (Linear Regression) ---

# Tạo pipeline hoàn chỉnh: 1. Tiền xử lý -> 2. Huấn luyện
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', MultiOutputRegressor(LinearRegression()))])

# Huấn luyện mô hình
print("Đang huấn luyện Hồi quy Tuyến tính...")
lr_model = lr_pipeline.fit(X_train, y_train)
print("Hoàn tất!")

# --- Mô hình 2: Rừng Ngẫu nhiên (Random Forest) ---
# Đây là mô hình cơ bản nhưng mạnh mẽ hơn Linear Regression

# Tạo pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
                              # n_estimators=100: Dùng 100 "cây"
                              # n_jobs=-1: Dùng tất cả CPU để chạy nhanh hơn

# Huấn luyện mô hình
print("\nĐang huấn luyện Random Forest...")
rf_model = rf_pipeline.fit(X_train, y_train)
print("Hoàn tất!")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Dự đoán trên tập X_test (dữ liệu mô hình chưa từng thấy)
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# 2. Tính toán các chỉ số lỗi
# R2 Score (càng gần 1 càng tốt)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# Mean Absolute Error (MAE) (càng nhỏ càng tốt)
# Đây chính là "sai số trung bình" mà bạn có thể giải thích
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"[Linear Regression]  R2 Score: {r2_lr:.4f} | Sai số tuyệt đối trung bình (MAE): {mae_lr:.2f}")
print(f"[Random Forest]      R2 Score: {r2_rf:.4f} | Sai số tuyệt đối trung bình (MAE): {mae_rf:.2f}")

# 3. Trực quan hóa Sai số (cho chỉ số Calories)

# Lấy cột Calories (cột 0) từ y_test và y_pred
y_test_calories = y_test['Calories']
y_pred_lr_calories = y_pred_lr[:, 0]
y_pred_rf_calories = y_pred_rf[:, 0]

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))

# Biểu đồ cho Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test_calories, y_pred_lr_calories, alpha=0.3, c='blue')
plt.plot([min(y_test_calories), max(y_test_calories)], [min(y_test_calories), max(y_test_calories)], '--', color='red', lw=2)
plt.title(f'Linear Regression (R2 = {r2_score(y_test_calories, y_pred_lr_calories):.3f})')
plt.xlabel('Lượng Calo Thực tế')
plt.ylabel('Lượng Calo Dự đoán')

# Biểu đồ cho Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test_calories, y_pred_rf_calories, alpha=0.3, c='green')
plt.plot([min(y_test_calories), max(y_test_calories)], [min(y_test_calories), max(y_test_calories)], '--', color='red', lw=2)
plt.title(f'Random Forest (R2 = {r2_score(y_test_calories, y_pred_rf_calories):.3f})')
plt.xlabel('Lượng Calo Thực tế')
plt.ylabel('Lượng Calo Dự đoán')

plt.suptitle('So sánh Lỗi Dự đoán Calo')
plt.tight_layout()
plt.show()