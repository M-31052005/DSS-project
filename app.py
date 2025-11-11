from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = 'dss-diet-secret-key-2024-vietnam'

# Load models
def load_models():
    try:
        with open('linear_model.pkl', 'rb') as f:
            linear_model = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return linear_model, rf_model, scaler
    except FileNotFoundError:
        return None, None, None

linear_model, rf_model, scaler = load_models()

# Chế độ ăn recommendations
DIET_RECOMMENDATIONS = {
    'low_calorie': {
        'name': 'Chế độ ăn giảm cân',
        'description': 'Phù hợp cho người muốn giảm cân, hạn chế calories',
        'foods': ['Rau xanh', 'Trái cây', 'Thịt gà không da', 'Cá', 'Sữa không đường'],
        'avoid': ['Đồ chiên rán', 'Đồ ngọt', 'Nước ngọt có gas']
    },
    'balanced': {
        'name': 'Chế độ ăn cân bằng',
        'description': 'Phù hợp cho người muốn duy trì sức khỏe',
        'foods': ['Ngũ cốc nguyên hạt', 'Thịt nạc', 'Cá', 'Rau củ', 'Trái cây', 'Sữa'],
        'avoid': ['Fast food', 'Đồ ăn chế biến sẵn']
    },
    'high_protein': {
        'name': 'Chế độ ăn tăng cơ',
        'description': 'Phù hợp cho người tập gym, muốn tăng cơ',
        'foods': ['Thịt bò', 'Thịt gà', 'Trứng', 'Cá hồi', 'Sữa tươi', 'Đậu nành'],
        'avoid': ['Đồ ăn nhiều đường', 'Thức ăn nhanh']
    },
    'low_carb': {
        'name': 'Chế độ ăn Keto/Low Carb',
        'description': 'Phù hợp cho người muốn giảm mỡ nhanh',
        'foods': ['Thịt', 'Cá', 'Trứng', 'Bơ', 'Dầu oliu', 'Rau xanh'],
        'avoid': ['Gạo', 'Bánh mì', 'Mì', 'Khoai tây', 'Đồ ngọt']
    },
    'heart_healthy': {
        'name': 'Chế độ ăn tốt cho tim mạch',
        'description': 'Phù hợp cho người có vấn đề về tim mạch',
        'foods': ['Cá hồi', 'Yến mạch', 'Hạnh nhân', 'Quả bơ', 'Dầu oliu'],
        'avoid': ['Thịt đỏ', 'Bơ sữa', 'Thức ăn chiên']
    }
}

def calculate_bmi(weight, height):
    """Tính BMI"""
    height_m = height / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    """Phân loại BMI"""
    if bmi < 18.5:
        return 'Thiếu cân'
    elif 18.5 <= bmi < 23:
        return 'Bình thường'
    elif 23 <= bmi < 25:
        return 'Thừa cân'
    elif 25 <= bmi < 30:
        return 'Béo phì độ I'
    else:
        return 'Béo phì độ II'

def recommend_diet(age, gender, weight, height, activity_level, goal):
    """Gợi ý chế độ ăn dựa trên thông tin người dùng"""
    bmi = calculate_bmi(weight, height)
    
    # Logic gợi ý
    if goal == 'giam_can' or bmi >= 25:
        return 'low_calorie'
    elif goal == 'tang_can':
        return 'high_protein'
    elif goal == 'duy_tri':
        if bmi < 18.5:
            return 'high_protein'
        return 'balanced'
    elif age > 40:
        return 'heart_healthy'
    else:
        return 'balanced'

def calculate_calories(age, gender, weight, height, activity_level):
    """Tính lượng calories cần thiết"""
    # BMR calculation (Mifflin-St Jeor)
    if gender == 'Nam':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Activity multiplier
    activity_multipliers = {
        'it_hoat_dong': 1.2,
        'hoat_dong_nhe': 1.375,
        'hoat_dong_trung_binh': 1.55,
        'hoat_dong_nang': 1.725,
        'hoat_dong_rat_nang': 1.9
    }
    
    tdee = bmr * activity_multipliers.get(activity_level, 1.55)
    return int(tdee)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    models_available = linear_model is not None and rf_model is not None
    return render_template('predict.html', models_available=models_available)

@app.route('/result')
def result_page():
    if 'analysis_result' not in session:
        return redirect(url_for('predict_page'))
    return render_template('result.html')

@app.route('/api/get_result', methods=['GET'])
def get_result():
    """API endpoint để lấy kết quả từ session"""
    if 'analysis_result' in session:
        return jsonify(session['analysis_result'])
    return jsonify({'error': 'No data found'}), 404

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Lấy dữ liệu từ form
        data = request.json
        print("Received data:", data)  # Debug
        
        name = data.get('ho_ten') or data.get('name')
        age = int(data.get('tuoi') or data.get('age'))
        gender = data.get('gioi_tinh') or data.get('gender')
        weight = float(data.get('can_nang') or data.get('weight'))
        height = float(data.get('chieu_cao') or data.get('height'))
        activity_level = data.get('hoat_dong') or data.get('activity_level')
        goal = data.get('muc_tieu') or data.get('goal')
        
        # Tính toán các chỉ số
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        calories = calculate_calories(age, gender, weight, height, activity_level)
        
        # Gợi ý chế độ ăn
        diet_type = recommend_diet(age, gender, weight, height, activity_level, goal)
        diet_info = DIET_RECOMMENDATIONS[diet_type]
        
        # Dự đoán bằng ML models (nếu có)
        ml_predictions = {}
        if linear_model is not None and rf_model is not None:
            # Tạo features cho model
            gender_encoded = 1 if gender == 'Nam' else 0
            activity_map = {
                'it_hoat_dong': 1,
                'hoat_dong_nhe': 2,
                'hoat_dong_trung_binh': 3,
                'hoat_dong_nang': 4,
                'hoat_dong_rat_nang': 5
            }
            activity_encoded = activity_map.get(activity_level, 3)
            
            features = np.array([[age, gender_encoded, weight, height, bmi, activity_encoded]])
            
            if scaler is not None:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            linear_pred = linear_model.predict(features_scaled)[0]
            rf_pred = rf_model.predict(features_scaled)[0]
            
            ml_predictions = {
                'linear_regression': round(linear_pred, 2),
                'random_forest': round(rf_pred, 2),
                'average': round((linear_pred + rf_pred) / 2, 2)
            }
      
        # Kết quả
        result = {
            'success': True,
            'user_info': {
                'name': name,
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height
            },
            'health_metrics': {
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'daily_calories': calories
            },
            'diet_recommendation': {
                'type': diet_type,
                'name': diet_info['name'],
                'description': diet_info['description'],
                'recommended_foods': diet_info['foods'],
                'avoid_foods': diet_info['avoid']
            },
            'ml_predictions': ml_predictions
        }
        
        # Lưu vào session
        session['analysis_result'] = result
        print("Saved to session:", result)  # Debug
        
        return jsonify(result)
        
    except Exception as e:
        print("Error:", str(e))  # Debug
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)