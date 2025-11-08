# 🩺 Diabetes Prediction | آیا دیابت داریم یا نه؟

این پروژه با هدف **پیش‌بینی ابتلا به دیابت** بر اساس ویژگی‌های پزشکی (مانند سن، BMI، فشار خون و غیره) طراحی شده است.  
در این پروژه از **مدل‌های یادگیری ماشین و شبکه عصبی مصنوعی (ANN)** برای مقایسه عملکرد در تشخیص دیابت استفاده شده است.

---

## 📘 مقدمه (Introduction)

دیابت یکی از شایع‌ترین بیماری‌های مزمن در جهان است و تشخیص زودهنگام آن نقش مهمی در پیشگیری از عوارض جدی دارد.  
هدف این پروژه ایجاد مدلی است که بتواند با استفاده از داده‌های پزشکی، احتمال ابتلا به دیابت را پیش‌بینی کند.

Diabetes is one of the most common chronic diseases worldwide, and early detection plays a crucial role in preventing severe complications.  
This project aims to build machine learning and neural network models to predict the likelihood of having diabetes.

---

## 📊 معرفی داده‌ها (Dataset Description)

- **منبع داده:** Kaggle – [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
- **تعداد نمونه‌ها:** 768  
- **ویژگی‌ها:** 8 ویژگی ورودی مانند:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age  
- **خروجی (Outcome):**
  - `0` → بدون دیابت  
  - `1` → مبتلا به دیابت  

---

## ⚙️ مراحل انجام پروژه (Steps)

### 1. آماده‌سازی داده‌ها (Data Preprocessing)
- حذف مقادیر گمشده (در صورت وجود)
- تقسیم داده به `X` (ویژگی‌ها) و `y` (برچسب‌ها)
- استانداردسازی داده‌ها با `StandardScaler`
- تقسیم به داده‌های آموزش و تست (80/20)

### 2. مدل‌های استفاده‌شده (Models Used)
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Artificial Neural Network (ANN)

---

## 🤖 شبکه عصبی مصنوعی (Artificial Neural Network)

مدل ANN با استفاده از **Keras/TensorFlow** ساخته شده است و شامل:
- 1 ورودی (با 8 نود برای ویژگی‌ها)
- 2 لایه‌ی مخفی با فعال‌ساز `ReLU`
- Dropout برای جلوگیری از overfitting  
- 1 لایه خروجی با فعال‌ساز `Sigmoid` برای پیش‌بینی دودویی

```python
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
