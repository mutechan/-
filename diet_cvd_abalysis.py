import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
# Используем датасет Cardiovascular Disease с Kaggle
# Цель: изучить влияние диеты и других факторов на сердечно-сосудистые заболевания
df = pd.read_csv('cardio_train.csv', sep=';')

# 2. Предварительный анализ данных
print("Информация о данных:")
print(df.info())
print("\nПроверка на пропуски:")
print(df.isnull().sum())
print("\nОсновные статистики:")
print(df.describe())

# 3. Очистка данных
# Удаление выбросов на основе роста и веса
df = df[(df['height'] > 140) & (df['height'] < 200)]
df = df[(df['weight'] > 40) & (df['weight'] < 150)]

# 4. Предобработка данных
X = df.drop(columns=['cardio', 'id'])  # Признаки
y = df['cardio']  # Целевая переменная

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Модель и обучение
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Кросс-валидация
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Средняя точность кросс-валидации: {scores.mean():.2f}")

# 8. Оценка модели
y_pred = model.predict(X_test)
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred))

# 9. Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 10. Визуализация значимости признаков
feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['cardio', 'id']).columns)
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# 11. Корреляция признаков
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
