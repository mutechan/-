import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os

# === 1. Загрузка данных ===
data_path = 'cardio_train.csv' 
df = pd.read_csv(data_path, sep=';')

# === 2. Предварительный анализ ===
print(df.info())
print(df.describe())

# === 3. Очистка данных ===
df.drop_duplicates(inplace=True)  # Убираем дубликаты
df = df.dropna()  # Убираем пропуски (если есть)

# === 4. Предобработка данных ===
X = df.drop(columns=['cardio'])  # Признаки
y = df['cardio']  # Целевая переменная

# === 5. Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 6. Модель ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 7. Матрица корреляций ===
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
os.makedirs('output', exist_ok=True)
plt.savefig('output/correlation_matrix.png')
plt.show()

# === 8. Значимость признаков ===
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot.bar(figsize=(10, 6))
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('output/feature_importances.png')
plt.show()

# === 9. Матрица ошибок ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=1)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('output/confusion_matrix.png')
plt.show()

# === 10. Отчет о модели ===
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
