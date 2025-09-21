import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import shap
import joblib

# Evaluación detallada 
# --- Importaciones necesarias ---

# --- Cargar y preparar los datos ---
df = pd.read_excel("colon.xlsx").dropna()
df["tumor_bin"] = df["tumor"].apply(lambda x: 0 if x == "control" else 1)

# Codificar variables categóricas
categorical_cols = ["sexo", "alcohol", "tabaco", "familiares"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["grupo", "tumor", "extension", "tumor_bin"])
y = df["tumor_bin"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Entrenar modelo ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predicciones y métricas ---
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Exactitud
accuracy = accuracy_score(y_test, y_pred)

# Sensibilidad (recall)
recall = recall_score(y_test, y_pred)

# Precisión (precision)
precision = precision_score(y_test, y_pred)

# Matriz de confusión → especificidad
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)

# AUC
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Exactitud: {accuracy:.2%}")
print(f"Sensibilidad: {recall:.2%}")
print(f"Precisión: {precision:.2%}")
print(f"Especificidad: {specificity:.2%}")
print(f"AUC: {auc:.2f}")



# --- Cargar datos ---
df = pd.read_excel("colon.xlsx")
df = df.dropna()  # eliminar filas incompletas

# --- Crear variable binaria objetivo ---
df["tumor_bin"] = df["tumor"].apply(lambda x: 0 if x == "control" else 1)

# --- Codificar variables categóricas ---
categorical_cols = ["sexo", "alcohol", "tabaco", "familiares"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Variables a eliminar del modelo ---
drop_cols = ["grupo", "tumor", "extension"]  # no deben influir en la predicción

# --- Preparar X e y ---
X = df.drop(columns=drop_cols + ["tumor_bin"])
y = df["tumor_bin"]
feature_order = list(X.columns)

# --- División de datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenamiento del modelo ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluación ---
print(f"Precisión: {model.score(X_test, y_test):.2f}")
print(classification_report(y_test, model.predict(X_test)))

# --- SHAP explainer ---
explainer = shap.Explainer(model, X_train)

# --- Guardar todo ---
joblib.dump(model, "model_rf.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_order, "feature_order.pkl")
joblib.dump(explainer, "explainer.pkl")

print("✅ Modelo, codificadores y explainer guardados correctamente.")
