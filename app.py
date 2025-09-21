import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import numpy as np 
from sklearn.preprocessing import MinMaxScaler


# Selección de idioma
LANG = st.sidebar.selectbox("🌐 Idioma", ["Español"], index=0)

# Diccionario de traducción simple
T = {
    "Español": {
        "Evaluación de riesgo": "Evaluación de riesgo",
        "Análisis de resultados": "Análisis de resultados",
        "Saber más": "Saber más"
    }
}

# Reemplazar claves del menú de navegación ( al final del codigo)


# Configuración general
st.set_page_config(page_title="Predicción cáncer colorrectal ", layout="wide")
# Cabecera con logo personalizado en el sidebar
st.sidebar.image("logo_tr_bachillerato_virolai_ai_cancer_colon.jpeg", width=190)
st.sidebar.markdown("### **Predicción Personalizada Cáncer Colorrectal**")

st.sidebar.markdown("---")
st.sidebar.title("🧬 Navegación")

# Cargar modelo y objetos serializados
@st.cache_resource
def load_model():
    model = joblib.load("model_rf.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_order = joblib.load("feature_order.pkl")
    explainer = joblib.load("explainer.pkl")
    return model, label_encoders, feature_order, explainer

model, label_encoders, feature_order, explainer = load_model()

# Codificación segura
def encode_input(value, col):
    le = label_encoders[col]
    if value not in le.classes_:
        st.error(f"Valor '{value}' no válido para '{col}'")
        st.stop()
    return le.transform([value])[0]

# Página: Predicción
def predict_page():
    st.title("🔍 Predicción personalizada")
    st.markdown("Introduce tus datos para obtener una predicción personalizada de riesgo de cáncer colorrectal.")

    with st.form(key="formulario_prediccion"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sexo = st.selectbox("Sexo", label_encoders['sexo'].classes_)
            edad = st.slider("Edad", 0, 100, 50)
            imc = st.number_input("IMC", 10.0, 50.0, 25.0)
        with col2:
            calorias = st.number_input("Calorías diarias", 500, 8000, 2000)
            alcohol = st.selectbox("Consumo de alcohol", label_encoders['alcohol'].classes_)
            alc_dur = st.slider("Duración alcohol (años)", 0, 100, 0)
        with col3:
            tabaco = st.selectbox("Consumo de tabaco", label_encoders['tabaco'].classes_)
            tab_dur = st.slider("Duración tabaco (años)", 0, 100, 0)
            familiares = st.selectbox("Antecedentes familiares", label_encoders['familiares'].classes_)

        submitted = st.form_submit_button("🔎 Predecir")

    if submitted:
        input_dict = {
            "sexo": encode_input(sexo, "sexo"),
            "edad": edad,
            "imc": imc,
            "calorias": calorias,
            "alcohol": encode_input(alcohol, "alcohol"),
            "alc.dur": alc_dur,
            "tabaco": encode_input(tabaco, "tabaco"),
            "tab.dur": tab_dur,
            "familiares": encode_input(familiares, "familiares")
        }

        input_df = pd.DataFrame([input_dict])[feature_order]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]  # probabilidad de tumor (1)

        st.markdown("---")
        st.subheader("🎯 Resultado de la predicción")

        if pred == 1:
            st.error(f"🟥 Alto riesgo de tumor colorrectal\n\nProbabilidad estimada: {prob:.2%}")
        else:
            st.success(f"🟩 Bajo riesgo de tumor colorrectal (control)\n\nProbabilidad estimada: {prob:.2%}")

        st.markdown("---")
        st.subheader("🧠 Interpretación del modelo")
        try:
            shap_values = explainer(input_df)

            if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
                shap_single = shap.Explanation(
                    values=shap_values.values[0, :, 1],
                    base_values=shap_values.base_values[0, 1],
                    data=shap_values.data[0],
                    feature_names=shap_values.feature_names
                )
            else:
                shap_single = shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_single, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error al mostrar SHAP: {e}")

                # Historial de predicciones almacenado en sesión
        if "historial" not in st.session_state:
            st.session_state.historial = []

        entrada_original = input_df.copy()
        entrada_original["predicción"] = "Tumor" if pred == 1 else "Control"
        entrada_original["probabilidad"] = f"{prob:.2%}"
        st.session_state.historial.append(entrada_original)

        st.markdown("---")
        st.subheader("📁 Historial de predicciones")
        historial_df = pd.concat(st.session_state.historial, ignore_index=True)
        st.dataframe(historial_df)

        csv = historial_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar historial en CSV", data=csv, file_name="historial_predicciones.csv", mime="text/csv")
    
                # --- Evaluación global del modelo ---
        st.markdown("---")
        st.subheader("📊 Evaluación global del modelo")

        from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

        # Cargar dataset original
        df_eval = pd.read_excel("colon.xlsx").dropna()

        # Codificar variables categóricas con los mismos encoders usados en el entrenamiento
        for col in label_encoders.keys():
            df_eval[col] = label_encoders[col].transform(df_eval[col])

        # Preparar X e y con el mismo orden de features
        X_eval = df_eval[feature_order]
        y_eval = (df_eval["tumor"] != "control").astype(int)

        # Predicciones
        y_pred_eval = model.predict(X_eval)
        y_pred_prob_eval = model.predict_proba(X_eval)[:, 1]

        # Métricas
        accuracy = accuracy_score(y_eval, y_pred_eval)
        recall = recall_score(y_eval, y_pred_eval)
        precision = precision_score(y_eval, y_pred_eval)
        cm = confusion_matrix(y_eval, y_pred_eval)
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP)
        auc = roc_auc_score(y_eval, y_pred_prob_eval)

        st.write(f"**Exactitud (Accuracy):** {accuracy:.2%}")
        st.write(f"**Sensibilidad (Recall):** {recall:.2%}")
        st.write(f"**Precisión (Precision):** {precision:.2%}")
        st.write(f"**Especificidad (Specificity):** {specificity:.2%}")
        st.write(f"**AUC-ROC:** {auc:.2f}")

        # Matriz de confusión visual
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Control", "Tumor"], 
                    yticklabels=["Control", "Tumor"], ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig_cm)



        
# Página: Análisis de datos reales
def insights_page():
    st.title("📈 Exploración de datos reales")
    df = pd.read_excel("colon.xlsx").dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de edades por tumor")
        fig_age = px.histogram(df, x="edad", color="tumor", nbins=30, barmode="overlay")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.subheader("Distribución de Índice de masa corporal (IMC)")
        fig_imc = px.histogram(df, x="imc", color="tumor", nbins=30, barmode="overlay")
        st.plotly_chart(fig_imc, use_container_width=True)

    st.subheader("Relación Calorías vs Índice de masa corporal (IMC)")
    fig_scatter = px.scatter(df, x="calorias", y="imc", color="tumor", trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Consumo de alcohol y tabaco")
    col3, col4 = st.columns(2)
    with col3:
        fig_alcohol = px.pie(df, names="alcohol", title="Alcohol")
        st.plotly_chart(fig_alcohol, use_container_width=True)
    with col4:
        fig_tabaco = px.pie(df, names="tabaco", title="Tabaco")
        st.plotly_chart(fig_tabaco, use_container_width=True)


    # Comparativa de medias por grupo (Control vs Tumor)
    st.subheader("📊 Comparativa de medias por gurpo de diagnóstico")

    numeric_cols = ["edad", "imc", "calorias", "alc.dur", "tab.dur"]
    df_scaled = df.copy()
    df_scaled["calorias"] = df_scaled["calorias"] / 100  # Escalado solo de calorías

    group_means = df_scaled.groupby("tumor")[numeric_cols].mean().T

    # Renombrar columnas según valores reales
    rename_dict = {}
    for col in group_means.columns:
        if col == 0:
            rename_dict[col] = "Control"
        elif col == 1:
            rename_dict[col] = "Tumor"
        else:
            rename_dict[col] = f"Otro_{col}"

    group_means = group_means.rename(columns=rename_dict)

 # Gráfico de barras

def insights_page():
    st.title("📈 Exploración de datos reales")

    # 1. Cargar dataset (se asegura de que df esté definido dentro de la función)
    df = pd.read_excel("colon.xlsx").dropna()

    # 2. Gráfico 1: Distribución de la edad por diagnóstico
    st.subheader("Edad según diagnóstico")
    fig_age = px.box(
        df, 
        x="tumor", 
        y="edad", 
        color="tumor", 
        title="Distribución de la edad según diagnóstico"
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # 3. Gráfico 2: Distribución del IMC por diagnóstico
    st.subheader("IMC según diagnóstico")
    fig_imc = px.box(
        df, 
        x="tumor", 
        y="imc", 
        color="tumor", 
        title="Distribución del IMC según diagnóstico"
    )
    st.plotly_chart(fig_imc, use_container_width=True)

    # 4. Gráfico 3: Relación calorías vs IMC
    st.subheader("Relación entre calorías e IMC")
    fig_scatter = px.scatter(
        df, 
        x="calorias", 
        y="imc", 
        color="tumor", 
        trendline="ols", 
        title="Relación entre calorías diarias e IMC"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 5. Gráfico 4: Consumo de tabaco y alcohol
    st.subheader("Consumo de tabaco y alcohol en la muestra")
    col1, col2 = st.columns(2)

    with col1:
        fig_tabaco = px.histogram(
            df, 
            x="tabaco", 
            color="tumor", 
            barmode="group", 
            title="Consumo de tabaco por diagnóstico"
        )
        st.plotly_chart(fig_tabaco, use_container_width=True)

    with col2:
        fig_alcohol = px.histogram(
            df, 
            x="alcohol", 
            color="tumor", 
            barmode="group", 
            title="Consumo de alcohol por diagnóstico"
        )
        st.plotly_chart(fig_alcohol, use_container_width=True)

    # Nota explicativa para el TR
    st.caption(
        "Estas gráficas muestran patrones claros en la distribución de variables clave. "
        "La edad y el IMC tienden a diferir entre controles y pacientes con tumor, "
        "mientras que el consumo de tabaco y alcohol aporta información complementaria sobre hábitos de riesgo."
    )






# Página: Información
def about_page():
    st.title("ℹ️ Saber más")
    st.markdown("""
     Esta herramienta ha sido desarrollada como parte de un proyecto de investigación
    para explorar cómo la **inteligencia artificial** puede contribuir a la detección
    precoz del cáncer colorrectal.

    El sistema se basa en un modelo de *machine learning* entrenado con datos clínicos
    estructurados, y permite generar una **predicción personalizada de riesgo** del 90% de precision aproximadamente..

    **Autor**: Pablo Villanueva (2025)  
    Proyecto personal de investigación en inteligencia artificial aplicada a la predicción del cáncer colorrectal.
    """)

# Navegación entre páginas
pages = {
    T[LANG]["Evaluación de riesgo"]: predict_page,
    T[LANG]["Análisis de resultados"]: insights_page,
    T[LANG]["Saber más"]: about_page
}

pagina = st.sidebar.radio("Por favor, seleccione una opción", list(pages.keys()))
pages[pagina]()
