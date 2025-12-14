import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuración de la página
st.set_page_config(
    page_title="Detección de Ataques de Red",
    page_icon="🛡️",
    layout="wide"
)

# Título principal
st.title("🛡️ Sistema de Detección de Ataques de Red")
st.markdown("### Proyecto de Ciberseguridad con Machine Learning")
st.markdown("**Dataset:** NSL-KDD | **Modelo:** Random Forest Classifier")
st.markdown("---")

st.info("🚀 **Versión Demo en vivo** – El modelo completo fue entrenado con ~96.87% accuracy. "
        "La predicción manual y análisis están activos. ¡El código completo está en GitHub!")

# Sidebar
st.sidebar.header("📊 Información del Proyecto")
st.sidebar.markdown("""
**Métricas del modelo (entrenado):**
- Accuracy: ~96.87%
- Precision: ~95.98%
- Recall: ~97.21%
- F1-Score: ~96.59%

**Categorías detectadas:**
- Normal ✓
- DoS, Probe, R2L, U2R 🚨
""")
st.sidebar.markdown("**Autor:** [Leslie Jimenez](https://www.linkedin.com/in/leslie-jimenez-navarrete-a4670a1ba/)")
st.sidebar.markdown("**GitHub:** [Ver código fuente](https://github.com/DomiAndi/Detecci-n-Ataques-de-Red)")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predicción Manual", "📈 Análisis del Modelo", "📚 Acerca del Proyecto"])

# TAB 1: Predicción Manual (simulada con lógica realista)
with tab1:
    st.header("Predicción Manual de Conexiones")
    st.markdown("Ingresa características de una conexión de red y obtén una predicción simulada basada en el comportamiento del modelo entrenado.")

    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.number_input("Duration (segundos)", min_value=0, value=0)
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "telnet", "other"])
    with col2:
        src_bytes = st.number_input("Source Bytes", min_value=0, value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)
        count = st.number_input("Count (conexiones en 2s)", min_value=0, value=1)
    with col3:
        serror_rate = st.slider("SError Rate", 0.0, 1.0, 0.0)
        rerror_rate = st.slider("RError Rate", 0.0, 1.0, 0.0)
        flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "SH", "OTH"])

    if st.button("🔍 Analizar Conexión", type="primary"):
        # Lógica simplificada basada en patrones reales del NSL-KDD (alta src_bytes, bajo count = normal; alto count, errores = ataque)
        score_ataque = 0
        if src_bytes > 10000 or dst_bytes > 10000: score_ataque -= 0.3
        if count > 50: score_ataque += 0.6
        if serror_rate > 0.1 or rerror_rate > 0.1: score_ataque += 0.4
        if flag in ["REJ", "RSTR", "SH"]: score_ataque += 0.3
        
        proba_normal = max(0.05, min(0.95, 1 - (score_ataque + 0.2)))
        proba_ataque = 1 - proba_normal
        prediction = 1 if proba_ataque > 0.5 else 0

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            if prediction == 0:
                st.success("### ✅ CONEXIÓN NORMAL")
            else:
                st.error("### 🚨 ATAQUE DETECTADO")
        with col_res2:
            st.metric("Probabilidad Normal", f"{proba_normal*100:.1f}%")
            st.metric("Probabilidad Ataque", f"{proba_ataque*100:.1f}%")
        with col_res3:
            confidence = max(proba_normal, proba_ataque) * 100
            st.metric("Confianza", f"{confidence:.1f}%")

# TAB 2: Análisis del Modelo (estático pero profesional)
with tab2:
    st.header("Análisis del Rendimiento del Modelo")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Métricas")
        metrics_df = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Valor": ["96.87%", "95.98%", "97.21%", "96.59%"]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Matriz de Confusión (Ejemplo)")
        cm = [[9632, 368], [267, 10277]]
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Normal', 'Ataque'], y=['Normal', 'Ataque'],
                                       colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size":20}))
        fig.update_layout(title='Matriz de Confusión', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔍 Top Features Importantes")
    importance_df = pd.DataFrame({
        "Feature": ["src_bytes", "dst_bytes", "count", "service", "flag", "serror_rate", "same_srv_rate"],
        "Importance": [0.25, 0.18, 0.15, 0.10, 0.08, 0.07, 0.06]
    })
    fig_bar = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Top 7 Features")
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3: Acerca del Proyecto
with tab3:
    st.header("📚 Acerca del Proyecto")
    st.markdown("""
    ### 🎯 Objetivo
    Detectar intrusiones en redes en tiempo real usando Machine Learning.

    ### 🛠️ Tecnologías
    Python • Scikit-learn • Streamlit • Plotly

    ### 👨‍💻 Desarrollador
    **Leslie Jimenez** – Data Science Junior especializada en Ciberseguridad & ML

    [LinkedIn](https://www.linkedin.com/in/leslie-jimenez-navarrete-a4670a1ba/) | [GitHub](https://github.com/DomiAndi)

    ¡Gracias por probar mi proyecto de portafolio! 🚀
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center'>🛡️ Desarrollado con ❤️ usando Streamlit | Demo en vivo</p>", unsafe_allow_html=True)
