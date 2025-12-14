import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
st.markdown("---")

# Cargar modelo y transformadores
@st.cache_resource
def load_model():
    model = pickle.load(open('data/processed/best_model.pkl', 'rb'))
    scaler = pickle.load(open('data/processed/scaler.pkl', 'rb'))
    encoders = pickle.load(open('data/processed/label_encoders.pkl', 'rb'))
    return model, scaler, encoders

try:
    model, scaler, encoders = load_model()
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Sidebar con información del proyecto
st.sidebar.header("📊 Información del Proyecto")
st.sidebar.markdown("""
**Dataset:** NSL-KDD

**Modelo:** Random Forest Classifier

**Métricas del modelo:**
- Accuracy: ~96.87%
- Precision: ~95.98%
- Recall: ~97.21%
- F1-Score: ~96.59%

**Categorías de ataques:**
- DoS (Denial of Service)
- Probe (Escaneo de red)
- R2L (Remote to Local)
- U2R (User to Root)
""")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "📈 Análisis del Modelo", "📚 Acerca del Proyecto"])

# TAB 1: Predicción
with tab1:
    st.header("Predicción de Conexiones de Red")
    
    # Subsección: Predicción con ejemplos reales
    st.subheader("🎯 Prueba con Ejemplos Reales del Dataset")
    st.markdown("Usa ejemplos completos del dataset de prueba con todas las features:")
    
    # Cargar datos de test
    try:
        X_test_full = pd.read_csv('data/processed/X_test.csv')
        y_test_full = pd.read_csv('data/processed/y_test.csv').values.ravel()
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("📘 Ejemplo: Conexión Normal", use_container_width=True):
                # Buscar un ejemplo normal del dataset
                normal_indices = np.where(y_test_full == 0)[0]
                normal_idx = normal_indices[0]
                example = X_test_full.iloc[normal_idx:normal_idx+1]
                
                prediction = model.predict(example)[0]
                proba = model.predict_proba(example)[0]
                
                st.markdown("---")
                st.write("**Características principales:**")
                st.write(f"- src_bytes: {example['src_bytes'].values[0]:.0f}")
                st.write(f"- dst_bytes: {example['dst_bytes'].values[0]:.0f}")
                st.write(f"- count: {example['count'].values[0]:.0f}")
                st.write(f"- serror_rate: {example['serror_rate'].values[0]:.2f}")
                
                st.markdown("**Resultado:**")
                if prediction == 0:
                    st.success(f"✅ **CONEXIÓN NORMAL** - Confianza: {proba[0]*100:.2f}%")
                else:
                    st.error(f"❌ **Falso Positivo** - Predicho como ataque con {proba[1]*100:.2f}%")
        
        with col_ex2:
            if st.button("🔴 Ejemplo: Ataque DoS/Probe", use_container_width=True):
                # Buscar un ataque del dataset
                attack_indices = np.where(y_test_full == 1)[0]
                attack_idx = attack_indices[10]  # Usar el índice 10 para variedad
                example = X_test_full.iloc[attack_idx:attack_idx+1]
                
                prediction = model.predict(example)[0]
                proba = model.predict_proba(example)[0]
                
                st.markdown("---")
                st.write("**Características principales:**")
                st.write(f"- src_bytes: {example['src_bytes'].values[0]:.0f}")
                st.write(f"- dst_bytes: {example['dst_bytes'].values[0]:.0f}")
                st.write(f"- count: {example['count'].values[0]:.0f}")
                st.write(f"- serror_rate: {example['serror_rate'].values[0]:.2f}")
                
                st.markdown("**Resultado:**")
                if prediction == 1:
                    st.error(f"🚨 **ATAQUE DETECTADO** - Confianza: {proba[1]*100:.2f}%")
                else:
                    st.warning(f"⚠️ **Falso Negativo** - No detectado ({proba[0]*100:.2f}% normal)")
        
        with col_ex3:
            if st.button("🔍 Ejemplo Aleatorio", use_container_width=True):
                # Ejemplo aleatorio
                random_idx = np.random.randint(0, len(X_test_full))
                example = X_test_full.iloc[random_idx:random_idx+1]
                real_label = y_test_full[random_idx]
                
                prediction = model.predict(example)[0]
                proba = model.predict_proba(example)[0]
                
                st.markdown("---")
                st.write(f"**Etiqueta real:** {'🚨 Ataque' if real_label == 1 else '✅ Normal'}")
                st.write(f"**Predicción:** {'🚨 Ataque' if prediction == 1 else '✅ Normal'}")
                st.write(f"**Confianza:** {max(proba)*100:.2f}%")
                
                st.write("**Características principales:**")
                st.write(f"- src_bytes: {example['src_bytes'].values[0]:.0f}")
                st.write(f"- dst_bytes: {example['dst_bytes'].values[0]:.0f}")
                st.write(f"- count: {example['count'].values[0]:.0f}")
                
                if prediction == real_label:
                    st.success("✅ Predicción correcta")
                else:
                    st.error("❌ Predicción incorrecta")
    
    except Exception as e:
        st.error(f"Error al cargar ejemplos: {e}")
    
    st.markdown("---")
    
    # Subsección: Predicción manual
    st.subheader("✍️ Predicción Manual (Entrada de Datos)")
    st.markdown("⚠️ **Nota:** Solo se ingresan 13 de 41 features. El resto usa valores por defecto.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Información de Conexión**")
        duration = st.number_input("Duration (segundos)", min_value=0, value=0)
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "telnet", "other"])
        flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "SH", "S1", "S2", "RSTOS0", "RSTO", "S3", "OTH"])
    
    with col2:
        st.markdown("**Volumen de Datos**")
        src_bytes = st.number_input("Source Bytes", min_value=0, value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)
        land = st.selectbox("Land", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0, value=0)
        urgent = st.number_input("Urgent", min_value=0, value=0)
    
    with col3:
        st.markdown("**Características de Conexión**")
        count = st.number_input("Count", min_value=0, value=0, help="Conexiones al mismo host en 2 segundos")
        srv_count = st.number_input("Srv Count", min_value=0, value=0)
        serror_rate = st.slider("SError Rate", 0.0, 1.0, 0.0)
        rerror_rate = st.slider("RError Rate", 0.0, 1.0, 0.0)
    
    # Botón de predicción manual
    if st.button("🔍 Analizar Conexión Manual", type="primary"):
        try:
            # Cargar las columnas originales del entrenamiento
            X_train_sample = pd.read_csv('data/processed/X_train.csv', nrows=1)
            
            # Crear DataFrame con TODAS las columnas en el MISMO ORDEN
            input_data = {}
            
            # Llenar con los valores ingresados por el usuario
            user_inputs = {
                'duration': duration,
                'protocol_type': protocol_type,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': land,
                'wrong_fragment': wrong_fragment,
                'urgent': urgent,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'rerror_rate': rerror_rate
            }
            
            # Valores por defecto para features no ingresadas
            default_values = {
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 1,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'srv_serror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': 255,
                'dst_host_srv_count': 255,
                'dst_host_same_srv_rate': 1.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 1.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0
            }
            
            # Crear DataFrame con el orden correcto de columnas
            for col in X_train_sample.columns:
                if col in user_inputs:
                    input_data[col] = user_inputs[col]
                elif col in default_values:
                    input_data[col] = default_values[col]
                else:
                    input_data[col] = 0
            
            df_input = pd.DataFrame([input_data])
            
            # Aplicar encoding a variables categóricas
            categorical_cols = ['protocol_type', 'service', 'flag']
            for col in categorical_cols:
                if col in encoders and col in df_input.columns:
                    try:
                        # Manejar valores no vistos durante entrenamiento
                        if df_input[col].values[0] not in encoders[col].classes_:
                            # Usar la clase más común como fallback
                            df_input[col] = encoders[col].classes_[0]
                        df_input[col] = encoders[col].transform(df_input[col])
                    except Exception as e:
                        st.warning(f"Advertencia en encoding de {col}: {e}")
                        df_input[col] = 0
            
            # Asegurar que todas las columnas son numéricas
            df_input = df_input.astype(float)
            
            # Verificar orden de columnas
            df_input = df_input[X_train_sample.columns]
            
            # Hacer predicción
            prediction = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            
            # Mostrar resultado
            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if prediction == 0:
                    st.success("### ✅ CONEXIÓN NORMAL")
                    st.metric("Clasificación", "Normal", delta="Seguro")
                else:
                    st.error("### 🚨 ATAQUE DETECTADO")
                    st.metric("Clasificación", "Ataque", delta="Peligro", delta_color="inverse")
            
            with col_res2:
                st.metric("Probabilidad Normal", f"{proba[0]*100:.2f}%")
                st.metric("Probabilidad Ataque", f"{proba[1]*100:.2f}%")
            
            with col_res3:
                confidence = max(proba) * 100
                st.metric("Confianza del Modelo", f"{confidence:.2f}%")
                
                if confidence > 90:
                    st.info("🎯 Alta confianza")
                elif confidence > 70:
                    st.warning("⚠️ Confianza moderada")
                else:
                    st.error("❗ Baja confianza - requiere revisión manual")
        
        except Exception as e:
            st.error(f"Error al hacer la predicción: {e}")

# TAB 2: Análisis del Modelo
with tab2:
    st.header("Análisis del Rendimiento del Modelo")
    
    # Cargar datos de test para mostrar métricas
    try:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Métricas Principales")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(X_test)
            
            metrics_data = {
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Valor': [
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred),
                    recall_score(y_test, y_pred),
                    f1_score(y_test, y_pred)
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics['Porcentaje'] = (df_metrics['Valor'] * 100).apply(lambda x: f"{x:.2f}%")
            df_metrics['Valor'] = df_metrics['Valor'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(df_metrics, hide_index=True, use_container_width=True)
            
            # Explicación de métricas
            st.markdown("""
            **Interpretación:**
            - **Accuracy**: % de predicciones correctas totales
            - **Precision**: De los predichos como ataque, % que sí lo eran
            - **Recall**: De los ataques reales, % que detectamos
            - **F1-Score**: Balance entre Precision y Recall
            """)
        
        with col2:
            st.subheader("🎯 Matriz de Confusión")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Ataque'],
                y=['Normal', 'Ataque'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                hovertemplate='Real: %{y}<br>Predicción: %{x}<br>Cantidad: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Matriz de Confusión',
                xaxis_title='Predicción',
                yaxis_title='Real',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Desglose de la matriz
            st.markdown(f"""
            **Resultados:**
            - ✅ True Negatives (TN): {cm[0,0]:,}
            - ❌ False Positives (FP): {cm[0,1]:,}
            - ❌ False Negatives (FN): {cm[1,0]:,}
            - ✅ True Positives (TP): {cm[1,1]:,}
            """)
        
        # Feature Importance
        st.markdown("---")
        st.subheader("🔍 Importancia de Features")
        
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Features Más Importantes',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla
        st.dataframe(
            feature_importance.reset_index(drop=True),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error al cargar datos de análisis: {e}")

# TAB 3: Acerca del Proyecto
with tab3:
    st.header("📚 Acerca del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo
        Desarrollar un sistema de detección de intrusiones en redes usando Machine Learning 
        para identificar conexiones maliciosas en tiempo real.
        
        ### 🗂️ Dataset
        **NSL-KDD** - Versión mejorada del dataset KDD Cup 1999
        - 125,973 registros de entrenamiento
        - 22,544 registros de prueba
        - 41 features por conexión
        - 5 categorías: Normal, DoS, Probe, R2L, U2R
        
        ### 🤖 Modelos Entrenados
        1. **Logistic Regression** (Baseline)
        2. **Decision Tree**
        3. **Random Forest** ⭐ (Mejor modelo)
        
        ### 📊 Pipeline del Proyecto
        1. Análisis Exploratorio de Datos (EDA)
        2. Preprocesamiento (Encoding, Normalización)
        3. Entrenamiento de Modelos
        4. Evaluación y Comparación
        5. Despliegue del Dashboard
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Tecnologías Utilizadas
        - **Python 3.x**
        - **Pandas & NumPy** - Manipulación de datos
        - **Scikit-learn** - Machine Learning
        - **Streamlit** - Dashboard interactivo
        - **Plotly** - Visualizaciones interactivas
        
        ### 📈 Resultados Clave
        - **Accuracy**: 96.87%
        - **Precision**: 95.98%
        - **Recall**: 97.21%
        - **F1-Score**: 96.59%
        
        ### 🔑 Features Más Importantes
        1. src_bytes - Bytes enviados
        2. dst_bytes - Bytes recibidos
        3. count - Conexiones en 2 segundos
        
        ### 👨‍💻 Desarrollador
        **[Leslie Jimenez]**  
        Proyecto de portafolio - Data Science Junior  
        Especialización: Ciberseguridad & ML
        
        ### 📞 Contacto
        - GitHub: [DomiAndi](https://github.com/DomiAndi)
        - LinkedIn: [leslie-jimenez-navarrete](https://linkedin.com/in/leslie-jimenez-navarrete-a4670a1ba/)
        - Email: tu-email@ejemplo.com
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Notas Importantes
    
    **Limitaciones:**
    - La predicción manual usa solo 13 de 41 features, el resto se rellena con valores por defecto
    - Para mejores resultados en producción, se necesitarían todas las features del tráfico de red
    - El modelo fue entrenado con datos del año 1999, patrones de ataques actuales pueden diferir
    
    **Mejoras Futuras:**
    - Integración con sistemas de monitoreo en tiempo real
    - Actualización del modelo con datasets más recientes
    - Detección de tipos específicos de ataques (no solo binario)
    - Implementación de técnicas de balanceo de clases avanzadas
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🛡️ Sistema de Detección de Ataques de Red | Desarrollado con ❤️ usando Python & Streamlit</p>
</div>
""", unsafe_allow_html=True)
