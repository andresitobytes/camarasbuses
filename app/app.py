import streamlit as st
import tempfile
import os

# Estilo general
st.set_page_config(page_title="Análisis de Video", layout="centered", page_icon="🚌")
st.markdown("""
    <style>
    .stApp { background-color: #e0f7fa; }
    .main-title {
        text-align: center; color: #004d40;
        font-size: 3em; font-weight: bold;
        margin-top: 1em; margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center; color: #007B8F;
        font-size: 2em; margin-bottom: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🚌 MovilIA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Análisis de Videos</div>', unsafe_allow_html=True)

# Subir video
video_file = st.file_uploader("🎥 Sube tu video aquí", type=["mp4", "avi", "mov"])

# Opciones de procesamiento disponibles
opciones_procesamiento = {
    "Detección de ingresos": "enterairport",
    "Detección de pagos": "payment",
    "Puerta abierta": "open_door",
    "Uso de teléfono": "phone"
}

# Mostrar opciones si hay video
if video_file:
    seleccion = st.selectbox("📌 Elige el análisis que deseas aplicar:", list(opciones_procesamiento.keys()))

    if st.button("🚀 Procesar video"):
        # Guardar video temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(video_file.read())
            video_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            output_path = tmp_out.name

        st.success("✅ Video subido exitosamente. Procesando...")

        try:
            # Importar módulo dinámicamente
            modulo = __import__(f"scripts.{opciones_procesamiento[seleccion]}", fromlist=["procesar_video"])
            procesar_funcion = getattr(modulo, "procesar_video")
        except (ImportError, AttributeError) as e:
            st.error(f"❌ No se pudo cargar el análisis seleccionado: {e}")
            st.stop()

        with st.spinner("⏳ Procesando video, por favor espera..."):
            try:
                resultados = procesar_funcion(video_path, output_path)
            except Exception as e:
                st.error(f"❌ Error al procesar el video: {e}")
                st.stop()

        # Mostrar video procesado
        st.video(resultados["output_path"])

        with open(resultados["output_path"], "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="📥 Descargar video procesado",
            data=video_bytes,
            file_name="video_procesado.mp4",
            mime="video/mp4"
        )

        # Mostrar métricas si existen
        st.subheader("📊 Métricas obtenidas:")
        if 'ingresos' in resultados:
            st.write(f"- Número de ingresos: **{resultados['ingresos']}**")
        if 'pagos' in resultados:
            st.write(f"- Número de pagos: **{resultados['pagos']}**")
        if 'tiempo_puerta_abierta' in resultados:
            st.write(f"- Tiempo con puerta abierta: **{resultados['tiempo_puerta_abierta']} seg**")
        if 'intervalos_puerta' in resultados:
            st.write("- Intervalos de apertura de puerta:")
            for i, (inicio, fin) in enumerate(resultados['intervalos_puerta']):
                st.write(f"    {i+1}. {inicio}s → {fin}s")
        if 'tiempo_telefono' in resultados:
            st.write(f"- Tiempo usando el teléfono: **{resultados['tiempo_telefono']} seg**")
