import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time

# Configuración de la página
st.set_page_config(
    page_title="TLS'UNAH - Traductor de Lenguaje de Señas", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS MEJORADO Y MODERNO ---
st.markdown("""
<style>
    /* Importar fuentes de Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS para colores */
    :root {
        --primary-color: #2ECC71;
        --secondary-color: #3498DB;
        --accent-color: #E74C3C;
        --dark-bg: #1a1a1a;
        --light-bg: #f8f9fa;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --border-radius: 12px;
        --shadow: 0 4px 20px rgba(0,0,0,0.1);
        --gradient: linear-gradient(135deg, #2ECC71 0%, #3498DB 100%);
    }
    
    /* Estilos generales */
    .main {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header principal mejorado */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .titulo-principal {
        font-size: 3.5em;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        animation: fadeInUp 1s ease-out;
    }
    
    .subtitulo-proyecto {
        font-size: 1.3em;
        text-align: center;
        color: rgba(255,255,255,0.9);
        margin-bottom: 5px;
        position: relative;
        z-index: 1;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .descripcion-proyecto {
        font-size: 1em;
        text-align: center;
        color: rgba(255,255,255,0.8);
        position: relative;
        z-index: 1;
        animation: fadeInUp 1s ease-out 0.4s both;
    }
    
    /* Animaciones */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Tarjetas mejoradas */
    .stContainer > div {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stContainer > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    /* Expansor de instrucciones mejorado */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white !important;
        border-radius: var(--border-radius);
        font-weight: 600;
        padding: 1rem;
        border: none !important;
    }
    
    .streamlit-expanderContent {
        background: white;
        border-radius: 0 0 var(--border-radius) var(--border-radius);
        border: 1px solid #e0e0e0;
        border-top: none;
        padding: 1.5rem;
    }
    
    /* Radio buttons mejorados */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: var(--primary-color);
        transform: translateY(-1px);
    }
    
    /* Botones mejorados */
    .stButton > button {
        background: var(--gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3) !important;
        animation: pulse 0.6s ease-in-out !important;
    }
    
    /* File uploader mejorado */
    .stFileUploader > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 2px dashed rgba(255,255,255,0.8) !important;
        border-radius: var(--border-radius) !important;
        padding: 2rem !important;
        text-align: center !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: white !important;
        transform: scale(1.02) !important;
    }
    
    /* Área de transcripción mejorada */
    .transcription-area {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        min-height: 350px;
        max-height: 450px;
        overflow-y: auto;
        color: white;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        line-height: 1.6;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.1);
        position: relative;
    }
    
    .transcription-area::before {
        content: '📝 Transcripción en tiempo real';
        position: absolute;
        top: -10px;
        left: 20px;
        background: var(--gradient);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        box-shadow: var(--shadow);
    }
    
    .transcription-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        flex-direction: column;
        opacity: 0.7;
    }
    
    .transcription-placeholder .emoji {
        font-size: 3em;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    /* Headers de sección */
    .section-header {
        background: var(--gradient);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.2em;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .section-header:hover::before {
        left: 100%;
    }
    
    /* Alertas y mensajes mejorados */
    .stAlert {
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow) !important;
        animation: slideIn 0.5s ease-out !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #2ECC71, #27AE60) !important;
        color: white !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #F39C12, #E67E22) !important;
        color: white !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #E74C3C, #C0392B) !important;
        color: white !important;
    }
    
    /* Spinner personalizado */
    .stSpinner > div {
        border-color: var(--primary-color) !important;
    }
    
    /* Footer mejorado */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        margin: 3rem -1rem -1rem -1rem;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .titulo-principal {
            font-size: 2.5em;
        }
        
        .subtitulo-proyecto {
            font-size: 1.1em;
        }
        
        .stContainer > div {
            padding: 1rem;
        }
    }
    
    /* Efectos de hover para imágenes */
    .stImage > img {
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }
    
    /* Indicador de confianza */
    .confidence-indicator {
        background: var(--gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: var(--shadow);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Mejoras para el contenedor principal */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header principal mejorado
st.markdown("""
<div class="header-container">
    <div class="titulo-principal">🤟 TLS-UNAH</div>
    <div class="subtitulo-proyecto">Traductor de Lenguaje de Señas Hondureño</div>
    <div class="descripcion-proyecto">Tecnología de reconocimiento de señas con IA avanzada</div>
</div>
""", unsafe_allow_html=True)

# Expansor para las instrucciones mejorado
with st.expander("📋 Guía de Uso - Instrucciones Detalladas"):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
        <h3 style="margin-top: 0; color: white;">🎯 ¿Cómo usar la aplicación?</h3>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white;">
            <h4 style="color: white; margin-top: 0;">📸 Modo Imagen</h4>
            <p>• Sube una imagen clara de una seña</p>
            <p>• La IA detectará automáticamente la letra</p>
            <p>• Verás el resultado con nivel de confianza</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: white;">
            <h4 style="color: white; margin-top: 0;">🎥 Modo Video</h4>
            <p>• Carga un video con señas</p>
            <p>• Procesamiento fotograma por fotograma</p>
            <p>• Transcripción automática completa</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%); padding: 1.5rem; border-radius: 12px; color: white;">
            <h4 style="color: white; margin-top: 0;">📹 Cámara en Vivo</h4>
            <p>• Detección en tiempo real</p>
            <p>• Funciona mejor en local</p>
            <p>• Transcripción instantánea</p>
        </div>
    </div>
    
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #3498DB;">
        <strong>💡 Consejos para mejores resultados:</strong><br>
        • Usa buena iluminación<br>
        • Mantén la mano centrada en el recuadro<br>
        • Evita fondos complejos<br>
        • Realiza las señas de forma clara y pausada
    </div>
    """, unsafe_allow_html=True)

# Cargar el modelo (con manejo de errores mejorado)
@st.cache_resource
def load_model():
    try:
        with st.spinner('🤖 Cargando modelo de IA...'):
            # Solución especial para el error de InputLayer
            custom_objects = {
                'InputLayer': lambda **kwargs: tf.keras.layers.InputLayer(
                    input_shape=kwargs['batch_shape'][1:] if 'batch_shape' in kwargs else (224, 224, 3),
                    dtype=kwargs.get('dtype', 'float32'),
                    name=kwargs.get('name', None)
                )
            }
            
            modelo = tf.keras.models.load_model(
                "modelo_mobilenetv21.h5",
                custom_objects=custom_objects,
                compile=False
            )
            st.success("✅ Modelo cargado exitosamente")
            return modelo
    except Exception as e:
        st.error(f"❌ Error crítico al cargar el modelo: {str(e)}")
        st.error(f"🔧 Versión TensorFlow: {tf.__version__}")
        st.stop()

modelo = load_model()

clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

# Funciones auxiliares optimizadas
def dibujar_recuadro_deteccion(image_pil):
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    box_size = int(min(width, height) * 0.7)
    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    # Recuadro con estilo más moderno
    draw.rectangle([x1, y1, x2, y2], outline="#2ECC71", width=4)
    # Esquinas decorativas
    corner_size = 20
    for corner in [(x1, y1), (x2-corner_size, y1), (x1, y2-corner_size), (x2-corner_size, y2-corner_size)]:
        draw.rectangle([corner[0], corner[1], corner[0]+corner_size, corner[1]+corner_size], 
                      outline="#E74C3C", width=2)
    return image_pil

def procesar_sena(imagen):
    try:
        img = imagen.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediccion_prob = modelo.predict(img, verbose=0)[0]
        
        max_prob = np.max(prediccion_prob)
        predicted_index = np.argmax(prediccion_prob)
        
        if max_prob >= 0.75:  # Umbral de confianza
            return clases[predicted_index], max_prob
        return None, None
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
        return None, None

# Inicializar variables de sesión
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'last_char' not in st.session_state:
    st.session_state.last_char = ""
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Layout principal con columnas mejoradas
col1, col2 = st.columns([5, 7], gap="large")

# Área de transcripción mejorada
with col2:
    st.markdown('<div class="section-header">📝 Panel de Transcripción</div>', unsafe_allow_html=True)
    
    transcription_display = st.empty()
    
    def show_transcription():
        if st.session_state.transcription:
            # Mostrar transcripción con estilo mejorado
            letters = st.session_state.transcription.strip().split()
            transcription_html = f"""
            <div class="transcription-area">
                <div style="padding-top: 2rem;">
                    <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 1rem;">
                        📊 Total de letras detectadas: {len(letters)}
                    </div>
                    <div style="font-size: 1.3em; line-height: 2; word-spacing: 10px;">
                        {' '.join([f'<span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 8px; margin: 2px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">{letter}</span>' for letter in letters if letter])}
                    </div>
                </div>
            </div>
            """
        else:
            transcription_html = """
            <div class="transcription-area">
                <div class="transcription-placeholder">
                    <div class="emoji">🤟</div>
                    <div style="font-size: 1.1em; font-weight: 500;">Esperando detección de señas...</div>
                    <div style="font-size: 0.9em; opacity: 0.7; margin-top: 0.5rem;">Las letras aparecerán aquí automáticamente</div>
                </div>
            </div>
            """
        
        transcription_display.markdown(transcription_html, unsafe_allow_html=True)
    
    show_transcription()
    
    # Botón de limpiar mejorado
    col_clear1, col_clear2, col_clear3 = st.columns([1, 2, 1])
    with col_clear2:
        if st.button("🗑️ Limpiar Transcripción", key="clear_button", use_container_width=True):
            st.session_state.transcription = ""
            st.session_state.last_char = ""
            show_transcription()
            st.success("✅ Transcripción limpiada")

# Sección de entrada mejorada
with col1:
    st.markdown('<div class="section-header">🎯 Panel de Control</div>', unsafe_allow_html=True)
    
    # Contenedor con borde mejorado
    with st.container(border=True):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem; text-align: center;">
            <h4 style="margin: 0; color: white;">🎮 Selecciona el Modo de Entrada</h4>
        </div>
        """, unsafe_allow_html=True)
        
        option = st.radio(
            "Elige una opción:", 
            ('📸 Imagen', '🎥 Video', '📹 Cámara en vivo'), 
            key="input_type",
            horizontal=True
        )

        if option == '📸 Imagen':
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; border-radius: 8px; color: white; margin: 1rem 0; text-align: center;">
                <h5 style="margin: 0; color: white;">📸 Modo Imagen Activado</h5>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "📁 Selecciona una imagen de seña", 
                type=['png', 'jpg', 'jpeg'], 
                key="image_uploader",
                help="Formatos soportados: PNG, JPG, JPEG"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = image.resize((280, 280))
                image_with_box = dibujar_recuadro_deteccion(image.copy())
                
                st.image(image_with_box, caption='🖼️ Imagen cargada - Lista para análisis', use_container_width=True)
                
                with st.spinner('🔍 Analizando imagen con IA...'):
                    time.sleep(1)  # Simular procesamiento
                    prediccion, confianza = procesar_sena(np.array(image.resize((224, 224))))
                
                if prediccion:
                    # Mostrar resultado con estilo mejorado
                    st.markdown(f"""
                    <div class="confidence-indicator">
                        ✅ Letra detectada: <strong>{prediccion}</strong> | 
                        🎯 Confianza: <strong>{confianza:.1%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                        st.balloons()  # Efecto visual de celebración
                else:
                    st.warning("⚠️ No se detectó una seña clara en la imagen. Intenta con mejor iluminación.")

        elif option == '🎥 Video':
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 8px; color: white; margin: 1rem 0; text-align: center;">
                <h5 style="margin: 0; color: white;">🎥 Modo Video Activado</h5>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "📹 Selecciona un video con señas", 
                type=['mp4', 'avi', 'mov'], 
                key="video_uploader",
                help="Formatos soportados: MP4, AVI, MOV"
            )
            
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                st.session_state.transcription = ""
                st.session_state.last_char = ""
                show_transcription()
                
                # Información del video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                st.info(f"📊 Video cargado: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s de duración")
                
                progress_bar = st.progress(0)
                stop_button = st.button('⏹️ Detener Procesamiento', key="stop_video")
                
                frame_count = 0
                frame_delay = 1.0 / fps if fps > 0 else 0.03
                
                while cap.isOpened() and not stop_button:
                    start_time = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_display = cv2.resize(frame_rgb, (280, 280))
                    img_pil = Image.fromarray(frame_display)
                    img_with_box = dibujar_recuadro_deteccion(img_pil)
                    
                    stframe.image(img_with_box, caption=f'🎬 Procesando frame {frame_count}/{total_frames}', use_container_width=True)
                    
                    frame_for_model = cv2.resize(frame_rgb, (224, 224))
                    prediccion, _ = procesar_sena(frame_for_model)
                    
                    if prediccion and prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                    
                    # Actualizar barra de progreso
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    frame_count += 1
                    processing_time = time.time() - start_time
                    sleep_time = max(0, frame_delay - processing_time)
                    time.sleep(sleep_time)
                
                cap.release()
                os.unlink(tfile.name)
                progress_bar.progress(1.0)
                st.success("✅ Video procesado completamente")

        elif option == '📹 Cámara en vivo':
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%); 
                        padding: 1rem; border-radius: 8px; color: white; margin: 1rem 0; text-align: center;">
                <h5 style="margin: 0; color: white;">📹 Modo Cámara en Vivo</h5>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ La cámara en vivo requiere acceso al hardware. Funciona mejor en ejecución local.")
            
            col_cam1, col_cam2 = st.columns(2)
            
            with col_cam1:
                if st.button('🎥 Iniciar Cámara', key="start_camera", use_container_width=True):
                    st.session_state.camera_active = True
                    st.session_state.cap = cv2.VideoCapture(0)
                    st.session_state.transcription = ""
                    st.session_state.last_char = ""
                    show_transcription()
                    st.success("✅ Cámara iniciada")
            
            with col_cam2:
                if st.button('⏹️ Detener Cámara', key="stop_camera", use_container_width=True):
                    st.session_state.camera_active = False
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    st.info("📴 Cámara detenida")
            
            if st.session_state.camera_active and st.session_state.cap is not None:
                FRAME_WINDOW = st.empty()
                status_placeholder = st.empty()
                
                frame_counter = 0
                
                while st.session_state.camera_active and st.session_state.cap is not None:
                    start_time = time.time()
                    
                    ret, frame = st.session_state.cap.read()
                    if not ret:
                        st.error("❌ Error al capturar frame de la cámara")
                        break
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_display = cv2.resize(frame_rgb, (280, 280))
                    img_pil = Image.fromarray(frame_display)
                    img_with_box = dibujar_recuadro_deteccion(img_pil)
                    
                    FRAME_WINDOW.image(img_with_box, caption=f'📹 Transmisión en vivo - Frame {frame_counter}', use_container_width=True)
                    
                    frame_for_model = cv2.resize(frame_rgb, (224, 224))
                    prediccion, confianza = procesar_sena(frame_for_model)
                    
                    if prediccion and prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                        
                        # Mostrar detección en tiempo real
                        status_placeholder.markdown(f"""
                        <div style="background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%); 
                                    padding: 0.5rem; border-radius: 8px; color: white; text-align: center; margin: 0.5rem 0;">
                            🎯 Detectado: <strong>{prediccion}</strong> ({confianza:.1%})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    frame_counter += 1
                    processing_time = time.time() - start_time
                    time.sleep(max(0, 0.1 - processing_time))
                    
                    # Auto-stop después de un tiempo para la demo
                    if frame_counter > 1000:  # Límite de frames
                        break

# Footer mejorado
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
        <div>
            <h4 style="margin: 0; color: white;">🏛️ TLS UNAH-IS</h4>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Universidad Nacional Autónoma de Honduras</p>
        </div>
        <div style="border-left: 2px solid rgba(255,255,255,0.3); padding-left: 2rem;">
            <p style="margin: 0; opacity: 0.8;">© 2024 Todos los derechos reservados</p>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.6;">Desarrollado con ❤️ y tecnología IA</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Limpieza al cerrar
def cleanup():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

import atexit
atexit.register(cleanup)
