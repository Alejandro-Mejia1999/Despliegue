import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time

# Configuración de la página
st.set_page_config(page_title="TLS'UNAH - Traductor de Lenguaje de Señas", layout="wide")

# --- NUEVOS ELEMENTOS DE DISEÑO ---
# Título personalizado con CSS
st.markdown("""
<style>
.titulo-principal {
    font-size: 3em;
    font-weight: bold;
    color: #2ECC71; /* Color verde lima */
    text-align: center;
    margin-bottom: 20px;
}
.subtitulo-proyecto {
    font-size: 1.2em;
    text-align: center;
    color: #a0a0a0;
}
</style>
<div class="titulo-principal">TLS-UNAH: Traductor de Lenguaje de Señas</div>
<div class="subtitulo-proyecto">Proyecto de reconocimiento de lenguaje de señas</div>
""", unsafe_allow_html=True)

# Expansor para las instrucciones, manteniendo la interfaz limpia
with st.expander("📝 Instrucciones de Uso"):
    st.markdown("""
        Esta aplicación traduce señas del lenguaje de señas hondureño a texto. Puedes usarla de las siguientes maneras:
        - **Imagen:** Sube una imagen de una seña para que sea detectada y transcrita.
        - **Video:** Sube un video y la aplicación procesará cada fotograma para transcribir las señas detectadas.
        - **Cámara en vivo:** Usa la cámara de tu dispositivo para la detección en tiempo real.
        
        **Importante:** La cámara en vivo solo funciona si ejecutas la aplicación en tu máquina local.
    """)
# ------------------------------------

# Cargar el modelo (con manejo de errores)
@st.cache_resource
def load_model():
    try:
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
        return modelo
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.error("Versión TF: " + tf.__version__)
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
    draw.rectangle([x1, y1, x2, y2], outline="#2ECC71", width=3)
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
        st.error(f"Error al procesar la imagen: {e}")
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

# Usar columnas con proporciones fijas (por ejemplo, 40/60)
col1, col2 = st.columns([4, 6])

# Área de transcripción
with col2:
    st.header("Transcripción")
    
    transcription_display = st.empty()
    
    def show_transcription():
        transcription_display.markdown(
            f"""
            <div style="
                border: 1px solid #ccc;
                padding: 10px;
                height: 300px;
                overflow-y: auto;
                background-color: #333333; /* Color de fondo un poco más oscuro */
                border-radius: 5px;
                white-space: pre-wrap;
            ">{st.session_state.transcription}</div>
            """,
            unsafe_allow_html=True
        )
    
    show_transcription()
    
    if st.button("Limpiar transcripción", key="clear_button"):
        st.session_state.transcription = ""
        st.session_state.last_char = ""
        show_transcription()

# Sección de entrada
with col1:
    st.header("Entrada")
    
    # --- AGRUPACIÓN CON CONTENEDOR ---
    with st.container(border=True):
        option = st.radio("Seleccione el tipo de entrada:", 
                         ('Imagen', 'Video', 'Cámara en vivo'), key="input_type")

        if option == 'Imagen':
            uploaded_file = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'], key="image_uploader")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = image.resize((280, 280))
                image_with_box = dibujar_recuadro_deteccion(image.copy())
                
                st.image(image_with_box, caption='Imagen cargada', use_container_width=True)
                
                with st.spinner('Procesando imagen...'):
                    prediccion, confianza = procesar_sena(np.array(image.resize((224, 224))))
                
                if prediccion:
                    st.success(f"Letra detectada: {prediccion} (Confianza: {confianza:.2%})")
                    if prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                else:
                    st.warning("No se detectó una seña clara en la imagen")

        elif option == 'Video':
            uploaded_file = st.file_uploader("Subir video", type=['mp4', 'avi', 'mov'], key="video_uploader")
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                st.session_state.transcription = ""
                st.session_state.last_char = ""
                show_transcription()
                
                stop_button = st.button('Detener procesamiento', key="stop_video")
                fps = cap.get(cv2.CAP_PROP_FPS)
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
                    
                    stframe.image(img_with_box, caption='Procesando video...', use_container_width=True)
                    
                    frame_for_model = cv2.resize(frame_rgb, (224, 224))
                    prediccion, _ = procesar_sena(frame_for_model)
                    
                    if prediccion and prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                    
                    processing_time = time.time() - start_time
                    sleep_time = max(0, frame_delay - processing_time)
                    time.sleep(sleep_time)
                
                cap.release()
                os.unlink(tfile.name)

        elif option == 'Cámara en vivo':
            st.warning("La cámara en vivo requiere acceso al hardware. Esta función funciona mejor cuando la aplicación se ejecuta localmente.")
            
            if st.button('Iniciar/Detener Cámara', key="toggle_camera"):
                st.session_state.camera_active = not st.session_state.camera_active
                if st.session_state.camera_active:
                    st.session_state.cap = cv2.VideoCapture(0)
                    st.session_state.transcription = ""
                    st.session_state.last_char = ""
                    show_transcription()
                else:
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
            
            if st.session_state.camera_active and st.session_state.cap is not None:
                FRAME_WINDOW = st.empty()
                stop_button = st.button('Detener Procesamiento', key="stop_camera")
                
                while st.session_state.camera_active and not stop_button and st.session_state.cap is not None:
                    start_time = time.time()
                    
                    ret, frame = st.session_state.cap.read()
                    if not ret:
                        st.error("Error al capturar frame de la cámara")
                        break
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_display = cv2.resize(frame_rgb, (280, 280))
                    img_pil = Image.fromarray(frame_display)
                    img_with_box = dibujar_recuadro_deteccion(img_pil)
                    
                    FRAME_WINDOW.image(img_with_box, use_container_width=True)
                    
                    frame_for_model = cv2.resize(frame_rgb, (224, 224))
                    prediccion, _ = procesar_sena(frame_for_model)
                    
                    if prediccion and prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                    
                    processing_time = time.time() - start_time
                    time.sleep(max(0, 0.1 - processing_time))
                
                if stop_button:
                    st.session_state.camera_active = False
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
    # ------------------------------------

# Notas al pie
st.markdown("---")
st.markdown("<p style='text-align: center; color: #808080;'>© 2024 TLS UNAH-IS. Todos los derechos reservados.</p>", unsafe_allow_html=True)


# Limpieza al cerrar
def cleanup():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

import atexit
atexit.register(cleanup)
