import streamlit as st
import os
import io
import tempfile
import uuid
import time
from pydub import AudioSegment
import PyPDF2
import torch
import gdown
import requests
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="PDF to Custom Voice Converter",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("📄 PDF to Custom Voice Converter 🎙️")
st.markdown("""
Convert your PDF documents to speech and clone the voice using your own voice sample!
1. Upload a PDF document
2. Provide your voice sample (optional, for voice cloning)
3. Convert to audio with your custom voice
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Select Language", ["english", "chinese", "japanese", "spanish"])
    speed = st.slider("Speech Speed", 0.5, 2.0, 1.0)
    st.divider()
    st.info("For best voice cloning results, use a clear voice sample of at least 5 seconds.")

# Initialize session state
if 'voice_samples' not in st.session_state:
    st.session_state.voice_samples = {}
if 'openvoice_initialized' not in st.session_state:
    st.session_state.openvoice_initialized = False
if 'models_downloaded' not in st.session_state:
    st.session_state.models_downloaded = False

# Function to download models
def download_models():
    model_dir = Path("checkpoints")
    model_dir.mkdir(exist_ok=True)
    
    model_urls = {
        "checkpoint_v1.pth": "https://drive.google.com/uc?id=1Z3Xz3Iu2RUMC0F_0aJQ2e7MF-ff0z5gW",
        "checkpoint_v2.pth": "https://drive.google.com/uc?id=1jX7Wp7ecjNxCmHJ4QM4I1JkA33IytcQH",
        "config.json": "https://drive.google.com/uc?id=1C0g3f6Q1qQ5R6qQ2qQ3qQ4qQ5qQ6qQ7qQ"  # Example config
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (filename, url) in enumerate(model_urls.items()):
        if not (model_dir / filename).exists():
            status_text.text(f"Downloading {filename}...")
            try:
                gdown.download(url, str(model_dir / filename), quiet=False)
            except Exception as e:
                st.error(f"Failed to download {filename}: {str(e)}")
                return False
        progress_bar.progress((i + 1) / len(model_urls))
    
    status_text.text("Models downloaded successfully!")
    st.session_state.models_downloaded = True
    return True

# Function to initialize OpenVoice
def initialize_openvoice():
    try:
        # Try to import OpenVoice
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Initialize models
        base_speaker_tts = BaseSpeakerTTS('checkpoints/config.json', device=device)
        base_speaker_tts.load_ckpt('checkpoints/checkpoint.pth')
        
        tone_color_converter = ToneColorConverter('checkpoints/config.json', device=device)
        tone_color_converter.load_ckpt('checkpoints/checkpoint.pth')
        
        st.session_state.openvoice_initialized = True
        st.session_state.base_speaker_tts = base_speaker_tts
        st.session_state.tone_color_converter = tone_color_converter
        st.session_state.se_extractor = se_extractor
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize OpenVoice: {str(e)}")
        return False

# Check and download models if needed
if not st.session_state.models_downloaded:
    st.warning("Required models not found. Please download them first.")
    if st.button("Download Models"):
        with st.spinner("Downloading models... This may take a few minutes."):
            if download_models():
                st.success("Models downloaded successfully!")
            else:
                st.error("Failed to download models.")

# Initialize OpenVoice if models are downloaded but not initialized
if st.session_state.models_downloaded and not st.session_state.openvoice_initialized:
    with st.spinner("Initializing OpenVoice..."):
        if initialize_openvoice():
            st.success("OpenVoice initialized successfully!")
        else:
            st.error("Failed to initialize OpenVoice.")

# Main functionality
if st.session_state.openvoice_initialized:
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Convert PDF", "Manage Voice Samples", "How to Use"])
    
    with tab1:
        st.header("Convert PDF to Speech")
        
        # PDF upload
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
        
        # Voice sample selection
        voice_options = ["Default Voice"] + list(st.session_state.voice_samples.keys())
        selected_voice = st.selectbox("Select Voice", voice_options)
        
        if pdf_file and st.button("Convert to Speech"):
            with st.spinner("Converting PDF to speech..."):
                try:
                    # Read PDF content
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                    
                    # Extract text from PDF
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    
                    if not text.strip():
                        st.error("No text could be extracted from the PDF.")
                    else:
                        # Create temporary directory for outputs
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            # Generate TTS audio from text
                            base_audio_path = os.path.join(tmp_dir, "base_audio.wav")
                            st.session_state.base_speaker_tts.tts(text, base_audio_path, language=language)
                            
                            # Apply voice cloning if a voice sample is selected
                            if selected_voice != "Default Voice":
                                # Get reference embedding
                                reference_info = st.session_state.voice_samples[selected_voice]
                                target_se = reference_info["embedding"]
                                
                                # Apply voice cloning
                                output_path = os.path.join(tmp_dir, "cloned_audio.wav")
                                src_se = st.session_state.se_extractor.get_se(
                                    base_audio_path, 
                                    st.session_state.tone_color_converter
                                )
                                
                                st.session_state.tone_color_converter.convert(
                                    audio_src_path=base_audio_path,
                                    src_se=src_se,
                                    tts_se=target_se,
                                    output_path=output_path,
                                )
                                
                                result_audio = output_path
                                message = "PDF converted to audio with voice cloning"
                            else:
                                result_audio = base_audio_path
                                message = "PDF converted to audio with default voice"
                            
                            # Load and play the audio
                            audio = AudioSegment.from_file(result_audio)
                            st.audio(result_audio, format="audio/wav")
                            
                            # Download button
                            with open(result_audio, "rb") as f:
                                st.download_button(
                                    label="Download Audio",
                                    data=f,
                                    file_name="converted_audio.wav",
                                    mime="audio/wav"
                                )
                            
                            st.success(message)
                            
                except Exception as e:
                    st.error(f"Error converting PDF: {str(e)}")
    
    with tab2:
        st.header("Manage Voice Samples")
        
        # Upload voice sample
        voice_file = st.file_uploader("Upload Voice Sample", type=["wav", "mp3"], key="voice_upload")
        voice_name = st.text_input("Voice Sample Name")
        
        if voice_file and voice_name and st.button("Save Voice Sample"):
            with st.spinner("Processing voice sample..."):
                try:
                    # Generate unique ID for this voice sample
                    voice_id = str(uuid.uuid4())
                    
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        content = voice_file.read()
                        tmp_file.write(content)
                        tmp_path = tmp_file.name
                    
                    # Process the audio file
                    audio = AudioSegment.from_file(tmp_path)
                    # Convert to mono, 24kHz sampling rate (required by OpenVoice)
                    audio = audio.set_channels(1).set_frame_rate(24000)
                    
                    # Save processed audio
                    processed_path = f"processed_{voice_id}.wav"
                    audio.export(processed_path, format="wav")
                    
                    # Extract speaker embedding
                    speaker_embedding = st.session_state.se_extractor.get_se(
                        processed_path, 
                        st.session_state.tone_color_converter
                    )
                    
                    # Store reference
                    st.session_state.voice_samples[voice_name] = {
                        "embedding": speaker_embedding,
                        "audio_path": processed_path
                    }
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    st.success(f"Voice sample '{voice_name}' saved successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing voice sample: {str(e)}")
        
        # Display saved voice samples
        if st.session_state.voice_samples:
            st.subheader("Saved Voice Samples")
            for name, info in st.session_state.voice_samples.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.audio(info["audio_path"], format="audio/wav")
                with col3:
                    if st.button("Delete", key=f"delete_{name}"):
                        if os.path.exists(info["audio_path"]):
                            os.remove(info["audio_path"])
                        del st.session_state.voice_samples[name]
                        st.rerun()
        else:
            st.info("No voice samples saved yet.")
    
    with tab3:
        st.header("How to Use This App")
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Download Models**: First, click the 'Download Models' button in the sidebar to download the required AI models.
        
        2. **Add Voice Samples (Optional)**: 
           - Go to the 'Manage Voice Samples' tab
           - Upload a clear audio sample of your voice (WAV or MP3)
           - Give it a name and save it
        
        3. **Convert PDF to Speech**:
           - Go to the 'Convert PDF' tab
           - Upload your PDF document
           - Select a voice (default or your custom voice)
           - Click 'Convert to Speech'
           - Download or listen to the generated audio
        
        ### Tips for Best Results:
        - Use PDFs with selectable text (not scanned images)
        - For voice cloning, use a clear voice sample of at least 5 seconds
        - Speak clearly in your voice sample without background noise
        """)

else:
    st.info("Please download the required models and initialize OpenVoice to use this app.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>PDF to Custom Voice Converter | Built with Streamlit and OpenVoice</p>
</div>
""", unsafe_allow_html=True)
