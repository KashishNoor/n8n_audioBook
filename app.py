import os
import io
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
import PyPDF2
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

app = FastAPI(title="PDF to Custom Voice API")

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
base_speaker_tts = BaseSpeakerTTS('OpenVoice/checkpoints/config.json', device=device)
base_speaker_tts.load_ckpt('OpenVoice/checkpoints/checkpoint.pth')
tone_color_converter = ToneColorConverter('OpenVoice/checkpoints/config.json', device=device)
tone_color_converter.load_ckpt('OpenVoice/checkpoints/checkpoint.pth')

# This will store reference audios for voice cloning
reference_audios = {}

@app.get("/")
async def root():
    return {"message": "PDF to Custom Voice API"}

@app.post("/upload_voice_sample/")
async def upload_voice_sample(file: UploadFile = File(...)):
    """Upload a voice sample for cloning"""
    try:
        # Generate unique ID for this voice sample
        voice_id = str(uuid.uuid4())
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
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
        speaker_embedding = se_extractor.get_se(processed_path, tone_color_converter)
        
        # Store reference
        reference_audios[voice_id] = {
            "embedding": speaker_embedding,
            "audio_path": processed_path
        }
        
        # Clean up
        os.unlink(tmp_path)
        
        return JSONResponse(content={"voice_id": voice_id, "message": "Voice sample uploaded successfully"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice sample: {str(e)}")

@app.post("/convert_pdf/")
async def convert_pdf(
    pdf_file: UploadFile = File(...),
    voice_id: str = None,
    language: str = "english"
):
    """Convert PDF to speech, optionally with voice cloning"""
    try:
        # Read PDF content
        pdf_content = await pdf_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text from PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Generate TTS audio from text
            base_audio_path = os.path.join(tmp_dir, "base_audio.wav")
            base_speaker_tts.tts(text, base_audio_path, language=language)
            
            # If voice_id is provided, clone the voice
            if voice_id and voice_id in reference_audios:
                # Get reference embedding
                reference_info = reference_audios[voice_id]
                target_se = reference_info["embedding"]
                
                # Apply voice cloning
                output_path = os.path.join(tmp_dir, "cloned_audio.wav")
                tone_color_converter.convert(
                    audio_src_path=base_audio_path,
                    src_se=se_extractor.get_se(base_audio_path, tone_color_converter),
                    tts_se=target_se,
                    output_path=output_path,
                )
                
                result_audio = output_path
                message = "PDF converted to audio with voice cloning"
            else:
                result_audio = base_audio_path
                message = "PDF converted to audio with default voice"
            
            # Return the audio file
            return FileResponse(
                result_audio,
                media_type="audio/wav",
                filename="converted_audio.wav"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF: {str(e)}")

@app.get("/list_voices/")
async def list_voices():
    """List all available voice samples"""
    return {"voices": list(reference_audios.keys())}

@app.delete("/delete_voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice sample"""
    if voice_id in reference_audios:
        # Remove audio file
        if os.path.exists(reference_audios[voice_id]["audio_path"]):
            os.remove(reference_audios[voice_id]["audio_path"])
        # Remove from storage
        del reference_audios[voice_id]
        return {"message": f"Voice {voice_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Voice ID not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)