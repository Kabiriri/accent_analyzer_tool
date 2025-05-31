import os
import tempfile
import numpy as np
import streamlit as st
import librosa
import yt_dlp
import requests
from speechbrain.inference import EncoderClassifier
import whisper
import torch
import soundfile as sf
from typing import Tuple, Dict
import logging
import shutil

# This fixes some weird compatibility issues I ran into on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Basic setup stuff - these values work well from my testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000  # Standard rate that works best with our models
MIN_AUDIO_LENGTH = 2  # Need at least 2 seconds to get meaningful results
MODEL_CACHE_DIR = "./model_cache"  # Keep downloaded models here so we don't re-download


class AccentAnalyzer:
    def __init__(self):
        # Make sure we have somewhere to put the models
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        self.classifier = self._init_classifier()
        self.whisper_model = self._init_whisper()

    def _init_classifier(self):
        """Load up the accent classification model - this is the heavy lifting part"""
        try:
            return EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",  # This model works pretty well for accents
                savedir=os.path.join(MODEL_CACHE_DIR, "ecapa"),
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}  # Use GPU if we can
            )
        except Exception as e:
            logger.error(f"Classifier init failed: {str(e)}")
            raise RuntimeError("Accent classifier unavailable")

    def _init_whisper(self):
        """Get Whisper ready for speech-to-text conversion"""
        try:
            # Base model is a good balance of speed vs accuracy
            return whisper.load_model("base", download_root=MODEL_CACHE_DIR)
        except Exception as e:
            logger.error(f"Whisper init failed: {str(e)}")
            raise RuntimeError("Speech recognition unavailable")

    def analyze(self, audio_path: str) -> Tuple[str, float, str, str]:
        """Main analysis pipeline - takes audio file and returns all the results"""
        try:
            audio = self._validate_audio(audio_path)
            accent, confidence = self._classify_accent(audio)
            transcript = self._generate_transcript(audio_path)
            features = self._extract_prosodic_features(audio)
            summary = self._generate_summary(accent, confidence, features)
            return accent, confidence, transcript, summary
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    def _validate_audio(self, audio_path: str) -> np.ndarray:
        """Check that the audio file is actually usable before we try to analyze it"""
        try:
            # Basic file existence check
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Load the audio and resample to our standard rate
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            # Make sure we actually got something
            if audio is None or len(audio) == 0:
                raise ValueError("Failed to load audio data")

            # Need at least a couple seconds to get meaningful accent analysis
            if len(audio) < SAMPLE_RATE * MIN_AUDIO_LENGTH:
                raise ValueError(f"Audio too short for analysis (minimum {MIN_AUDIO_LENGTH} seconds required)")

            logger.info(f"Audio loaded successfully: {len(audio) / SAMPLE_RATE:.2f} seconds")
            return audio

        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            raise

    def _classify_accent(self, audio: np.ndarray) -> Tuple[str, float]:
        """This is where the magic happens - actually classify the accent"""
        # Create a temp file that we can pass to the classifier
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Need to close it so we can write to it

        try:
            # Make sure we're working with a simple 1D array
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Normalize to prevent clipping - learned this the hard way
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95

            # Write out the audio file in a format the classifier likes
            sf.write(temp_path, audio, SAMPLE_RATE, subtype='PCM_16')

            # Double-check we actually created a valid file
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("Failed to create valid temporary audio file")

            logger.info(f"Created temp audio file: {temp_path} ({os.path.getsize(temp_path)} bytes)")

            # The classifier is picky about path format, especially on Windows
            abs_temp_path = os.path.abspath(temp_path)
            # Forward slashes work better with SpeechBrain
            normalized_path = abs_temp_path.replace('\\', '/')

            logger.info(f"Using normalized path: {normalized_path}")

            # Finally, run the actual classification
            prediction = self.classifier.classify_file(normalized_path)

            label = prediction[3][0]
            confidence = float(prediction[1][0]) * 100

            # Map the model outputs to more readable accent names
            accent_mapping = {
                "American": ["American", "US", "USA"],
                "British": ["British", "UK", "England"],
                "Australian": ["Australian", "Australia"],
                "Indian": ["Indian", "India"],
            }

            detected_accent = "Non-native"  # Default fallback
            for accent, keywords in accent_mapping.items():
                if any(keyword.lower() in label.lower() for keyword in keywords):
                    detected_accent = accent
                    break

            return detected_accent, confidence

        except Exception as e:
            logger.error(f"Accent classification error: {str(e)}")
            logger.error(f"Audio shape: {audio.shape}, Sample rate: {SAMPLE_RATE}")
            logger.error(f"Temp path: {temp_path}")
            raise RuntimeError(f"Accent classification failed: {str(e)}")
        finally:
            # Always clean up temp files, even if something goes wrong
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Temp file cleanup warning: {cleanup_error}")

    def _generate_transcript(self, audio_path: str) -> str:
        """Convert speech to text using Whisper - pretty straightforward"""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract pitch and timing features that can help with accent analysis"""
        try:
            # Get pitch information using librosa's pitch detection
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, sr=SAMPLE_RATE
            )
            voiced = f0[voiced_flag]  # Only look at parts where there's actually voice

            # Figure out speech rate by detecting speech segments
            intervals = librosa.effects.split(audio, top_db=30)
            speech_duration = sum([(end - start) / SAMPLE_RATE for start, end in intervals])
            total_duration = len(audio) / SAMPLE_RATE

            return {
                "pitch_mean": float(np.nanmean(voiced)) if len(voiced) > 0 else 0,
                "pitch_std": float(np.nanstd(voiced)) if len(voiced) > 0 else 0,
                "speech_rate": float(speech_duration / total_duration) if total_duration > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Feature extraction warning: {str(e)}")
            # Return safe defaults if feature extraction fails
            return {"pitch_mean": 0, "pitch_std": 0, "speech_rate": 0}

    def _generate_summary(self, accent: str, confidence: float, features: Dict) -> str:
        """Put together a human-readable explanation of what we found"""
        # Some basic explanations for different accent types
        explanations = {
            "American": "Rhotic /r/ pronunciation, flatter intonation patterns.",
            "British": "Non-rhotic /r/, more varied pitch modulation.",
            "Australian": "Higher average pitch, distinctive vowel shifts.",
            "Indian": "Syllable-timed rhythm, retroflex consonants.",
            "Non-native": "May show influence from speaker's first language."
        }

        # Convert confidence to something more readable
        confidence_level = (
            "high" if confidence > 85 else "moderate" if confidence > 60 else "low"
        )

        # Add some pitch-based observations
        feature_notes = []
        if features["pitch_mean"] > 125:
            feature_notes.append("Higher pitch suggests American/Australian English.")
        elif features["pitch_mean"] < 110:
            feature_notes.append("Lower pitch suggests British/Indian English.")

        return (
            f"{explanations.get(accent, 'Accent characteristics detected.')} {' '.join(feature_notes)} "
            f"(Classification made with {confidence_level} confidence)."
        )


def validate_url(url: str) -> bool:
    """Quick check to see if a URL looks valid and accessible"""
    try:
        if any(domain in url for domain in ["youtube.com", "youtu.be", "loom.com"]):
            return True
        elif url.endswith((".mp4", ".webm", ".wav", ".m4a")):
            # For direct file URLs, try to ping them
            return requests.head(url, timeout=5).status_code == 200
        return False
    except Exception:
        return False


def download_media(url: str) -> str:
    """Download audio from URL and return path to the downloaded file"""
    # Use temp directory to avoid cluttering up the main folder
    temp_dir = tempfile.mkdtemp(prefix='accent_download_')
    logger.info(f"Created temp directory: {temp_dir}")

    try:
        output_template = os.path.join(temp_dir, "audio.%(ext)s")

        # yt-dlp options that work well for our purposes
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',  # Get the best audio quality
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # Convert everything to WAV for consistency
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ar', str(SAMPLE_RATE),  # Standardize sample rate
                '-ac', '1',  # Convert to mono
            ]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading from: {url}")
            ydl.download([url])

        # Find what we downloaded
        downloaded_files = []
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                downloaded_files.append(file_path)
                logger.info(f"Found file: {file}")

        # Prefer WAV files since that's what we asked for
        for file_path in downloaded_files:
            if file_path.endswith('.wav'):
                logger.info(f"Using WAV file: {file_path}")
                return file_path

        # If no WAV, use any audio file we can find
        audio_extensions = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac']
        for file_path in downloaded_files:
            if any(file_path.lower().endswith(ext) for ext in audio_extensions):
                logger.info(f"Using audio file: {file_path}")
                return file_path

        # If we get here, something went wrong
        all_files = os.listdir(temp_dir)
        logger.error(f"No audio file found. Available files: {all_files}")
        raise RuntimeError(f"No audio file found after download. Files in temp dir: {all_files}")

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        # Clean up on failure
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise RuntimeError(f"Media download failed: {str(e)}")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Accent Evaluation Tool",
        layout="centered",

    )
    st.title(" Professional Accent Analysis Tool")

    # Some basic CSS to make things look nicer
    st.markdown("""
        <style>
            .stAlert { border-radius: 10px }
            .stButton>button { width: 100% }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **How to use:**
    1. Enter a URL to a public video (YouTube, Loom) or audio file (preferably 30 seconds or less for faster processing)
    2. Click "Analyze Accent" to process the audio
    3. View the detected accent type, confidence score, and analysis summary

    **Note:** First run may take longer as models are downloaded (~1GB total).
    """)

    # Test button to help users debug issues
    if st.button("Test System", help="Test if all components are working"):
        with st.spinner("Testing system components..."):
            try:
                st.write("Testing model initialization...")
                analyzer = AccentAnalyzer()

                st.write("Testing with synthetic audio...")
                # Create a test audio signal that's long enough
                duration = 3
                t = np.linspace(0, duration, SAMPLE_RATE * duration)
                # Make it sound somewhat speech-like with multiple frequencies
                test_audio = (np.sin(2 * np.pi * 440 * t) +
                              0.5 * np.sin(2 * np.pi * 880 * t) +
                              0.3 * np.sin(2 * np.pi * 220 * t))
                # Keep it from clipping
                test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8

                # Create temp file for testing
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    test_path = temp_file.name

                try:
                    sf.write(test_path, test_audio, SAMPLE_RATE, subtype='PCM_16')

                    # Make sure the file was actually created
                    if not os.path.exists(test_path):
                        raise RuntimeError("Failed to create test audio file")

                    file_size = os.path.getsize(test_path)
                    st.write(f" Test file created: {file_size} bytes")

                    # Try running classification on our test audio
                    normalized_path = os.path.abspath(test_path).replace('\\', '/')
                    result = analyzer.classifier.classify_file(normalized_path)

                    st.success(" All system components working correctly!")
                    st.info(
                        f"Test classification result: {result[3][0]} (confidence: {float(result[1][0]) * 100:.1f}%)")

                finally:
                    # Always clean up test files
                    if os.path.exists(test_path):
                        os.remove(test_path)

            except Exception as e:
                st.error(f" System test failed: {str(e)}")

                # Give users some debugging info
                with st.expander("Detailed Error Information"):
                    st.write("**Error Details:**")
                    st.code(str(e))

                    st.write("**System Information:**")
                    try:
                        import platform
                        st.code(f"""
Operating System: {platform.system()} {platform.release()}
Python Version: {platform.python_version()}
PyTorch Available: {'Yes' if 'torch' in globals() else 'No'}
CUDA Available: {torch.cuda.is_available() if 'torch' in globals() else 'N/A'}
Working Directory: {os.getcwd()}
Temp Directory: {tempfile.gettempdir()}
User Permissions: {'Admin' if os.name == 'nt' and os.getenv('USERNAME') == 'ADMIN' else 'Standard'}
                        """)
                    except Exception as info_error:
                        st.write(f"Could not gather system info: {info_error}")

                    st.write("**Troubleshooting Steps:**")
                    st.write("""
                    1. **File Permissions**: Try running the app as administrator (Windows)
                    2. **Antivirus**: Temporarily disable antivirus and try again
                    3. **Path Issues**: Make sure no special characters in your folder path
                    4. **Dependencies**: Reinstall speechbrain: `pip install --upgrade speechbrain`
                    5. **FFmpeg**: Ensure FFmpeg is properly installed and in PATH
                    6. **Disk Space**: Check you have enough space for temporary files
                    """)

    st.divider()

    # Main input area
    url = st.text_input(
        "Enter public video URL (YouTube, Loom, or direct audio file):",
        placeholder="https://www.youtube.com/watch?v=...",
        key="media_url"
    )

    if st.button("Analyze Accent", type="primary"):
        if not url:
            st.error("Please enter a valid URL")
            st.stop()

        if not validate_url(url):
            st.error("Invalid or inaccessible media URL")
            st.stop()

        with st.spinner("Downloading and processing audio... This may take a moment."):
            try:
                # Load models first to catch any issues early
                analyzer = AccentAnalyzer()

                # Download the media
                audio_path = download_media(url)

                # Run the analysis
                accent, confidence, transcript, summary = analyzer.analyze(audio_path)

                # Show results
                st.success("Analysis complete!")
                st.balloons()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Accent", accent)
                with col2:
                    st.metric("Confidence Score", f"{confidence:.1f}%")

                st.subheader("Analysis Summary")
                st.info(summary)

                with st.expander("View Full Transcript"):
                    st.write(transcript)

                # Clean up downloaded files
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        # Remove temp directory if it's empty
                        temp_dir = os.path.dirname(audio_path)
                        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                            os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup warning: {cleanup_error}")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.exception("Processing error")

                with st.expander("Troubleshooting"):
                    st.write("""
                    **Common issues:**
                    - Ensure the URL is publicly accessible
                    - Video should contain clear speech (not music-only)
                    - Audio should be at least 2 seconds long  
                    - Check your internet connection
                    - Make sure ffmpeg is installed on your system

                    **For Windows users:**
                    - Try running as administrator if you get file permission errors
                    - Ensure your antivirus isn't blocking temporary file creation
                    - Check that your path doesn't contain special characters
                    """)


if __name__ == "__main__":
    main()