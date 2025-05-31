# Professional Accent Analyzer

A powerful AI-powered web application that analyzes and identifies accents from audio and video content. Built with Streamlit and advanced machine learning models.

## Features

- **Multi-Platform Support**: Analyze audio from YouTube, Loom, and direct audio file URLs
- **Accent Detection**: Identifies American, British, Australian, Indian, and Non-native accents
- **Confidence Scoring**: Provides accuracy confidence levels for each prediction
- **Speech Transcription**: Full text transcription using OpenAI Whisper
- **Prosodic Analysis**: Analyzes pitch patterns, speech rate, and intonation
- **Real-time Processing**: Fast analysis with detailed explanations
- **User-Friendly Interface**: Clean, intuitive web interface

## Live Demo

Try the app: https://accentanalyzertool.streamlit.app/

## How to Use

1. **Enter a URL**: Paste a link to a YouTube video, Loom recording, or direct audio file
2. **Click Analyze**: The app will download and process the audio
3. **View Results**: Get accent type, confidence score, transcript, and detailed analysis

### Best Practices

- **Audio Length**: Use clips of 30 seconds or less for optimal performance
- **Audio Quality**: Clear speech works best (avoid background music/noise)
- **Supported Formats**: MP4, WAV, M4A, WebM, MP3
- **Public URLs**: Ensure your links are publicly accessible

## Technical Details

### Models Used
- **Accent Classification**: SpeechBrain ECAPA-VOXCELEB (speaker recognition model)
- **Speech Recognition**: OpenAI Whisper Base model
- **Audio Processing**: LibROSA for feature extraction

### Supported Accents
- American English
- British English  
- Australian English
- Indian English
- Non-native English

## Local Installation

### Prerequisites
- Python 3.8+ (preferably 3.10)
- FFmpeg installed on your system

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Kabiriri/accent_analyzer_tool.git
cd accent_analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### System Requirements
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for model downloads
- **Internet**: Required for first-time model downloads

## Dependencies

```
streamlit==1.34.0
openai-whisper==20231117
torch==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu118
torchaudio==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu118
librosa==0.10.1
yt-dlp==2023.12.30
requests==2.31.0
soundfile==0.12.1
numpy==1.26.3
scipy==1.11.4
huggingface-hub==0.20.2
speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop
sentencepiece==0.1.99
protobuf==3.20.3  # Required for SpeechBrain
```

## Testing

### Test the System
1. Click the "Test System" button in the app
2. Verify all components are working properly
3. Check model loading and audio processing


## How It Works

1. **Audio Download**: Uses yt-dlp to extract audio from video platforms
2. **Preprocessing**: Converts audio to 16kHz mono WAV format
3. **Feature Extraction**: Analyzes prosodic features (pitch, rhythm, timing)
4. **Classification**: ECAPA model generates accent embeddings
5. **Transcription**: Whisper converts speech to text
6. **Results**: Combines all analysis into readable summary

## Limitations

- **Processing Time**: First run slower due to model downloads (~1GB)
- **Audio Length**: Long files may timeout on free hosting
- **Accuracy**: Results depend on audio quality and speaker clarity
- **Languages**: Currently optimized for English accents only

## Troubleshooting

### Common Issues

**"System test failed"**
```bash
# Check FFmpeg installation
ffmpeg -version

# Reinstall dependencies
pip install --upgrade speechbrain torch
```

**"Download failed"**
- Verify URL is publicly accessible
- Check internet connection
- Try shorter audio clips

**"Audio too short"**
- Use audio clips longer than 2 seconds
- Ensure audio contains speech (not just music)

**"Classification failed"**
- Check available disk space
- Verify write permissions in temp directory
- Try running as administrator (Windows)

## Privacy & Security

- **No Data Storage**: Audio files are processed and immediately deleted
- **Temporary Files**: All processing uses temporary directories
- **No User Tracking**: No personal information collected
- **Public URLs Only**: App cannot access private/authenticated content

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
git clone https://github.com/Kabiriri/accent_analyzer_tool.git
cd accent_analyzer
pip install -r requirements.txt
streamlit run app.py
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- **SpeechBrain Team** for the ECAPA-VOXCELEB model
- **OpenAI** for the Whisper speech recognition model
- **Streamlit** for the amazing web app framework
- **LibROSA** for audio processing capabilities

## Support

Having issues? Please check the troubleshooting section or open an issue on GitHub.

---

**Note**: This application requires significant computational resources. For production use, consider upgrading to a paid hosting plan with more memory and processing power.

## Deployment Status

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://accentanalyzertool.streamlit.app/)

**Version**: 1.0.0  
**Last Updated**: 5/31/2025 
**Status**: Active
