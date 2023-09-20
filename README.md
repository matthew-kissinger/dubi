# Dubi - Automatic Video Dubbing Prototype

## Introduction

Welcome to Dubi, a prototype that aims to automate video dubbing with a blend of machine learning algorithms and audio processing libraries. Developed in Python, this application leverages several APIs and packages to translate spoken language in videos and replace it with a translated voice-over.

## Features

- **Audio Extraction**: Extracts audio from video files of multiple formats (MP4, AVI, MOV).
- **Speech-to-Text**: Transcribes the extracted audio to text.
- **Translation**: Translates the transcribed text to a target language.
- **Text-to-Speech**: Converts the translated text back to speech.
- **Audio Merging**: Merges the new voice-over back into the original video.

## Requirements

- Python 3.11
- Flask
- ffmpeg
- OpenAI API
- Google Cloud Text-to-Speech and Translate API
- pydub
- yt_dlp

## Installation

1. Clone the repository
    ```
    git clone https://github.com/matthew-kissinger/dubi.git
    ```
2. Navigate to the project directory
    ```
    cd dubi
    ```
3. Install required packages
    ```
    pip install -r requirements.txt
    ```
4. Add your API keys for OpenAI and Google Cloud services as environment variables or directly in `app.py`.
5. Run the Flask application
    ```
    flask run
    ```

## Usage

1. Open your web browser and go to `http://localhost:5000/`.
2. Upload a video file or provide a YouTube URL.
3. Choose the target language for translation.
4. Click the 'Submit' button and wait for the magic to happen.
5. Download the processed video with the new voice-over.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, open an issue first to discuss the proposed changes.
