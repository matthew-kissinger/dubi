from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import ffmpeg
import openai
from google.cloud import texttospeech, translate_v2
from google.oauth2.service_account import Credentials
from pydub import AudioSegment
from pydub.silence import split_on_silence
import json
import tempfile
import shutil
import html
import requests
import yt_dlp

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', "uploads/")
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', "processed/")
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'mp4,avi,mov').split(','))
CHUNK_LENGTH = int(os.getenv('CHUNK_LENGTH', 10000))
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration for Eleven Labs TTS
ELEVEN_LABS_API_URL_TEMPLATE = os.getenv('ELEVEN_LABS_API_URL_TEMPLATE', "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}")
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
ELEVEN_LABS_HEADERS = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": ELEVEN_LABS_API_KEY
}

# Google Cloud TTS setup
key_data = json.loads(os.getenv('GOOGLE_CLOUD_KEY_DATA'))
credentials = Credentials.from_service_account_info(key_data)
translate_client = translate_v2.Client(credentials=credentials)

def eleven_labs_tts(text, language_code, voice_id):
    """
    Converts text to speech using Eleven Labs API and saves the result as a .mp3 file.
    Returns the path to the saved audio file.
    """
    # Use the voice_id to complete the API URL
    api_url = ELEVEN_LABS_API_URL_TEMPLATE.format(voice_id=voice_id)
    
    # For this example, we will use the "eleven_multilingual_v2" model. 
    # In real-world scenarios, you may want to select models based on requirements.
    model_id = "eleven_multilingual_v2"
    
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(api_url, json=data, headers=ELEVEN_LABS_HEADERS)
    if response.status_code != 200:
        # Handle error (for simplicity, raising an exception here)
        raise Exception(f"Error with Eleven Labs TTS: {response.content}")
    
    # Save the audio content to a file as MP3
    mp3_output_path = os.path.join(tempfile.mkdtemp(), "translated_chunk.mp3")
    with open(mp3_output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_output_path)
    wav_output_path = mp3_output_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_output_path, format="wav")

    return wav_output_path


def download_youtube_video(youtube_url):
    tmp_dir = tempfile.mkdtemp()
    save_path = os.path.join(tmp_dir, "downloaded.mp4")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': save_path,
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # Use yt_dlp.YoutubeDL
            ydl.download([youtube_url])
        return save_path
    except yt_dlp.utils.DownloadError as e:  # Catch yt_dlp's exception
        print(f"DownloadError: {e}")
        raise Exception("Failed to download YouTube video")
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        raise Exception("An unknown error occurred while downloading the video")

def get_video_duration(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return float(video_stream['duration']) * 1000  # Convert to milliseconds

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + ".wav"
    ffmpeg.input(video_path).output(audio_path, acodec="pcm_s16le", ac=1, ar="16k").run()
    return audio_path

def generate_chunks(audio_path):
    audio = AudioSegment.from_wav(audio_path)

    # Get chunks based on silence
    chunks = split_on_silence(audio, min_silence_len=800, silence_thresh=-40, keep_silence=500)

    chunks_with_times = []
    accumulated_time = 0
    for chunk in chunks:
        # Calculate start time of this chunk
        chunk_start_time = accumulated_time
        chunks_with_times.append((chunk, chunk_start_time))

        # Move the accumulated_time pointer to the end of this chunk
        accumulated_time += len(chunk)  # Adding 500ms because keep_silence is set to 500ms

    return chunks_with_times


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Handling upload request...")
    
    target_language = request.form.get('language')
    voice_id = request.form.get('voice')  # Retrieve voice_id from the form data
    video_filename = None
    
    # Handle YouTube URL
    youtube_url = request.form.get('url')
    if youtube_url:
        try:
            video_path = download_youtube_video(youtube_url)
            video_filename = os.path.basename(video_path)
            print(f"Downloaded YouTube video to: {video_path}")
        except Exception as e:
            print(f"Error downloading YouTube video: {e}")
            return jsonify({'error': 'Failed to download YouTube video'})
    
    # Handle File Upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            print("Error: No filename provided.")
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            video_filename = file.filename
            file.save(video_path)
            print(f"Saved video file to: {video_path}")
        else:
            print("Error: Invalid file type.")
            return jsonify({'error': 'Invalid file type'})
    
    else:
        print("Error: No file part in the request.")
        return jsonify({'error': 'No file part'})

    # Extract audio and chunk it
    audio_path = extract_audio_from_video(video_path)
    print(f"Extracted audio saved to: {audio_path}")
    chunk_data = generate_chunks(audio_path)
    print(f"Generated {len(chunk_data)} audio chunks.")

    translated_audio_paths = []

    for chunk, chunk_start_time in chunk_data:
        print(f"Chunk Start Time: {chunk_start_time}ms")

        chunk_path = os.path.join(tempfile.mkdtemp(), "chunk.wav")
        chunk.export(chunk_path, format="wav")
        print(f"Saved chunk to: {chunk_path}")

        with open(chunk_path, 'rb') as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file).get('text')
        print(f"Transcription: {transcription}")
        
        translation = translate_client.translate(transcription, target_language=target_language)['translatedText']
        translation = html.unescape(translation)  # Decode HTML entities
        print(f"Translation: {translation}")

        # Include voice_id when calling eleven_labs_tts
        translated_audio_path = eleven_labs_tts(translation, target_language, voice_id)
        print(f"Translated audio saved to: {translated_audio_path}")

        translated_audio_paths.append((translated_audio_path, chunk_start_time))

    video_duration = get_video_duration(video_path)
    print(f"Total video duration: {video_duration}ms")
    
    processed_video_path = os.path.join(PROCESSED_FOLDER, "processed_" + video_filename)
    print(f"Preparing to merge voice overs to video...")
    final_audio = AudioSegment.silent(duration=500)  # Start with a 0.5-second silent segment
    last_end_time = 500  # Start time is now 0.5 seconds later due to the initial silence

    for path, start_time in translated_audio_paths:
        audio_chunk = AudioSegment.from_wav(path)
        
        # Calculate the duration of silence needed before this chunk
        silence_duration = start_time - last_end_time
        silence = AudioSegment.silent(duration=silence_duration)
        
        # Append the silence and the audio chunk to final_audio
        final_audio += silence + audio_chunk
        
        # Update last_end_time
        last_end_time = start_time + len(audio_chunk)


    # Add any remaining silence to the end
    remaining_silence_duration = video_duration - len(final_audio)
    if remaining_silence_duration > 0:
        final_audio += AudioSegment.silent(duration=remaining_silence_duration)

    # Save the final audio to a temporary file
    final_audio_path = os.path.join(tempfile.mkdtemp(), "final_audio.mp3")
    final_audio.export(final_audio_path, format="mp3")

    # Merge the final audio with the video using ffmpeg
    ffmpeg_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(final_audio_path)
    ffmpeg_output = ffmpeg.output(ffmpeg_input.video, audio_input.audio, processed_video_path, vcodec='copy', acodec='aac')
    ffmpeg_output.run()

    print(f"Processed video saved to: {processed_video_path}")

    return render_template('index.html', video_filename="processed_" + video_filename)



@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)