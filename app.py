import os
import whisper
import yt_dlp
import cv2
import numpy as np
import shutil
import uuid
from ffmpeg import input, output, probe  # Importing ffmpeg-python
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from transformers import pipeline
import streamlit as st
from detoxify import Detoxify
from pyngrok import ngrok
import torch
import asyncio

# Streamlit page config should be first
st.set_page_config(page_title="🎬 YouTube Video Summarizer", layout="wide")

# Check for available device and set to CPU if GPU is not available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
asyncio.set_event_loop(asyncio.new_event_loop())

# Load SentenceTransformer model (embedding model)
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optionally, move to the correct device
embedding_model = embedding_model.to(device)

# Initialize KeyBERT with SentenceTransformer model
kw_model = KeyBERT(model=embedding_model)

# Load Whisper model (speech-to-text)
whisper_model = whisper.load_model("small")

# Initialize the toxicity model
toxicity_model = Detoxify('original')

# Sentiment analysis model
with st.spinner('Loading sentiment analysis model...'):
    sentiment_model = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
    print("Sentiment model loaded successfully!")

print("Models loaded successfully!")

# Functions
def download_youtube_video(url):
    try:
        uid = str(uuid.uuid4())[:8]
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'video_{uid}.%(ext)s',
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Check if file is downloaded
        for file in os.listdir():
            if file.startswith(f"video_{uid}") and file.endswith((".mp4", ".mkv", ".webm")):
                return file
        raise FileNotFoundError("Download failed.")
    except Exception as e:
        print(f"Error downloading video: {e}")
        st.error(f"Error downloading video: {e}")
        return None

def extract_audio(video_path, output_audio="audio.wav"):
    try:
        input_video = input(video_path)
        output_audio_path = output(input_video, output_audio)
        output_audio_path.run()
        return output_audio
    except Exception as e:
        print(f"Error extracting audio: {e}")
        st.error(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path, language="en"):
    try:
        result = whisper_model.transcribe(audio_path, language=language)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        st.error(f"Error transcribing audio: {e}")
        return None

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def analyze_sentiment(text):
    return sentiment_model(text[:512])[0]

def extract_keywords(text):
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

def generate_chapters(text, duration, num_chapters=5):
    words = text.split()
    parts = np.array_split(words, num_chapters)
    timestamps = [round(i * duration / num_chapters, 2) for i in range(num_chapters)]
    return [{"time": t, "summary": " ".join(p[:15]) + "..."} for t, p in zip(timestamps, parts)]

def extract_scene_frames(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    prev, count = None, 0
    saved = []
    os.makedirs("scene_frames", exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            score = diff.mean()
            if score > threshold:
                path = f"scene_frames/frame_{count}.jpg"
                cv2.imwrite(path, frame)
                saved.append(path)
        prev = gray
        count += 1
    cap.release()
    return saved

def detect_toxicity(text):
    scores = toxicity_model.predict(text)
    flagged = {k: v for k, v in scores.items() if v > 0.5}
    return flagged if flagged else "No major toxicity/controversy detected."

def generate_blog_or_tweet(transcript, keywords, mode="blog"):
    key_phrases = ", ".join([kw[0] for kw in keywords])
    if mode == "blog":
        return f"""📝 **Auto Blog Summary**

**Overview:**
{transcript[:300]}...

**Key Topics:**
{key_phrases}

**Conclusion:**
This video dives into {keywords[0][0]}, exploring insights and takeaways you won’t want to miss!

#AI #Summarizer #VideoToBlog
"""
    else:
        tweets = [
            "🧵 Thread: Summary of a fascinating video!",
            f"1. Key Theme: {keywords[0][0]}",
            f"2. Other Highlights: {', '.join([kw[0] for kw in keywords[1:4]])}",
            f"3. Summary Snippet: {transcript[:200]}...",
            "4. My Takeaway: Super informative and relevant!",
            "#AI #VideoSummary #ContentToThread"
        ]
        return "\n\n".join(tweets)

# Function to get video duration
def get_video_duration(video_file):
    try:
        # Use ffmpeg.probe to get the metadata
        metadata = probe(video_file)
        # Extract the duration from the metadata
        duration = float(metadata['streams'][0]['duration'])
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        st.error(f"Error getting video duration: {e}")
        return None

# Streamlit UI
st.title("🎥 summarAIze")

video_url = st.text_input("Enter YouTube Video URL")
trigger = st.button("Summarize Video 🎬")

if trigger and video_url:
    # Clean up old files
    for f in os.listdir():
        if f.startswith("video_") or f.endswith((".wav", ".jpg")):
            os.remove(f)
    if os.path.exists("scene_frames"):
        shutil.rmtree("scene_frames")

    # Debug message for start of each process
    st.write("Starting video download...")

    with st.spinner("Downloading video..."):
        video_file = download_youtube_video(video_url)
        if video_file is None:
            st.error("Failed to download video. Please check the URL.")
            st.stop()

    st.write("Starting audio extraction...")

    with st.spinner("Extracting audio..."):
        audio_file = extract_audio(video_file)
        if audio_file is None:
            st.error("Failed to extract audio.")
            st.stop()

    st.write("Starting transcription...")

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(audio_file)
        if transcript is None:
            st.error("Failed to transcribe audio.")
            st.stop()

    st.subheader("📄 Transcript")
    st.text(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)

    # Get video duration
    video_duration = get_video_duration(video_file)
    if video_duration is None:
        st.error("Failed to get video duration.")
        st.stop()

    # Keywords Extraction
    st.subheader("🔍 Keywords")
    keywords = extract_keywords(transcript)
    st.write(keywords)

    # Sentiment Analysis
    st.subheader("😃 Emotion")
    st.write(analyze_sentiment(transcript))

    # Toxicity Analysis
    st.subheader("☢️ Toxicity/Controversy Flags")
    toxicity_report = detect_toxicity(transcript)
    st.write(toxicity_report)

    # Blog Generation
    st.subheader("📝 Auto-Generated Blog")
    blog = generate_blog_or_tweet(transcript, keywords, mode="blog")
    st.markdown(blog)

    # Tweet Generation
    st.subheader("🐦 Twitter Thread Summary")
    tweet = generate_blog_or_tweet(transcript, keywords, mode="tweet")
    st.markdown(tweet)

    # Translations
    st.subheader("🌐 Translations")
    for lang in ["en", "hi", "te"]:
        st.markdown(f"**{lang.upper()}**")
        st.info(translate_text(transcript, lang))

    # Timeline Summary
    st.subheader("🕐 Timeline Summary")
    chapters = generate_chapters(transcript, video_duration)
    for ch in chapters:
        st.markdown(f"⏱️ `{ch['time']}s`: {ch['summary']}")

    # Scene Frames
    st.subheader("🖼️ Scene Frames")
    frames = extract_scene_frames(video_file)
    for f in frames[:10]:
        st.image(f, width=350)

# Ngrok Tunnel
ngrok.kill()
public_url = ngrok.connect(8501)
print("🌐 Streamlit app URL:", public_url)
