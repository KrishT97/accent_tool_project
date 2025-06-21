import streamlit as st
import tempfile
import subprocess
import platform
import ctypes
import whisper
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import yt_dlp
from urllib.error import HTTPError, URLError

# Windows workaround for libc
if platform.system() == "Windows":
    whisper.libc = ctypes.CDLL("msvcrt.dll")

# Load models once
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    accent_clf = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent",
    )
    return whisper_model, accent_clf

whisper_model, accent_clf = load_models()

st.title("Accent Detection for English Speakers")
url = st.text_input("Enter public video URL (MP4 or YouTube/Loom)")

if st.button("Analyze Accent") and url:
    # Ensure ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        st.error("`ffmpeg` not found. Please install FFmpeg and ensure it's in your PATH: https://ffmpeg.org/download.html")
        st.stop()

    with st.spinner("Downloading video..."):
        vid_path = None
        try:
            if any(domain in url for domain in ["youtube.com", "youtu.be"]):
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': '%(id)s.%(ext)s',
                    'quiet': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',  # ← spoof real browser
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    vid_path = ydl.prepare_filename(info)
            else:
                vid_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                subprocess.run(["ffmpeg", "-y", "-i", url, "-c", "copy", vid_path], check=True)
        except (HTTPError, URLError) as e:
            st.error(f"Failed to download video: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error downloading video: {e}")
            st.stop()

    with st.spinner("Extracting audio..."):
        audio_path = vid_path.rsplit('.', 1)[0] + '.wav'
        try:
            subprocess.run([
                "ffmpeg", "-i", vid_path,
                "-ar", "16000", "-ac", "1", audio_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg audio extraction failed: {e}")
            st.stop()

    st.success("Audio extracted")

    with st.spinner("Classifying accent..."):
        try:
            # Load audio
            signal, fs = torchaudio.load(audio_path)
            if signal.size(0) > 1:
                signal = signal.mean(dim=0, keepdim=True)

            # Raw classification
            result = accent_clf.classify_batch(signal)
            # Unpack results
            if isinstance(result, (tuple, list)):
                if len(result) == 2:
                    out_prob, out_labels = result
                else:
                    out_prob = result[0]; out_labels = result[-1]
            else:
                raise ValueError(f"Unexpected classify_batch result type: {type(result)}")
            probs = torch.softmax(out_prob, dim=-1)[0]

            # Filter to core English accents and re-normalize
            english_accents = {"england", "american", "australia", "scotland", "ireland", "wales", "new zealand"}
            filtered = [(lbl, probs[idx].item()) for idx, lbl in enumerate(out_labels) if lbl.lower() in english_accents]
            if not filtered:
                # Fallback to global best
                top_idx = torch.argmax(probs).item()
                label = out_labels[top_idx]
                confidence = round(probs[top_idx].item() * 100, 2)
            else:
                # pick top among English accents
                total = sum(score for _, score in filtered)
                lbl, score = max(filtered, key=lambda x: x[1])
                label = lbl
                confidence = round((score / total) * 100, 2)
        except Exception as e:
            st.error(f"Accent classification failed: {e}")
            st.stop()

    # Display results
    st.write(f"**Predicted Accent:** {label}")
    st.write(f"**Confidence:** {confidence}%")
    st.write("#### Explanation")
    st.write(f"Normalized among English accents, this sample best matches {label} with {confidence}% confidence.")
    st.write(f"**Predicted Accent:** {label}")
    st.write(f"**Confidence:** {confidence}%")
    st.write(f"The classifier detected characteristics common to {label}-accented English speech.")
