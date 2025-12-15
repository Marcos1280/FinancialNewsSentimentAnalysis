import streamlit as st
from transformers import pipeline
import numpy as np
from io import BytesIO
from scipy.io import wavfile

# Fine-tuned sentiment model on Hugging Face
SENTIMENT_REPO = "frangipaninpools/Group2"

@st.cache_resource
def load_models():
    """Load Hugging Face pipelines once and cache them for the Streamlit session."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment  = pipeline("sentiment-analysis", model=SENTIMENT_REPO, tokenizer=SENTIMENT_REPO)
    tts        = pipeline("text-to-audio", model="facebook/mms-tts-eng")
    return summarizer, sentiment, tts

summarizer, sentiment_ft, tts = load_models()

st.set_page_config(page_title="Market News Analyzer", page_icon="📰", layout="centered")

st.title("📰 Market News Analyzer")
st.write("This app summarizes market news, runs sentiment analysis on the summary, and reads the result aloud.")

news = st.text_area("Please enter the news here", height=220, placeholder="Paste a financial news article or paragraph...")

col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Summary max_length", min_value=60, max_value=200, value=130, step=5)
with col2:
    min_length = st.slider("Summary min_length", min_value=20, max_value=120, value=40, step=5)

if st.button("Run", type="primary", use_container_width=True):
    if not news or len(news.strip()) < 20:
        st.warning("Please enter a longer news text (at least ~20 characters).")
        st.stop()

    # --- Summarization ---
    with st.spinner("Summarizing..."):
        try:
            summary = summarizer(
                news,
                max_length=int(max_length),
                min_length=int(min_length),
                do_sample=False
            )[0]["summary_text"]
        except Exception as e:
            st.error("Summarization failed.")
            st.exception(e)
            st.stop()

    # --- Sentiment on summary ---
    with st.spinner("Running sentiment analysis on the summary..."):
        try:
            sent = sentiment_ft(summary)[0]
        except Exception as e:
            st.error("Sentiment analysis failed.")
            st.exception(e)
            st.stop()

    st.subheader("Summary of News")
    st.write(summary)

    st.subheader("Sentiment Analysis")
    st.write(f"[{sent.get('label', 'N/A')} | {float(sent.get('score', 0.0)):.3f}]")

    # --- Text-to-Speech ---
    spoken = f"Summary. {summary}. Sentiment. {sent.get('label', 'unknown')}."
    with st.spinner("Generating audio..."):
        try:
            audio_out = tts(spoken)
        except Exception as e:
            st.error("Text-to-speech failed.")
            st.exception(e)
            st.stop()

    # Convert float audio to WAV (int16) and play in Streamlit
    audio_arr = np.asarray(audio_out["audio"], dtype=np.float32).squeeze().reshape(-1)
    sr = int(audio_out["sampling_rate"])
    wav_int16 = (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16)

    buf = BytesIO()
    wavfile.write(buf, sr, wav_int16)
    buf.seek(0)
    st.audio(buf.read(), format="audio/wav")
