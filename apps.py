import streamlit as st
import fasttext
import os
import urllib.request

MODEL_PATH = "lid.176.bin"


LANGUAGE_MAP = {
    "en": "English", "bn": "Bengali", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "id": "Indonesian", "pt": "Portuguese", "it": "Italian", "tr": "Turkish",
    "ru": "Russian", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "pa": "Punjabi",
    "mr": "Marathi", "fa": "Persian", "ne": "Nepali", "si": "Sinhala", "sw": "Swahili",
    "th": "Thai", "my": "Burmese", "vi": "Vietnamese", "pl": "Polish", "uk": "Ukrainian",
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading fastText mode")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, MODEL_PATH)
        st.success(" Download complete")
    return fasttext.load_model(MODEL_PATH)

model = load_model()

# FOR UI
st.title("Language Detector with fastText")
st.markdown("Enter any text in any language. ")

user_input = st.text_area("Type something below:", height=150)

if st.button("Detect Language"):
    if user_input.strip():
        label, confidence = model.predict(user_input.strip())
        lang_code = label[0].replace("__label__", "")
        lang_name = LANGUAGE_MAP.get(lang_code, f"Unknown (code: {lang_code})")
        st.success(f"**Language:** {lang_name}  \n**Confidence:** {confidence[0]:.4f}")
    else:
        st.warning("Please enter some text.")
