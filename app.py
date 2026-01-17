import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Vibe-Shift Engine", page_icon="🧬")
st.title("🧬 Vibe-Shift: The Zero-Token Engine")
st.markdown("*Local NLP Inference — No API Keys Required*")

# --- LOCAL MODEL LOADING (CACHED) ---
@st.cache_resource
def load_local_nlp():
    # We use 'base' instead of 'large' to save memory on Streamlit's free tier
    return pipeline("text2text-generation", model="google/flan-t5-base")

st_model = load_local_nlp()

# --- INPUT SECTION ---
mode = st.radio("Select NLP Task", ["Professional ➡️ Gen Z", "Gen Z ➡️ Boomer (Explain)"])
user_text = st.text_area("Input Text", placeholder="Enter yapping here...")

if st.button("Execute Vibe Shift"):
    if user_text:
        with st.spinner("AI is thinking locally..."):
            # We create a 'Prompt' that the model understands
            if mode == "Professional ➡️ Gen Z":
                # Instruction Tuning: We tell the model exactly what the 'target domain' is
                prompt = f"Translate this formal text into funny Gen Z internet slang like 'no cap' and 'fr': {user_text}"
            else:
                prompt = f"Explain this Gen Z slang in very formal, academic English: {user_text}"

            # Local Inference
            result = st_model(prompt, max_length=100)[0]['generated_text']

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("Shifted Result:")
            st.success(result)
            
            # --- AUDIO OUTPUT ---
            tts = gTTS(text=result, lang='en')
            tts.save("vibe_output.mp3")
            st.audio("vibe_output.mp3")
    else:
        st.error("Please enter some text first!")
