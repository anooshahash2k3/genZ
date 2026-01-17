import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Vibe-Shift Engine", page_icon="🧬")
st.title("🧬 Vibe-Shift: The Zero-Token Engine")
st.markdown("---")

# --- LOAD LOCAL MODEL ---
@st.cache_resource
def load_local_nlp():
    # google/flan-t5-base is ideal for zero-token local execution
    return pipeline("text2text-generation", model="google/flan-t5-base")

st_model = load_local_nlp()

# --- SIDEBAR & INPUT ---
mode = st.sidebar.radio("NLP Task", ["Professional ➡️ Gen Z", "Gen Z ➡️ Boomer (Explain)"])
user_text = st.text_area("Input Text", placeholder="Enter your text here...")

if st.button("Execute Vibe Shift"):
    if user_text.strip():
        with st.spinner("AI is analyzing context..."):
            
            # 1. FEW-SHOT PROMPTING LOGIC
            if mode == "Professional ➡️ Gen Z":
                # Providing examples teaches the model the style
                prompt = f"""
                Translate formal English into Gen Z internet slang.
                Formal: I am very tired. -> Slang: I am straight cooked, fr.
                Formal: That is excellent. -> Slang: That is bussin, no cap.
                Formal: I am telling the truth. -> Slang: I'm for real, on god.
                Formal: {user_text} -> Slang:
                """
            else:
                prompt = f"""
                Translate Gen Z slang into formal, academic English.
                Slang: No cap, that's mid. -> Formal: Honestly, that is of average quality.
                Slang: He has rizz. -> Formal: He possesses great charisma.
                Slang: Stop yapping. -> Formal: Please cease this unnecessary talking.
                Slang: {user_text} -> Formal:
                """

            # 2. MODEL INFERENCE
            output = st_model(
                prompt, 
                max_new_tokens=50, 
                repetition_penalty=3.5, # Stops the AI from repeating your words
                do_sample=True,         # Adds variety
                temperature=0.8         # Controls creativity
            )
            
            # Clean the output to remove any remaining prompt text
            result = output[0]['generated_text'].strip()

            # --- DISPLAY ---
            st.subheader("Shifted Result:")
            st.success(result)
            
            # Audio output
            tts = gTTS(text=result, lang='en')
            tts.save("vibe_output.mp3")
            st.audio("vibe_output.mp3")
    else:
        st.error("Please enter some text, bestie.")
