import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# 1. Page Config
st.set_page_config(page_title="Vibe-Shift Engine", page_icon="🧬")
st.title("🧬 Vibe-Shift: The Zero-Token Engine")
st.markdown("---")

# 2. Load the NLP Model (Local Inference)
@st.cache_resource
def load_model():
    # We use Flan-T5 because it's an 'Instruction-Tuned' model
    return pipeline("text2text-generation", model="google/flan-t5-base")

vibe_ai = load_model()

# 3. Sidebar for Mode Selection
st.sidebar.header("NLP Settings")
mode = st.sidebar.radio("Translation Direction:", 
                       ["Boring ➡️ Gen Z", "Gen Z ➡️ Boring"])

# 4. User Input
user_text = st.text_area("Enter text to transform:", placeholder="Type here...")

if st.button("✨ Transform Vibe"):
    if user_text.strip():
        with st.spinner("Processing semantics..."):
            
            # 5. Advanced Prompt Engineering
            # We add clear markers so the AI knows what is a command and what is data
            if mode == "Boring ➡️ Gen Z":
                input_prompt = f"Rewrite this in heavy Gen Z slang: {user_text}"
            else:
                input_prompt = f"Explain this Gen Z slang in formal English: {user_text}"

            # 6. Model Inference with Anti-Repetition logic
            output = vibe_ai(
                input_prompt, 
                max_length=50,
                repetition_penalty=3.0, # Forces the AI to use new words
                do_sample=True,          # Makes it creative
                temperature=0.8          # Higher temperature = more 'vibe'
            )
            
            result = output[0]['generated_text']

            # 7. Display Result
            st.subheader("Shifted Result:")
            st.success(result)
            
            # 8. Speech Output
            tts = gTTS(text=result, lang='en')
            tts.save("vibe.mp3")
            st.audio("vibe.mp3")
    else:
        st.warning("Please enter some yapping first!")
