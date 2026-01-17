import streamlit as st
from huggingface_hub import InferenceClient
from transformers import pipeline
from gtts import gTTS
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Vibe-Shift Engine", page_icon="🧬")
st.title("🧬 Vibe-Shift: The Gen Z NLP Engine")
st.markdown("*Advanced Style Transfer & Brain Rot Analysis*")

# --- API SETUP ---
# Get your token from Hugging Face Settings
hf_token = st.sidebar.text_input("HF API Token", type="password")

if hf_token:
    client = InferenceClient(api_key=hf_token)
    
    # NLP Model for Sentiment (The 'Cool' factor)
    @st.cache_resource
    def load_sentiment():
        return pipeline("sentiment-analysis")
    
    sentiment_pipe = load_sentiment()

    # --- INPUT SECTION ---
    mode = st.radio("Select NLP Task", ["Professional ➡️ Gen Z", "Gen Z ➡️ Boomer (Explain)"])
    user_text = st.text_area("Input Text", placeholder="Enter yapping here...")

    if st.button("Execute Vibe Shift"):
        if user_text:
            with st.spinner("Analyzing semantics..."):
                # 1. NLP Feature: Sentiment Analysis
                sentiment = sentiment_pipe(user_text)[0]
                
                # 2. LLM Style Transfer (The Core Logic)
                if mode == "Professional ➡️ Gen Z":
                    system_prompt = "You are a Gen Z translator. Convert formal text into heavy internet slang like 'no cap', 'fr', 'bussin', 'rizz', and 'on god'. Keep it short."
                else:
                    system_prompt = "You are an English professor. Explain Gen Z slang in very formal, academic British English for an older audience."

                # Calling the LLM
                response = client.chat_completion(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    max_tokens=100
                )
                
                result = response.choices[0].message.content

                # --- OUTPUTS ---
                st.divider()
                st.subheader("Shifted Result:")
                st.success(result)

                # 3. The 'Dr. of AI' Feature: Complexity Analysis
                st.info(f"**NLP Analysis:** Original sentiment is {sentiment['label']} with {round(sentiment['score']*100)}% confidence.")
                
                # 4. Audio Output
                tts = gTTS(text=result, lang='en')
                tts.save("output.mp3")
                st.audio("output.mp3")
        else:
            st.error("Text required, bestie.")
else:
    st.warning("Please provide an HF Token in the sidebar to power the LLM.")
