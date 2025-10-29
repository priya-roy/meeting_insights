import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from meeting_processor import (
    transcribe_video,
    save_transcript_to_file,
    process_and_store_embeddings,
    generate_mom_and_insights
)

st.set_page_config(page_title="Meeting Insights Generator", layout="wide")

st.title("🎥 Meeting Insights Generator (MP4 → Transcript → MOM + Insights)")

# --- Upload & Process in One Go ---
uploaded_file = st.file_uploader("Upload your MP4 meeting recording", type=["mp4"])

if uploaded_file and st.button("Generate Meeting Insights"):
    with st.spinner("⏳ Processing your meeting recording..."):
        try:
            # Step 1: Transcribe
            transcript = transcribe_video(uploaded_file)
            transcript_path = save_transcript_to_file(uploaded_file.name, transcript)
            st.success("✅ Transcription complete!")

            # Step 2: Embedding
            num_chunks, faiss_path = process_and_store_embeddings(transcript_path)
            st.info(f"📚 Created {num_chunks} text chunks and stored in FAISS.")

            # Step 3: Generate Insights
            result_text, model_used = generate_mom_and_insights(faiss_path)

            st.success(f"✅ Meeting Insights generated successfully using {model_used}!")
            st.markdown("### 📝 Meeting Summary, Action Items & Insights")
            st.markdown(result_text)

            # Step 4: Download Option
            st.download_button(
                label="📄 Download MOM + Insights",
                data=result_text,
                file_name="meeting_summary.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("👆 Upload your MP4 meeting file and click **Generate Meeting Insights**.")
