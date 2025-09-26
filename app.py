import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from faster_whisper import WhisperModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuration ---
TRANSCRIPT_DIR = os.path.join("data", "transcripts")
VECTOR_STORE = os.path.join("data", "vectordb")
TEMP_DIR = os.path.join("data", "temp")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# --- Initialize Session State for multi-step workflow ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'transcript_path' not in st.session_state:
    st.session_state.transcript_path = None
if 'faiss_index_path' not in st.session_state:
    st.session_state.faiss_index_path = None


# --- Caching functions ---
@st.cache_resource
def load_whisper_model():
    try:
        model_size = "tiny"
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

@st.cache_resource
def load_huggingface_embeddings():
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Failed to load HuggingFace embeddings model: {e}")
        return None

# --- Utility functions ---
def save_transcript_to_file(filename, transcript):
    base_filename, _ = os.path.splitext(os.path.basename(filename))
    txt_filename = f"{base_filename}.txt"
    file_path = os.path.join(TRANSCRIPT_DIR, txt_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return file_path

def process_and_store_embeddings(transcript_path, embeddings_model):
    loader = TextLoader(transcript_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(docs, embeddings_model)
    faiss_index_path = os.path.join(VECTOR_STORE, "faiss_index")
    vector_store.save_local(faiss_index_path)
    return len(docs), faiss_index_path

def generate_mom_from_faiss(faiss_index_path, embeddings_model, llm_model):
    if not os.path.exists(faiss_index_path):
        st.error("FAISS index not found. Please run the embedding step first.")
        return None

    vector_store = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})
    docs = retriever.invoke("all relevant information")
    full_transcript = " ".join([d.page_content for d in docs])
    
    mom_template = """
    You are an expert at summarizing meeting transcripts and creating concise, professional Minutes of Meeting (MOM).
    Based on the following transcript, generate the Minutes of Meeting with the following sections:
    1.  **Attendees**: List participants if mentioned.
    2.  **Summary of Discussion**: A high-level overview of the key topics discussed.
    3.  **Key Decisions**: Document all final decisions made during the meeting.
    4.  **Action Items**: Clearly list all tasks assigned, including who is responsible.

    **Transcript:**
    {transcript}

    **Minutes of Meeting:**
    """
    
    prompt = PromptTemplate(template=mom_template, input_variables=["transcript"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_model)
    mom_result = llm_chain.invoke(input={"transcript": full_transcript})
    return mom_result["text"]

# --- Streamlit App UI ---
st.title("MP4 to Transcript and RAG Setup")

# --- UI for Transcription (Step 0) ---
if st.session_state.step == 0:
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
    if uploaded_file is not None and st.button("Transcribe Video"):
        with st.spinner("Processing your video..."):
            try:
                model = load_whisper_model()
                if model is None:
                    st.stop()
                
                with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_mp4:
                    temp_mp4.write(uploaded_file.getvalue())
                    temp_mp4_path = temp_mp4.name
                
                audio = AudioSegment.from_file(temp_mp4_path, format="mp4")
                temp_mp3_path = os.path.join(TEMP_DIR, f"{os.path.splitext(uploaded_file.name)[0]}.mp3")
                audio.export(temp_mp3_path, format="mp3")
                
                segments, info = model.transcribe(temp_mp3_path, beam_size=1)
                transcript_text = "".join(segment.text for segment in segments)
                
                st.session_state.transcript_path = save_transcript_to_file(uploaded_file.name, transcript_text)
                
                st.success("Transcription complete!")
                st.write(f"Transcript saved to: `{st.session_state.transcript_path}`")
                st.subheader("Extracted Transcript")
                st.text_area("Full Transcript", transcript_text, height=300)
                
                os.remove(temp_mp4_path)
                os.remove(temp_mp3_path)
                
                st.session_state.step = 1
                st.rerun() # Trigger a rerun to advance the UI
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")

# --- UI for Chunking and Embedding (Step 1) ---
if st.session_state.step >= 1:
    st.write(f"Transcript available at: `{st.session_state.transcript_path}`")
    if st.button("Chunk, Embed, and Store in FAISS"):
        with st.spinner("Creating embeddings and storing in FAISS..."):
            try:
                embeddings_model = load_huggingface_embeddings()
                if embeddings_model:
                    num_chunks, faiss_path = process_and_store_embeddings(st.session_state.transcript_path, embeddings_model)
                    st.session_state.faiss_index_path = faiss_path
                    st.success(f"Successfully processed {num_chunks} chunks and stored embeddings in FAISS.")
                    st.write(f"FAISS index saved to: `{faiss_path}`")
                    st.session_state.step = 2
                    st.rerun() # Trigger a rerun to advance the UI
            except Exception as e:
                st.error(f"An error occurred during embedding and storage: {e}")

# --- UI for Generating MOM (Step 2) ---
if st.session_state.step >= 2:
    st.write(f"FAISS index available at: `{st.session_state.faiss_index_path}`")
    if st.button("Generate Minutes of Meeting (MOM)"):
        with st.spinner("Generating Minutes of Meeting..."):
            try:
                ollama_llm = Ollama(model="llama3")
                embeddings_model = load_huggingface_embeddings()
                mom_text = generate_mom_from_faiss(st.session_state.faiss_index_path, embeddings_model, ollama_llm)
                
                if mom_text:
                    st.success("Minutes of Meeting generated successfully!")
                    st.subheader("Minutes of Meeting")
                    st.markdown(mom_text)
            except Exception as e:
                st.error(f"An error occurred during MOM generation: {e}")
                st.error("Please ensure Ollama is installed and running, and the 'llama3' model is downloaded.")
