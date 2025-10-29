import os
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
from langchain_openai import ChatOpenAI

# --- Directory Configuration ---
TRANSCRIPT_DIR = os.path.join("data", "transcripts")
VECTOR_STORE = os.path.join("data", "vectordb")
TEMP_DIR = os.path.join("data", "temp")

for path in [TRANSCRIPT_DIR, VECTOR_STORE, TEMP_DIR]:
    os.makedirs(path, exist_ok=True)

# --- Model Loaders ---
def load_whisper_model():
    """Load Whisper model for transcription."""
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def load_huggingface_embeddings():
    """Load sentence transformer model for embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- File Utilities ---
def save_transcript_to_file(filename, transcript):
    """Save transcript text to file."""
    base_filename, _ = os.path.splitext(os.path.basename(filename))
    txt_filename = f"{base_filename}.txt"
    file_path = os.path.join(TRANSCRIPT_DIR, txt_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return file_path

# --- Audio to Transcript ---
def transcribe_video(uploaded_file):
    """Convert MP4 video to audio, then transcribe using Whisper."""
    model = load_whisper_model()
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name

    audio = AudioSegment.from_file(temp_path, format="mp4")
    temp_mp3_path = os.path.join(TEMP_DIR, f"{os.path.splitext(uploaded_file.name)[0]}.mp3")
    audio.export(temp_mp3_path, format="mp3")

    segments, _ = model.transcribe(temp_mp3_path, beam_size=1)
    transcript_text = "".join(seg.text for seg in segments)

    os.remove(temp_path)
    os.remove(temp_mp3_path)
    return transcript_text

# --- Embedding & Storage ---
def process_and_store_embeddings(transcript_path):
    """Split transcript into chunks, create embeddings and save FAISS index."""
    loader = TextLoader(transcript_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = load_huggingface_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    faiss_index_path = os.path.join(VECTOR_STORE, "faiss_index")
    vector_store.save_local(faiss_index_path)

    return len(docs), faiss_index_path

# --- MOM + Insights Generator ---
def generate_mom_and_insights(faiss_index_path):
    """Generate structured Meeting Summary and Action Items."""
    embeddings = load_huggingface_embeddings()
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 80})
    docs = retriever.invoke("all relevant meeting information")
    transcript_text = " ".join([d.page_content for d in docs])

    mom_prompt = """
    You are an expert meeting summarizer. Review the transcript below and produce a structured report with three clear sections:

    **1. Minutes of Meeting (MOM)**
    - Attendees (if mentioned)
    - Summary of Discussion (key points only)
    - Key Decisions (explicitly made during the meeting)
    - Action Items (use bullet points, mention owner and task if available; infer realistic action items if not stated)

    **2. Meeting Insights**
    - Key Themes or Topics
    - Tone or Sentiment (e.g., collaborative, tense, optimistic)
    - Next Steps or Follow-up Opportunities
    - Potential Risks or Concerns

    **3. Executive Summary**
    - Provide a short paragraph (under 100 words) summarising the overall meeting outcome and direction.

    Keep the output concise, formatted, and professional.

    Transcript:
    {transcript}

    Output:
    """

    prompt = PromptTemplate(template=mom_prompt, input_variables=["transcript"])

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=700,
            openai_api_key=openai_key
        )
        model_name = "OpenAI GPT-4o-mini"
    else:
        llm = Ollama(model="llama3")
        model_name = "Ollama llama3"

    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.invoke({"transcript": transcript_text})
    return result["text"], model_name
