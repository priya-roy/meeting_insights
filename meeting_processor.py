# meeting_processor.py

from dotenv import load_dotenv
load_dotenv()
import os
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydub import AudioSegment
from faster_whisper import WhisperModel
from tempfile import NamedTemporaryFile
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

BASE_DIR = "data"
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
VECTORDIR = os.path.join(BASE_DIR, "vectordb")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
for p in [TRANSCRIPT_DIR, VECTORDIR, TEMP_DIR]:
    os.makedirs(p, exist_ok=True)

# --- Models / embeddings loader ---
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- transcribe audio from downloaded video file ---
def transcribe_audio(video_local_path):
    if not video_local_path or not os.path.exists(video_local_path):
        raise FileNotFoundError("Video file not found for transcription.")
    model = load_whisper_model()
    # export to mp3 via pydub (ffmpeg must be installed)
    mp3_path = os.path.join(TEMP_DIR, os.path.basename(video_local_path) + ".mp3")
    audio = AudioSegment.from_file(video_local_path)
    audio.export(mp3_path, format="mp3")

    segments, _ = model.transcribe(mp3_path, beam_size=1)
    text = " ".join(seg.text for seg in segments)
    try:
        os.remove(mp3_path)
    except Exception:
        pass
    return text.strip()


# --- extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    reader = PdfReader(pdf_path)
    text = []
    for pg in reader.pages:
        pg_text = pg.extract_text() or ""
        text.append(pg_text)
    return "\n".join(text).strip()


# --- save combined text to file for loader ---
def _save_combined_to_file(text, session_name):
    fn = re.sub(r'[\\/:"*?<>|]+', '_', f"{session_name}_combined.txt")
    path = os.path.join(TRANSCRIPT_DIR, fn)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --- build embeddings and FAISS index (keeps same behaviour) ---
def _build_faiss_from_textfile(text_file_path):
    loader = TextLoader(text_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    faiss_index_path = os.path.join(VECTORDIR, "faiss_index")
    vector_store.save_local(faiss_index_path)
    return len(split_docs), faiss_index_path


# --- generate meeting summary (uses embedding retrieval + LLM) ---
def generate_meeting_summary(text_input, session_name="session", competency="PHP Drupal"):
    """
    text_input: large text (transcript or pdf text) to summarise
    """
    if not text_input or not text_input.strip():
        return "No content available to generate insights."

    # save combined content
    combined_path = _save_combined_to_file(text_input, session_name)
    num_chunks, faiss_path = _build_faiss_from_textfile(combined_path)

    # load and retrieve
    embeddings = load_embeddings()
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 40})
    docs = retriever.invoke("all relevant information")
    big_text = " ".join(d.page_content for d in docs)
    
    """Generate structured Knowledge Meet summary for a technical competency."""

    prompt = f"""
You are an expert technical summariser specialised in Knowledge Meet (KM) sessions.
The session belongs to the "{competency}" technical competency, and the target audience are experienced working professionals in this field.

Your goal is to generate a highly accurate, realistic, and concise summary of a 2-hour technical session, so that any professional can fully understand the discussion in just a few minutes.

Structure your response with the following sections:

1) **Executive Summary** – 2–3 lines explaining what the session covered overall and its key outcome.
2) **Session Overview** – Purpose of the session, technologies or tools discussed, and who led or contributed (if mentioned).
3) **Key Technical Topics Discussed** – Summarise each major topic or subtopic separately (e.g. new features, architecture, module, or workflow). Use bullet points.
4) **Best Practices and Learnings** – List clear technical learnings, patterns, or improvements that were discussed or demonstrated.
5) **Actionable Takeaways** – List specific recommendations, implementation tips, or next steps relevant to professionals in this competency.
6) **Challenges, Risks or Gaps Identified** – Summarise any technical blockers, issues, or questions raised.
7) **Conclusion** – A short closing summary of the session impact and how it contributes to skill growth in this technical area.

Ensure the tone is professional, factual, and concise.
Avoid repetition and do not invent details that were not part of the transcript.

Transcript:
{text_input}
"""
    prompt_template = PromptTemplate(template=prompt, input_variables=["transcript"])

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=700, openai_api_key=openai_key)
        used = "OpenAI GPT"
    else:
        llm = Ollama(model="llama3")
        used = "Ollama Llama3"

    chain = LLMChain(prompt=prompt_template, llm=llm)
    res = chain.invoke({"transcript": big_text})
    return res["text"], used


# --- send email with summary and optional video link ---
def send_email_with_summary(recipients, summary_text, video_drive_link=None, session_name="session"):
    sender = os.getenv("SMTP_SENDER_EMAIL", os.getenv("SMTP_USER"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))

    if not sender or not smtp_user or not smtp_pass:
        raise EnvironmentError("Missing SMTP config: set SMTP_SENDER_EMAIL, SMTP_USER and SMTP_PASS environment variables.")

    html_body = f"""
    <html><body>
    <h3>KM Session Insights - {session_name}</h3>
    {f'<p>🎥 <a href="{video_drive_link}">View Recording</a></p>' if video_drive_link else ''}
    <hr>
    <pre style="white-space:pre-wrap;">{summary_text}</pre>
    <p>--<br>KM Insights Bot</p>
    </body></html>
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"KM Session Insights - {session_name}"
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(sender, recipients, msg.as_string())
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Email send failed: {e}")
