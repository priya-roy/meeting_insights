# meeting_processor.py

from dotenv import load_dotenv
load_dotenv()
import os
import smtplib
import re
import markdown
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
You are a **technical assistant** that analyses technical knowledge-sharing or project discussion videos.  
Your goal is to extract the most relevant and actionable insights for internal knowledge management.  
Focus on technical accuracy, logical structure, and readability.  
The summary will be shared via email with the engineering team.

---

### 🔹 Step-by-Step Instructions

1. **Extract Key Points:**  
   Identify all distinct discussion areas such as topics, tools, frameworks, blockers, dependencies, and next steps.

2. **Summarise Clearly:**  
   Write a structured and detailed summary under the following fixed sections — always include all, even if brief:

   - **Session Type** – Identify the nature of the session (e.g., Technical Deep Dive, Design Review, POC Discussion, Sprint Retrospective).  
   - **Agenda** – State the primary goal or purpose of the session summarise in 4-5 points. 
   - **Overview of Session** – Provide a concise summary of discussion flow, key themes, and main directions.  
   - **Technical Discussion** – Detail the technical content discussed:
     - Include mentions of tools, frameworks, APIs, configurations, workflows, or demos.  
     - Add architecture insights, version numbers, performance metrics, or comparisons if available.  
     - Retain technical terms and quotes accurately.  
   - **Dependencies** – Mention dependencies, blockers, integrations, or external systems referenced.  
   - **Decisions / Next Steps** – Clearly list outcomes, conclusions, and follow-up actions.  
   - **Key Takeaways / Learning Points** – Capture lessons, insights, or best practices valuable for future reference.

---

### 🔹 Consistency Rules
- Maintain **factual and logical continuity** — do not skip or merge unrelated points.  
- Avoid altering or paraphrasing **critical technical terminology**.  
- Keep **section headers identical** in every output for consistency.  
- Use **bullet points** under each section for clarity and quick reading.

---

### 🔹 Formatting Requirements
- Write in a **clean, professional markdown style**, suitable for internal knowledge-sharing emails.  
- Keep tone **formal and concise**, while retaining technical depth.  
- Aim for a length of **400–600 words**.  
- Structure content with logical flow and precise wording.

---

### 🔹 Preprocessing Tip (optional, for Whisper + GPT workflow)
Before summarisation, clean the transcript by removing filler words such as *“uh”*, *“you know”*, and *“basically”* to improve coherence.

---

**Transcript:**  
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
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))

    if not sender or not smtp_user or not smtp_pass:
        raise EnvironmentError("Missing SMTP config: set SMTP_SENDER_EMAIL, SMTP_USER and SMTP_PASS environment variables.")

    # 1. Convert the Markdown summary_text to HTML
    # This correctly translates markdown headings (#), bold (**), lists (*), etc., into HTML tags (<h1>, <strong>, <ul>, <li>).
    html_summary_content = markdown.markdown(summary_text)

    # 2. Embed the converted HTML content directly into the email body
    html_body = f"""
    <html>
    <head>
        <style>
            /* Optional: Add some basic styling for better readability */
            body {{ font-family: sans-serif; line-height: 1.6; color: #333; }}
            h3 {{ color: #004d99; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            ul {{ list-style-type: disc; margin-left: 20px; }}
            /* Optional: Style for code blocks, if your LLM uses them (```code```) */
            pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h3>KM Session Insights - {session_name}</h3>
        {f'<p>🎥 <a href="{video_drive_link}">View Recording</a></p>' if video_drive_link else ''}
        <hr>
        
        {html_summary_content}
        
        <p>--<br>KM Insights Bot</p>
    </body>
    </html>
    """

    # 3. Rest of the email setup remains the same
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"KM Session Insights - {session_name}"
    
    # It's good practice to send both HTML and plain text (original markdown) for compatibility, 
    # but for simplicity, we'll stick to sending just the HTML part as you did previously.
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(sender, recipients, msg.as_string())
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Email send failed: {e}")
