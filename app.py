# app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from dotenv import load_dotenv
load_dotenv()

from config import RECIPIENTS, COMPETENCY
from drive_utils import download_latest_session_files
from meeting_processor import (
    transcribe_audio,
    extract_text_from_pdf,
    generate_meeting_summary,
    send_email_with_summary,
)

def main():
    print("🛰️ Connecting to Google Drive and fetching latest session folder...")
    try:
        video_path, pdf_path, video_link, session_folder_name = download_latest_session_files()
        print(f"✅ Video local path: {video_path}")
        print(f"✅ PDF local path: {pdf_path}")
        print(f"📁 Session folder: {session_folder_name}")

        text_source = ""

        # Transcribe video if available
        if video_path and os.path.exists(video_path):
            print("🎙️ Transcribing video...")
            try:
                text_source = transcribe_audio(video_path)
                print("✅ Transcription complete.")
            except Exception as e:
                print(f"⚠️ Transcription failed: {e}")
                text_source = ""

        # Fallback to PDF if video transcription fails
        if not text_source:
            if pdf_path and os.path.exists(pdf_path):
                print("📄 Extracting PDF text...")
                try:
                    text_source = extract_text_from_pdf(pdf_path)
                    print("✅ PDF text extracted.")
                except Exception as e:
                    raise RuntimeError(f"Failed to extract PDF text: {e}")
            else:
                raise FileNotFoundError("No video or PDF text available to generate insights.")

        # Generate meeting summary
        print("🧠 Generating meeting summary ...")
        summary, model_used = generate_meeting_summary(
            text_source, session_name=session_folder_name, competency=COMPETENCY
        )
        print(f"✅ Summary generated using {model_used}")

        # Send email summary
        print("📧 Sending email to recipients ...")
        send_email_with_summary(RECIPIENTS, summary, video_drive_link=video_link, session_name=session_folder_name)
        print("✅ Email sent successfully.")

        # Save local summary file
        out_file = f"session_insights_{session_folder_name}.txt"
        with open(out_file, "w", encoding="utf-8") as fh:
            fh.write(summary)
        print(f"✅ Summary saved to {out_file}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
