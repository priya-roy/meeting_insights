# 📋 Overview

This project automatically fetches Knowledge Meet recordings and related PDFs from Google Drive, transcribes the meeting video (or reads the PDF if no video), and then generates AI-powered meeting insights tailored to your technical competency — such as PHP Drupal.

## The final output includes:

 - A detailed, structured summary of a 2-hour technical session
 - Key learnings, takeaways, and gaps
 - Automatic email delivery of the summary and Drive link to a target group

# ⚙️ Features

 - ✅ Fetches latest session folder from Google Drive (PythonKMSessionInsightsApp)
 - ✅ Handles both video + PDF summarisation
 - ✅ Uses OpenAI API for concise and topic-based AI summaries
 - ✅ Supports competency-specific context (e.g., “PHP Drupal”)
 - ✅ Email delivery with optional Google Drive link
 - ✅ Modular 3-file architecture for clarity and scalability


# 📁 Project Structure

meeting-insights/
│
├── app.py                   # Main entry point
├── meeting_processor.py     # Core processing logic: transcription, summary, email
├── drive_utils.py           # Google Drive integration and file handling
├── meetinginsightsapp-xxxx.json   # Google service account credentials
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


# 🧩 Prerequisites

## Before running the app, make sure you have:

 - Python 3.9 or above
 - A Google Cloud service account with Drive API enabled
 - Your OpenAI API key

# 🔐 Setting Up Google Drive Access

## Create a Service Account from Google Cloud Console

 - Enable the Google Drive API for your project.
 - Download the JSON credentials file, rename it to:meetinginsightsapp-xxxx.json
 - Place this file in your project root.

# In your Google Drive:
 -Create a root folder named PythonKMSessionInsightsApp
 - Inside it, create subfolders named like: 2025-04-26_KM

# Each folder should contain:
 - A meeting recording (Google Drive link or shortcut to an MP4)
 - A related PDF deck (slides or notes)

# Share this root folder with your service account email (from the JSON file).

# ⚡ Installation

## Clone the repository and install dependencies:

git clone https://github.com/yourusername/meeting-insights.git
cd meeting-insights
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows

### pip install -r requirements.txt

# 🚀 Usage

Run the app manually (no UI):
### python app.py

# It will:

 - Fetch the latest session folder from Google Drive

 - Download the video (or PDF if video fails)

 - Transcribe and generate AI meeting insights

 - Email the summary to pre-defined recipients

# 🧠 Example Output

# Executive Summary:
 - The Knowledge Meet focused on Drupal CMS GenAI capabilities, covering module integration, voice search enhancements, and AI-driven patch automation.

# Key Learnings:

 - Using OpenAI Whisper for automated speech-to-text in Drupal

 - Optimising GenAI API calls for cost efficiency

 - Roadmap for AI-assisted Drupal module testing

# Actionable Takeaways:

 - Enable AI summarisation in upcoming sprints

 - Create internal contribution tracking dashboard
