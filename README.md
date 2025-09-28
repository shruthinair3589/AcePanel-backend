# AcePanel-backend
🚀 AcePanel – AI Recruitment Platform (Client)

AcePanel is an AI-powered recruitment application designed to streamline the hiring process by automating candidate management, interview scheduling, AI-driven interviews, and structured feedback generation.

It also addresses a key pain point: employees on in the project often have to adjust their schedules for project interviews and aim to hire candidates who are on the breanch, and managers need to repeatedly request interviews. AcePanel automates this process, reducing time and effort for both employees and recruiters.
<br />
<br />
✨ Features

📂 Candidate Management – Add, view, and track candidates with resume upload & parsing.

📅 Smart Scheduling – Schedule interviews automatically, with email invitations sent to candidates.

🎤 AI-Powered Interviews – Conduct voice/video interviews using Vapi voice agents.

📝 Automated Feedback – Generate structured interview feedback instantly after the session.

🤖 Chatbot Assistance – Get candidate insights in real time using LangChain + Ollama embeddings + ChromaDB (Groq-powered).

⚡ Modern UI – Built with React.js for a seamless recruiter & candidate experience.

<br />
🏗️ Tech Stack

Frontend: React.js, Tailwind CSS

Backend (companion repo): FastAPI

AI/ML: LangChain, Ollama embeddings, ChromaDB, Groq

Voice/Video: Vapi Voice Agent API

Database: SQL-based system (SQLite3)

<br />

📸 Workflow Overview

Recruiter adds candidate with resume.

Email with interview link sent to candidate.

Candidate joins AI voice/video interview.

System auto-generates structured feedback.

Recruiter can query chatbot for instant insights.

This workflow ensures employees on the bench can efficiently participate in interviews without manual coordination, and recruiters can manage multiple candidates seamlessly.

<br />
<br />

🖥️ Backend Setup (FastAPI)

The backend provides the API for candidate management, interview scheduling, AI-driven interviews, and chatbot functionality.

Prerequisites

Python 3.10+

PostgreSQL (or any SQL-based DB)

Environment variables (Vapi, Groq, DB URL, etc.)

# Installation
git clone https://github.com/<your-org>/AcePanel-backend.git<br /><br />
cd AcePanel-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

Environment Variables

Create a .env file in the backend root:

VAPI_KEY=your_vapi_api_key
GROQ_KEY=your_groq_api_key
SMTP_EMAIL=ace.panel123@gmail.com
SMTP_PASSWORD=ylgu cdhx vztt rfak
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
