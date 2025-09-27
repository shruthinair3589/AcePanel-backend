import io
import shutil
import uuid
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File,APIRouter,Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import mammoth
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from models.models import Base, Candidate, Interview, User
from typing import List, Optional
import pdfplumber
import docx
import nltk
import re
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# Download required NLP resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
router = APIRouter()
RESUME_DIR = "resumes"
os.makedirs(RESUME_DIR, exist_ok=True)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents = []
for filename in os.listdir(RESUME_DIR):
    if filename.endswith(".pdf"):
        loader =  PyPDFLoader(os.path.join(RESUME_DIR, filename))
        docs = loader.load()
        documents.extend(docs)


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)


embedding_model = OllamaEmbeddings(model="all-minilm")
vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_db")
vectordb.persist()
print(vectordb)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(api_key="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", model_name="openai/gpt-oss-120b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

    
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Pydantic models ----------
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # "recruiter" or "candidate"
    position: str | None = None
    years_of_experience: float | None = None
    technology: List[str] | None = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    role: str  # "recruiter" or "candidate"

class CandidateCreate(BaseModel):
    name: str
    email: str
    phone: str
    position: str        # required
    experience_years: float # required
    technology: List[str]   # required
# ---------------- Pydantic models ----------------
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # "recruiter" or "candidate"

class InterviewCreate(BaseModel):
    candidate_id: int
    scheduled_at: datetime

class InterviewResultUpdate(BaseModel):
    transcript: str
    video_url: str
    status: str = "Completed"
    score: float | None = None
    feedback: str | None = None

class ChatRequest(BaseModel):
    message: str


# ---------------- Registration ----------------
@app.post("/register")
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        name=user.name,
        email=user.email,
        password=user.password,
        role=user.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": f"{user.role.capitalize()} registered successfully",
        "name": new_user.name,
        "role": new_user.role,
        "email": new_user.email
    }


# ---------- Login ----------
@app.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email, User.role == user.role).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "message": f"{user.role.capitalize()} login successful",
        "name": db_user.name,
        "role": db_user.role,
        "email": db_user.email
    }
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from models.models import Base, Candidate
from typing import List
import json

# Create tables
Base.metadata.create_all(bind=engine)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Candidate helper ----------
def tech_list_to_string(tech_list: List[str]) -> str:
    return ",".join(tech_list)


# ---------- Create Candidate ----------
@app.post("/create-candidate")
async def create_candidate(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    position: str = Form(None),
    experience_years: float = Form(None),
    technology: str = Form(None),  # comma-separated
    resume_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    existing = db.query(Candidate).filter(Candidate.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Candidate email already exists")

    tech_list = [t.strip() for t in technology.split(",")] if technology else []
    resume_bytes = await resume_file.read() if resume_file else None

    candidate = Candidate(
        name=name,
        email=email,
        phone=phone,
        position=position,
        years_of_experience=experience_years,
        resume=resume_bytes
    )
    candidate.set_technology(tech_list)

    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "position": candidate.position,
        "experience_years": candidate.years_of_experience,
        "technology": candidate.get_technology(),
        "resume_stored": bool(candidate.resume)
    }


# ---------- Edit Candidate ----------
@app.put("/edit-candidate/{candidate_id}")
async def edit_candidate(
    candidate_id: int,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    position: str = Form(None),
    experience_years: float = Form(None),
    technology: str = Form(None),  # comma-separated
    resume_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if candidate.email != email:
        existing = db.query(Candidate).filter(Candidate.email == email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")

    candidate.name = name
    candidate.email = email
    candidate.phone = phone
    candidate.position = position
    candidate.years_of_experience = experience_years

    if technology:
        candidate.set_technology([t.strip() for t in technology.split(",")])

    if resume_file:
        candidate.resume = await resume_file.read()

    db.commit()
    db.refresh(candidate)

    return {
        "message": "Candidate updated successfully",
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "position": candidate.position,
        "experience_years": candidate.years_of_experience,
        "technology": candidate.get_technology(),
        "resume_stored": bool(candidate.resume)
    }


# ---------- Delete Candidate ----------
@app.delete("/delete-candidate/{candidate_id}")
def delete_candidate(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    db.delete(candidate)
    db.commit()

    return {"message": f"Candidate with ID {candidate_id} deleted successfully"}


# ---------- Get Single Candidate ----------
@app.get("/get-candidate/{candidate_id}")
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "position": candidate.position,
        "experience_years": candidate.years_of_experience,
        "technology": candidate.get_technology(),
        "resume_stored": bool(candidate.resume)
    }


# ---------- Get All Candidates ----------
@app.get("/get-all-candidates")
def get_all_candidates(db: Session = Depends(get_db)):
    candidates = db.query(Candidate).all()
    result = []
    for c in candidates:
        result.append({
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "phone": c.phone,
            "position": c.position,
            "experience_years": c.years_of_experience,
            "technology": c.get_technology(),
            "resume_stored": bool(c.resume)
        })
    return result


# Hardcoded tech skills
REFERENCE_TECH_SKILLS = list({
    "Python", "Java", "JavaScript", "React", "Node.js", "SQL",
    "AWS", "Docker", "Kubernetes", "GCP", "Azure", "HTML", "CSS",
    "TypeScript", "C++", "C#", "Go", "Ruby", "PHP", "MongoDB",
    "PostgreSQL", "MySQL", "Django", "Flask", "Spring", "Angular",
    "Vue.js", "Swift", "Objective-C", "R", "MATLAB", "TensorFlow", "PyTorch",
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
    "Data Science", "DevOps", "Agile", "Scrum", "Microservices", "REST", "GraphQL",
    "NoSQL", "Redis", "Linux", "Git", "CI/CD", "Jenkins", "Terraform", "Ansible",
    "Prometheus", "Grafana", "Hadoop", "Spark", "Kafka", "Elasticsearch", "Tableau", "Power BI",
    "Salesforce", "SAP", "Oracle", "Unity", "Unreal Engine", "Blockchain", "Cryptography",
    "Cybersecurity", "Networking", "Virtualization", "VMware", "Hyper-V", "Big Data", "ETL",
    "Data Warehousing", "Business Intelligence", "UI/UX Design", "Figma", "Adobe XD", "Redux",
    "Next.js", "Nuxt.js", "Svelte", "Flutter", "React Native", "Ionic", "Xamarin", "Apache",
    "Nginx", "Load Balancing", "WebSockets", "OAuth", "OpenID Connect", "JWT", "Accessibility",
    "Express.js", "Laravel", "Symfony", "CodeIgniter", "CakePHP", "Zend Framework", "Phalcon",
    "Yii", "ColdFusion", "Delphi", "Pascal", "Fortran", "COBOL", "Assembly Language",
    ".NET", ".NET Core", "ASP.NET", "Blazor", "WPF", "WinForms", "XAML", "Silverlight",
    "Entity Framework", "LINQ", "Visual Basic .NET", "F#", "PowerShell", "Bash Scripting",
    "Shell Scripting", "Java EE", "Java ME", "J2EE", "J2ME", "EJB", "JPA", "JSF", "GWT",
    "Vaadin", "Apache Struts", "Apache Wicket", "Play Framework", "Dropwizard", "Micronaut",
    "Quarkus", "Vert.x", "JavaFX", "Swing", "AWT", "SWT", "JDBC", "JMS", "Apache Camel",
    "Apache Kafka", "Apache ActiveMQ", "RabbitMQ", "ZeroMQ", "gRPC", "Thrift", "Protocol Buffers",
    "JSON-RPC", "Hibernate", "Maven", "Gradle", "Ant", "Travis CI", "CircleCI", "GitLab CI/CD",
    "Bamboo", "SonarQube", "Nexus Repository", "Artifactory", "JIRA", "Confluence",
    # AI-specific skills
    "Scikit-learn", "Keras", "OpenCV", "spaCy", "NLTK", "Hugging Face", "Transformers",
    "Stable Diffusion", "LangChain", "LLMs", "Prompt Engineering", "Generative AI",
    "Reinforcement Learning", "Speech Recognition", "Image Segmentation", "Object Detection",
    "AutoML", "MLflow", "ONNX", "Triton Inference Server", "Model Deployment", "MLOps"
})
COMMON_DESIGNATIONS = ["Engineer", "Developer", "Manager", "Analyst", "Consultant", "Lead", "Architect"]
def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    elif file.filename.endswith(".docx") or file.filename.endswith(".doc"):
        result = mammoth.extract_raw_text(file.file)
        return result.value
    else:
        raise ValueError("Unsupported file type")

def extract_email(text: str) -> str:
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group(0) if match else ""

def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text)
    return match.group(0) if match else ""

def extract_name(text: str) -> str:
    skip_keywords = ["RESUME", "CURRICULUM", "VITAE", "PROFILE", "DETAILS"]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes like "Name:", "Candidate Name:"
        line = re.sub(r"^(Name|Candidate Name)[:\-]\s*", "", line, flags=re.IGNORECASE)

        if any(word in line.upper() for word in skip_keywords):
            continue

        # Match Title Case with optional middle initials
        if re.match(r"^[A-Z][a-z]+(?:\s[A-Z]\.?)?(?:\s[A-Z][a-z]+)*$", line):
            return line

        # Match ALL CAPS with optional middle initials
        if re.match(r"^[A-Z]+(?:\s[A-Z]\.?)?(?:\s[A-Z]+)*$", line):
            return line.title()

        # Fallback: first line with <=5 words and no digits
        if len(line.split()) <= 5 and not any(char.isdigit() for char in line):
            return line.title()

    return ""


def extract_tech_skills(text: str) -> List[str]:
    skills_found = [tech for tech in REFERENCE_TECH_SKILLS if re.search(rf"\b{re.escape(tech)}\b", text, re.IGNORECASE)]
    return skills_found

def extract_years_of_experience(text: str) -> int | None:
    # Look for patterns like "5 years" or "3+ years"
    match = re.search(r"(\d+)(\+)?\s*years?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_designation(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:15]:  # check first 15 lines
        for pat in COMMON_DESIGNATIONS:
            # Match designation along with preceding word (like "React Developer")
            match = re.search(rf"(\b\w+\s+)?\b{pat}\b", line, re.IGNORECASE)
            if match:
                return match.group(0).strip()
    return ""

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
    except ValueError as e:
        return {"error": str(e)}

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "designation": extract_designation(text),
        "years_of_experience": extract_years_of_experience(text),
        "technical_skills": extract_tech_skills(text)
    }
    
@app.get("/download-resume/{candidate_id}")
def download_resume(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    if not candidate.resume:
        raise HTTPException(status_code=404, detail="No resume stored for this candidate")

    # Default to PDF mime type; you can extend logic to detect from filename
    mime_type = "application/pdf"
    file_stream = io.BytesIO(candidate.resume)

    # Suggest a filename
    filename = f"{candidate.name.replace(' ', '_')}_resume.pdf"

    return StreamingResponse(file_stream, media_type=mime_type, headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })
# ------------------- Interview Endpoints -------------------
@app.get("/get-all-interviews")
def get_all_interviews(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    all_interviews = db.query(Interview).all()
    upcoming, past = [], []

    for i in all_interviews:
        interview_data = {
            "id": i.id,
            "candidate_id": i.candidate_id,
            "candidate_name": i.candidate.name,
            "scheduled_at": i.scheduled_at,
            "status": i.status,
            "call_id": i.call_id,
            "feedback": i.feedback,
            "score": i.score,
            "transcript": i.transcript,
            "video_url": i.video_url
        }
        if i.scheduled_at >= now:
            upcoming.append(interview_data)
        else:
            past.append(interview_data)

    return {"upcoming": upcoming, "past": past}

@app.put("/interview/{interview_id}/complete")
def complete_interview(interview_id: int, result: InterviewResultUpdate, db: Session = Depends(get_db)):
    interview = db.query(Interview).filter(Interview.id == interview_id).first()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    interview.transcript = result.transcript
    interview.video_url = result.video_url
    interview.status = result.status
    interview.score = result.score
    interview.feedback = result.feedback
    db.commit()
    db.refresh(interview)
    return {
        "message": "Interview completed and data saved",
        "interview_id": interview.id,
        "call_id": interview.call_id,
        "status": interview.status
    }

@app.get("/candidate/{candidate_id}/interviews")
def get_candidate_interviews(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return [
        {
            "id": i.id,
            "scheduled_at": i.scheduled_at,
            "status": i.status,
            "feedback": i.feedback,
            "score": i.score,
            "call_id": i.call_id,
            "transcript": i.transcript,
            "video_url": i.video_url
        } for i in candidate.interviews
    ]

@app.get("/interview/{interview_id}")
def get_interview(interview_id: int, db: Session = Depends(get_db)):
    interview = db.query(Interview).filter(Interview.id == interview_id).first()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    return {
        "id": interview.id,
        "candidate_id": interview.candidate_id,
        "scheduled_at": interview.scheduled_at,
        "status": interview.status,
        "feedback": interview.feedback,
        "score": interview.score,
        "call_id": interview.call_id,
        "transcript": interview.transcript,
        "video_url": interview.video_url
    }

transcript_storage = {}

@app.post("/save-transcript")
async def save_transcript(request: Request):
    data = await request.json()
    call_id = data.get("call_id")
    role = data.get("role")
    transcript = data.get("transcript")

    if not call_id or not role or not transcript:
        return JSONResponse({"error": "call_id, role, transcript required"}, status_code=400)

    if call_id not in transcript_storage:
        transcript_storage[call_id] = []

    transcript_storage[call_id].append({"role": role, "text": transcript})
    return {"message": "Transcript saved successfully"}

def fetch_call_details(call_id: str):
    url = f"https://api.vapi.ai/call/{call_id}"
    headers = {
        "Authorization": f"Bearer {os.getenv('VAPI_API_KEY')}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

@app.get("/call-details")
async def get_call_details(call_id: str = Query(None)):
    if not call_id:
        return JSONResponse(content={"error": "Call ID is required"}, status_code=400)

    try:
        print("calling function from BE")
        response = fetch_call_details(call_id)
        summary = response.get("summary")
        analysis = response.get("analysis")
        transcripts = transcript_storage.get(call_id, [])
        return {"analysis": analysis, "summary": summary, "transcripts": transcripts}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.message
    if not query.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        # Run through RAG pipeline
        result = qa_chain.invoke({"query": query})

        # Extract answer + source docs if available
        answer = result.get("result")
        sources = []
        if "source_documents" in result:
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No selected file")

        file_path = os.path.join(RESUME_DIR, file.filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            chunks = splitter.split_documents(docs)
            vectordb.add_documents(chunks)
            vectordb.persist()
            

        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Email Function ----------------
def send_interview_email(to_email: str, candidate_name: str, candidate_id: int, interview_id: int, scheduled_at: datetime):
    sender_email = os.getenv("SMTP_EMAIL")       # ace.panel123@gmail.com
    sender_password = os.getenv("SMTP_PASSWORD") # Gmail App Password
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    
    subject = "Your Interview is Scheduled"
    
    # Candidate-specific login link with prefill
    interview_link = (
        f"http://127.0.0.1:3000/login?"
        f"role=candidate&candidate_id={candidate_id}&interview_id={interview_id}&email={to_email}"
    )
    
    scheduled_str = scheduled_at.strftime("%A, %d %B %Y at %I:%M %p")
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height:1.5;">
        <h2 style="color:#333;">Hi {candidate_name},</h2>
        <p>Your interview has been scheduled successfully. Please find the details below:</p>
        <table style="border-collapse: collapse;">
            <tr>
                <td style="padding: 5px; font-weight:bold;">Date & Time:</td>
                <td style="padding: 5px;">{scheduled_str}</td>
            </tr>
            <tr>
                <td style="padding: 5px; font-weight:bold;">Interview Link:</td>
                <td style="padding: 5px;"><a href="{interview_link}" style="color:#1a73e8;">Login & Join Interview</a></td>
            </tr>
        </table>
        <p>Kindly be ready 5 minutes before the scheduled time.</p>
        <p>Best regards,<br/>Recruitment Team</p>
    </body>
    </html>
    """
    
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = to_email
    message.attach(MIMEText(html_content, "html"))
    
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message.as_string())
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

# ---------------- Schedule Interview Endpoint ----------------
@app.post("/schedule-interview")
def schedule_interview(interview: InterviewCreate, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == interview.candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    call_id = str(uuid.uuid4())
    new_interview = Interview(
        candidate_id=interview.candidate_id,
        scheduled_at=interview.scheduled_at,
        status="Scheduled",
        call_id=call_id
    )
    db.add(new_interview)
    db.commit()
    db.refresh(new_interview)
    
    # Send email with correct arguments
    send_interview_email(
        to_email=candidate.email,
        candidate_name=candidate.name,
        candidate_id=candidate.id,
        interview_id=new_interview.id,
        scheduled_at=new_interview.scheduled_at
    )
    
    return {
        "message": "Interview scheduled and email sent",
        "interview_id": new_interview.id,
        "call_id": new_interview.call_id,
        "candidate_id": new_interview.candidate_id,
        "scheduled_at": new_interview.scheduled_at,
        "status": new_interview.status
    }
