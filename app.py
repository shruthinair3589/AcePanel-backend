from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from db import SessionLocal, engine
from models.models import Base, Candidate, User
from typing import List

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for dev
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
    email: EmailStr
    position: str
    years_of_experience: float
    technology: List[str]
# ---------------- Pydantic models ----------------
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # "recruiter" or "candidate"

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

    return {"message": f"{user.role.capitalize()} registered successfully"}

# ---------- Login ----------
@app.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email, User.role == user.role).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": f"{user.role.capitalize()} login successful"}

# ---------- Admin creates candidate (no password) ----------
@app.post("/create-candidate")
def create_candidate(candidate: CandidateCreate, db: Session = Depends(get_db)):
    # Ensure email is unique in candidates table
    db_candidate = db.query(Candidate).filter(Candidate.email == candidate.email).first()
    if db_candidate:
        raise HTTPException(status_code=400, detail="Candidate email already exists")

    new_candidate = Candidate(
        name=candidate.name,
        email=candidate.email,
        position=candidate.position,
        years_of_experience=candidate.years_of_experience
    )
    new_candidate.set_technology(candidate.technology)

    db.add(new_candidate)
    db.commit()
    db.refresh(new_candidate)

    return {
        "id": new_candidate.id,
        "name": new_candidate.name,
        "email": new_candidate.email,
        "position": new_candidate.position,
        "years_of_experience": new_candidate.years_of_experience,
        "technology": new_candidate.get_technology(),
    }


# ---------- Get all candidates ----------
@app.get("/get-all-candidates")
def get_all_candidates(db: Session = Depends(get_db)):
    candidates = db.query(Candidate).all()
    result = []
    for candidate in candidates:
        result.append({
            "id": candidate.id,
            "name": candidate.name,
            "email": candidate.email,
            "position": candidate.position,
            "years_of_experience": candidate.years_of_experience,
            "technology": candidate.get_technology(),
        })
    return result
