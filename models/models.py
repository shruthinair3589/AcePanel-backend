from sqlalchemy import Column, Integer, LargeBinary, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from db import Base
import json
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "recruiter" or "candidate"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)  
    position = Column(String, nullable=True)
    years_of_experience = Column(Float, nullable=True)
    technology = Column(Text, nullable=True)
    resume = Column(LargeBinary, nullable=True)  # new field
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    interviews = relationship("Interview", back_populates="candidate")

    def set_technology(self, tech_list: list[str]):
        self.technology = json.dumps(tech_list) if tech_list else None

    def get_technology(self) -> list[str]:
        return json.loads(self.technology) if self.technology else []

class Interview(Base):
    __tablename__ = "interviews"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    scheduled_at = Column(DateTime, nullable=True)
    status = Column(String, default="Scheduled")  # Scheduled, Completed, Cancelled
    feedback = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    call_id = Column(String, nullable=True, unique=True)
    transcript = Column(Text, nullable=True)
    video_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    candidate = relationship("Candidate", back_populates="interviews")

