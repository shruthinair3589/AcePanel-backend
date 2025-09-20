from sqlalchemy import Column, Integer, String, Float, Text
from db import Base
import json

# Login accounts
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "recruiter" or "candidate"

# Candidate profiles (no password)
class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    position = Column(String, nullable=True)
    years_of_experience = Column(Float, nullable=True)
    technology = Column(Text, nullable=True)

    # Helper methods
    def set_technology(self, tech_list: list[str]):
        if tech_list:
            self.technology = json.dumps(tech_list)
        else:
            self.technology = None

    def get_technology(self) -> list[str]:
        if self.technology:
            return json.loads(self.technology)
        return []
