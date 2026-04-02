# 🏆 ResumeRank — AI-Powered Resume Screening System

 Live Demo: https://resume-ranker-8ozo.onrender.com  
 GitHub Repo: https://github.com/aliya2702/resume-ranker  
 Demo Video: https://drive.google.com/file/d/1vWLzrWHyg-BdLgezwJ_-ZPc9ESS5bENB/view?usp=drive_link

---

## 📌 Problem Statement

Recruiters often receive hundreds of resumes for a single job opening.  
Manually reviewing and shortlisting candidates is:

-  Time-consuming  
- Error-prone  
-  Subjective and biased  

There is a need for an intelligent system that can **automate resume screening**, ensure **fair evaluation**, and provide **data-driven insights**.

---

## 💡 Solution

**ResumeRank** is an AI-powered web application that:

- Automatically analyzes resumes against a job description  
- Ranks candidates based on relevance  
- Identifies skill gaps and strengths  
- Provides explainable insights for each candidate  

This helps HRs make **faster, smarter, and unbiased hiring decisions**.

---

## ✨ Key Features

### Resume Ranking
- Upload job description + multiple resumes  
- AI ranks candidates based on relevance  

###  Skill Gap Analysis
- Shows matched vs missing skills  
- Highlights candidate weaknesses  

###  Explainable AI
- Provides reasoning behind each ranking  
- Improves transparency in hiring  

### Pool Insights
- Analyze overall candidate pool  
- Identify trends and skill distribution  

###  Candidate Clustering
- Groups candidates into skill-based clusters  

###  JD Analyzer
- Improves job descriptions using AI  

###  History Tracking
- Stores previous hiring sessions  

###  Export & Sharing
- Export results as CSV  
- Email shortlisted candidates  

### Authentication System
- Secure login with role-based access  

---
## 🛠️ Setup Instructions (Run Locally)

### 1. Clone the Repository
git clone https://github.com/aliya2702/resume-ranker.git  
cd resume-ranker  

### 2. Create Virtual Environment (Recommended)
python -m venv venv  
venv\Scripts\activate   (for Windows)

### 3. Install Dependencies
pip install -r requirements.txt  
## 4. Environment Variables

No external API keys are required for this project.

👉 The system uses local NLP techniques (TF-IDF and Cosine Similarity) and works completely offline.

👉 If you want to extend the project in future with Generative AI (like OpenAI), you can add an API key here.



### 5. Run the Application

bash
python app.py

---

### 6. Open in Browser


http://127.0.0.1:5000/
---

## 🧠 How It Works

1. HR uploads a **Job Description**
2. Uploads multiple **Resume PDFs**
3. System extracts and processes text
4. Uses **NLP (TF-IDF)** to compute similarity
5. Ranks candidates based on relevance
6. Generates insights, summaries, and explanations

---

## ⚙️ Tech Stack

### 🔹 Backend
- Python
- Flask

### 🔹 AI / NLP
- Scikit-learn (TF-IDF)
- OpenAI API (for insights & summaries)

### 🔹 Frontend
- HTML, CSS, JavaScript

### 🔹 Deployment
- Render (Cloud Hosting)

### 🔹 Other Tools
- PyPDF2 (PDF parsing)
- Flask-CORS
- Gunicorn

---

##  System Inputs

- Job Description (text)
- Resume PDFs (multiple files)

---

##  Constraints

- Only PDF files allowed  
- Max 30 resumes per upload  
- Max 5 MB per file  
- Requires minimum job description length  

---

## Edge Case Handling

- Skips empty or corrupted resumes  
- Detects duplicate resumes  
- Validates file type and size  
- Handles missing or invalid inputs  
- Prevents unauthorized access  

---

##  Failure Handling

- Returns clear error messages for invalid inputs  
- Skips failed resume processing without crashing  
- Handles AI failures gracefully  
- Ensures system stability under partial failures  

---

##  Future Enhancements

- Use advanced LLMs for semantic matching  
- Add real-time collaboration for HR teams  
- Integrate ATS (Applicant Tracking Systems)  
- Add analytics dashboard with hiring trends  
- Multi-language resume support  

---

##  Impact

- Reduces resume screening time significantly  
- Improves hiring accuracy  
-  Minimizes bias in candidate selection  
-  Introduces AI-driven hiring decisions  

---

##  Author

**Aliya Kousar**  
AI & Full Stack Developer  

---

##  Final Note

ResumeRank is designed as a **production-ready intelligent hiring assistant**, combining AI, NLP, and real-world constraints to solve a critical problem in recruitment.
