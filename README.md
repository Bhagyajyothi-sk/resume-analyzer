# ResumeIQ — AI-Powered Resume Analyzer

A full-stack AI & Data Science project built with **Python Flask** backend and a modern HTML/CSS/JS frontend.

---

## 🚀 Features

### Page 1 — Authentication
- Login & Register forms with validation
- Session-based authentication

### Page 2 — Resume Upload
- Upload **PDF** or **DOCX** resumes (drag & drop or click)
- Automatic text extraction using `pdfplumber` & `python-docx`
- Word count and file size display

### Page 3 — Analysis Dashboard

#### Button 1: Full Resume Analysis
- **ATS Score** (0–100) with animated ring chart and section breakdown
- **Candidate Info** extraction (name, email, phone, experience)
- **Skill Extraction** from 500+ skills across 16 job roles
- **Skill Categorization**: Programming, ML/AI, Cloud, DBs, Web, etc.
- **TF-IDF + Cosine Similarity** job matching across 16 roles
- **4 Charts**: Skills pie, Skill demand bar, Job demand doughnut, Radar
- **Matching jobs** with salary, demand, companies, match %
- **PDF Report Download** with full ReportLab-generated document

#### Button 2: Skill Gap Analyzer
- Enter any target job title (with autocomplete dropdown)
- **Missing Skills** = Job Skills − Candidate Skills
- **TF-IDF Vectorization** + **Cosine Similarity** matching
- **Match % progress bar** and AI similarity score
- **Matched vs Missing** skill pills
- **2 Charts**: Gap pie, Skills radar
- **Personalised Learning Roadmap** per missing skill:
  - 🎓 Curated courses (Coursera, Udemy, free resources)
  - 🔨 Hands-on project ideas
  - 🏆 Industry certifications
  - ⏱ Estimated timeline
- **PDF Roadmap Download**

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.8+, Flask 3.0 |
| ML / NLP | scikit-learn (TF-IDF, Cosine Similarity), NumPy |
| PDF Parsing | pdfplumber |
| DOCX Parsing | python-docx |
| PDF Generation | ReportLab |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Google Fonts (Syne + DM Sans) |

---

## 📁 Project Structure

```
resume_analyzer/
├── app.py                  # Main Flask backend (all AI logic)
├── requirements.txt        # Python dependencies
├── run.sh                  # Quick start script
├── README.md               # This file
├── uploads/                # Uploaded resumes (auto-created)
└── templates/
    └── index.html          # Full frontend SPA
```

---

## ⚙️ Setup & Run

### Option A — Quick Start (Linux/Mac)
```bash
cd resume_analyzer
bash run.sh
```

### Option B — Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open browser
# → http://localhost:5000
```

---

## 🤖 AI/ML Methods Used

### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),       # Unigrams + bigrams
    stop_words='english',
    min_df=1,
    max_features=5000
)
```

### Cosine Similarity
```python
tfidf_matrix = vectorizer.fit_transform([resume_text, job_skills_text])
similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
```

### Skill Gap Formula
```
Missing Skills = Job Skills − Candidate Skills
```

### ATS Scoring
- 8 weighted sections (Contact, Education, Experience, Skills, etc.)
- Bonus for quantifiable achievements (numbers, %)
- Resume length optimization check

---

## 📊 Dataset

- **16 Job Roles** across Data Science, ML, Software Engineering, DevOps, Cybersecurity, Product
- **500+ skills** mapped across all roles
- **200+ learning resources** (courses, projects, certifications)
- `test_size=0.3` equivalent for evaluation splits

---

## 🔑 Demo Login

> Any email + any password (min 6 characters)

---

## 📦 Dependencies

```
flask==3.0.3
flask-cors==4.0.1
pdfplumber==0.11.4
python-docx==1.1.2
scikit-learn==1.5.2
numpy==1.26.4
reportlab==4.2.5
```
