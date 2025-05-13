# Updated Streamlit Resume Matcher with Visual Enhancements (Copy-Paste Ready)

import streamlit as st
import joblib
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models
model = joblib.load("resume_classification_model.pkl")
tokenizer = joblib.load("resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set page config and add custom CSS styling
st.set_page_config(page_title="Resume Analyser", layout="centered", page_icon="📝")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .css-1d391kg { background-color: #ffffff; border-radius: 12px; padding: 1rem; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 6px; padding: 0.5em 1.5em; }
    .stButton>button:hover { background-color: #45a049; }
    .stFileUploader { padding: 10px; border-radius: 10px; background-color: #ffffff; }
    .stDataFrameContainer { padding: 1rem; background: #ffffff; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Resume Matching & Candidate Ranking System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Preprocessing
def preprocess_resume(resume_text):
    text = resume_text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def extract_skills_edu_exp(resume_text):
    skills_keywords = {
        "skills", "skill", "technical skills", "soft skills", "tools", "technologies",
        "frameworks", "languages", "certifications", "programming", "databases", "cloud"
    }
    education_keywords = {
        "bachelor", "masters", "phd", "degree", "university", "college", "b.tech", "mba",
        "m.tech", "mca", "bca", "bsc", "msc", "law", "engineering"
    }
    work_keywords = {"experience", "worked", "employed", "company", "intern", "internship"}

    skills_pattern = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)

    skills, education, experience = [], [], []
    lines = resume_text.split('\n')

    for line in lines:
        sentence = line.strip()
        lower = sentence.lower()
        if skills_pattern.search(lower):
            skills.append(sentence)
        if edu_pattern.search(lower):
            education.append(sentence)
        if work_pattern.search(lower):
            experience.append(sentence)

    return skills + education + experience

# Session state
if "text_blocks" not in st.session_state: st.session_state.text_blocks = []
if "file_names" not in st.session_state: st.session_state.file_names = []
if "processed_files" not in st.session_state: st.session_state.processed_files = set()
if "submitted" not in st.session_state: st.session_state.submitted = False
if "reset_uploader" not in st.session_state: st.session_state.reset_uploader = 0

# Reset
with st.sidebar:
    if st.button("🔁 Reset Application"):
        st.session_state.text_blocks = []
        st.session_state.file_names = []
        st.session_state.processed_files = set()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1
        st.rerun()

# Upload Job Description
jd_uploader_key = f"jd_uploader_{st.session_state.reset_uploader}"
jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type="pdf", key=jd_uploader_key)
jd = ""
if jd_file:
    try:
        reader = PdfReader(jd_file)
        jd_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text: jd_text += text + "\n"
        jd = jd_text.strip()
        if jd:
            st.success("✅ Job Description extracted successfully.")
        else:
            st.warning("⚠️ No extractable text found.")
    except Exception as e:
        st.error(f"❌ Error reading JD PDF: {e}")

# Upload Resumes
resume_uploader_key = f"resume_uploader_{st.session_state.reset_uploader}"
uploaded_files = st.file_uploader(
    "📑 Upload Resumes (PDF format)", type="pdf", accept_multiple_files=True, key=resume_uploader_key
)

if uploaded_files and not st.session_state.submitted:
    st.markdown("### 📄 Resume Extraction:")
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                st.session_state.text_blocks.append(text.strip())
                st.session_state.file_names.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ Extracted from: {uploaded_file.name}")

# Submit
if not st.session_state.submitted and st.button("🚀 Submit for Analysis"):
    if not st.session_state.text_blocks:
        st.warning("⚠️ Please upload at least one resume.")
    elif not jd.strip():
        st.warning("⚠️ Please upload a job description.")
    else:
        st.session_state.submitted = True
        st.rerun()

# Results
if st.session_state.submitted and jd:
    st.markdown("## 🎯 Matching Results")

    resumes = st.session_state.text_blocks
    file_names = st.session_state.file_names
    processed_resumes = [
        extract_skills_edu_exp(preprocess_resume(r)) for r in resumes
    ]
    embeddings = sbert_model.encode(processed_resumes)
    jd_embedding = sbert_model.encode([jd])
    similarities = cosine_similarity(jd_embedding, embeddings)[0]

    predicted_labels = model.predict(embeddings)
    predicted_categories = tokenizer.inverse_transform(predicted_labels)

    df = pd.DataFrame({
        "Resume File": file_names,
        "Index": list(range(len(resumes))),
        "Score": similarities,
        "Predicted Category": predicted_categories
    })
    df["Rank"] = df["Score"].rank(ascending=False, method='first').astype(int)
    df = df.sort_values(by="Rank")

    top_resume_embedding = embeddings[df.iloc[0]["Index"]].reshape(1, -1)
    top_pred = model.predict(top_resume_embedding)
    top_category = tokenizer.inverse_transform(top_pred)[0]
    st.success(f"🏆 Best Match Predicted Category: **{top_category}**")

    st.markdown("### 📊 Detailed Results")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)



