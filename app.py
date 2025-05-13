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

# Page config (must be at top before any Streamlit code)
st.set_page_config(
    page_title="Resume Matcher",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load ML models
model = joblib.load("resume_classification_model.pkl")
tokenizer = joblib.load("resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- STYLES ---
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
        }
        .block-container {
            padding: 2rem 2rem 1rem 2rem;
        }
        h1 {
            color: #6C63FF;
            font-weight: 700;
            text-align: center;
        }
        h2, h3 {
            color: #4A4A4A;
        }
        .stButton>button {
            background-color: #6C63FF;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #7F75FF;
            color: white;
        }
        .stDataFrame {
            background-color: #f4f4f4;
            border-radius: 6px;
        }
        .stMarkdown {
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1>✨ Resume Matcher & Candidate Ranking</h1>", unsafe_allow_html=True)

# --- SESSION STATE ---
if "text_blocks" not in st.session_state:
    st.session_state.text_blocks = []

if "file_names" not in st.session_state:
    st.session_state.file_names = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = 0

# --- RESET BUTTON ---
st.markdown("---")
cols = st.columns([4, 1])
with cols[1]:
    if st.button("🔁 Reset"):
        st.session_state.text_blocks = []
        st.session_state.file_names = []
        st.session_state.processed_files = set()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1
        st.rerun()

# --- UPLOAD INPUTS ---
jd_key = f"jd_uploader_{st.session_state.reset_uploader}"
resumes_key = f"resume_uploader_{st.session_state.reset_uploader}"

jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type="pdf", key=jd_key)
jd = ""
if jd_file is not None:
    try:
        reader = PdfReader(jd_file)
        jd_text = "".join([page.extract_text() or "" for page in reader.pages])
        jd = jd_text.strip()
        if jd:
            st.success("✅ Job Description extracted.")
        else:
            st.warning("⚠️ Could not extract text from JD.")
    except Exception as e:
        st.error(f"❌ Error reading JD PDF: {e}")

uploaded_files = st.file_uploader(
    "📁 Upload Resumes (PDF format)",
    type="pdf",
    accept_multiple_files=True,
    key=resumes_key
)

# --- PROCESS RESUMES ---
if uploaded_files and not st.session_state.submitted:
    st.markdown("### 🧾 Resume Text Extraction:")
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                st.session_state.text_blocks.append(text.strip())
                st.session_state.file_names.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✔️ {uploaded_file.name} processed.")

# --- SUBMIT BUTTON ---
if not st.session_state.submitted:
    if st.button("🚀 Submit for Matching"):
        if not st.session_state.text_blocks:
            st.warning("Upload at least one resume.")
        elif not jd.strip():
            st.warning("Upload a valid job description.")
        else:
            st.session_state.submitted = True
            st.rerun()

# --- TEXT CLEANING AND MATCHING ---
def preprocess_resume(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

def extract_skills_edu_exp(text):
    keywords = {
        "skills": {"skills", "skill", "tools", "technologies"},
        "education": {"bachelor", "master", "phd", "msc", "degree", "university"},
        "experience": {"experience", "worked", "intern", "company"}
    }
    lines = text.split('\n')
    output = []
    for line in lines:
        lower = line.strip().lower()
        if any(k in lower for k in keywords["skills"] | keywords["education"] | keywords["experience"]):
            output.append(line.strip())
    return output

# --- RESULTS ---
if st.session_state.submitted and jd:
    st.markdown("## 🎯 Results")
    resumes = st.session_state.text_blocks
    file_names = st.session_state.file_names

    processed = [extract_skills_edu_exp(preprocess_resume(r)) for r in resumes]
    embeddings = sbert_model.encode(processed)
    jd_embedding = sbert_model.encode([jd])
    similarities = cosine_similarity(jd_embedding, embeddings)[0]

    labels = model.predict(embeddings)
    categories = tokenizer.inverse_transform(labels)

    df = pd.DataFrame({
        "Resume File": file_names,
        "Index": list(range(len(resumes))),
        "Score": similarities,
        "Predicted Category": categories
    })
    df["Rank"] = df["Score"].rank(ascending=False, method='first').astype(int)
    df = df.sort_values("Rank")

    top_idx = df.iloc[0]["Index"]
    top_cat = tokenizer.inverse_transform([model.predict([embeddings[top_idx]])[0]])[0]
    st.success(f"🏆 Best Match Predicted Job Category: **{top_cat}**")

    st.markdown("### 📊 Ranked Candidates")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)

