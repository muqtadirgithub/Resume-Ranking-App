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

# Page config (must be at top before any other Streamlit commands)
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

# Load ML components
model = joblib.load("resume_classification_model.pkl")
tokenizer = joblib.load("resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- STYLES ---
st.markdown("""
    <style>
        body {
            color: #ffffff;
            background-color: #0a192f;
        }
        .stApp {
            background-color: #0a192f;
            color: #ffffff;
        }
        .css-18e3th9 {
            padding-top: 3rem;
        }
        h1, h2, h3 {
            color: #64ffda;
        }
        .stButton>button {
            background-color: #112240;
            color: #64ffda;
            border: 1px solid #64ffda;
        }
        .stButton>button:hover {
            background-color: #64ffda;
            color: #0a192f;
        }
        .stDataFrame {
            background-color: #112240;
        }
        .block-container {
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🧠 Resume Matcher & Candidate Ranking</h1>", unsafe_allow_html=True)

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
cols = st.columns([3, 1, 1])
with cols[2]:
    if st.button("🔁 Reset App"):
        st.session_state.text_blocks = []
        st.session_state.file_names = []
        st.session_state.processed_files = set()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1
        st.rerun()

# --- INPUT WIDGETS ---
jd_uploader_key = f"jd_uploader_{st.session_state.reset_uploader}"
resume_uploader_key = f"resume_uploader_{st.session_state.reset_uploader}"

jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type="pdf", key=jd_uploader_key)
jd = ""
if jd_file is not None:
    try:
        reader = PdfReader(jd_file)
        jd_text = "".join([page.extract_text() or "" for page in reader.pages])
        jd = jd_text.strip()
        if jd:
            st.success("✅ Job Description extracted successfully.")
        else:
            st.warning("⚠️ No extractable text found.")
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

uploaded_files = st.file_uploader(
    "📁 Upload Multiple Resumes (PDF format)",
    type="pdf",
    accept_multiple_files=True,
    key=resume_uploader_key
)

# --- PROCESS UPLOADED RESUMES ---
if uploaded_files and not st.session_state.submitted:
    st.markdown("### 📑 Extracting Resume Text:")
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                st.session_state.text_blocks.append(text.strip())
                st.session_state.file_names.append(uploaded_file.name)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ Extracted: {uploaded_file.name}")

# --- SUBMIT BUTTON ---
if not st.session_state.submitted:
    if st.button("🚀 Submit"):
        if len(st.session_state.text_blocks) == 0:
            st.warning("Please upload at least one valid resume.")
        elif not jd.strip():
            st.warning("Please upload a valid job description.")
        else:
            st.session_state.submitted = True
            st.rerun()

# --- PROCESS AND DISPLAY RESULTS ---
def preprocess_resume(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def extract_skills_edu_exp(text):
    keywords = {
        "skills": {"skills", "skill", "technical skills", "tools", "technologies"},
        "education": {"bachelor", "master", "phd", "degree", "msc", "university"},
        "experience": {"experience", "worked", "intern", "company", "organization"}
    }
    lines = text.split('\n')
    collected = []
    for line in lines:
        lower = line.strip().lower()
        if any(kw in lower for kw in keywords["skills"] | keywords["education"] | keywords["experience"]):
            collected.append(line.strip())
    return collected

if st.session_state.submitted and jd:
    st.markdown("## ✅ Submission Complete!")
    resumes = st.session_state.text_blocks
    file_names = st.session_state.file_names

    processed_resumes = [extract_skills_edu_exp(preprocess_resume(resume)) for resume in resumes]
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

    top_index = df.iloc[0]["Index"]
    top_pred = model.predict([embeddings[top_index]])[0]
    top_category = tokenizer.inverse_transform([top_pred])[0]

    st.success(f"🎯 Best Match Predicted Category: **{top_category}**")
    st.markdown("### 📊 Ranked Resume Results:")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)



    st.markdown("### 📊 Detailed Results")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)



