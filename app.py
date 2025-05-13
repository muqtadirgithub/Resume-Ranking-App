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

# Must go BEFORE any Streamlit rendering
st.set_page_config(page_title="Resume Matcher", layout="centered")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Now CSS injection and Streamlit UI commands can safely follow


# --------------------- Custom CSS ---------------------
custom_css = """
<style>
body, .main {
    background-color: #f0f2f5;
    color: #050505;
    font-family: 'Helvetica Neue', sans-serif;
}
h1 {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1877f2;
    text-align: center;
    margin-bottom: 1rem;
}
h2, h3 {
    color: #1c1e21;
}
button[kind="primary"], div.stButton > button {
    background-color: #1877f2 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
div.stButton > button:hover {
    background-color: #166fe5 !important;
}
section[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px solid #dfe1e5;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.css-1iyq9l3 {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1rem;
}
.stAlert {
    border-radius: 6px;
}
thead tr th {
    background-color: #f5f6f7;
    color: #050505;
    font-weight: 600;
}
tbody tr td {
    color: #1c1e21;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
}
::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #999;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------- NLP Setup ---------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
model = joblib.load("resume_classification_model.pkl")
tokenizer = joblib.load("resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_resume(resume_text):
    text = resume_text.lower()
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def extract_skills_edu_exp(resume_text):
    skills_keywords = {"skills", "tools", "technologies", "programming", "databases", "cloud", "analytics", "networking"}
    education_keywords = {
        "bachelor", "master", "phd", "degree", "university", "college", "b.tech", "mba", "mca", "pgdm", "engineering",
        "computer science", "business", "law", "arts", "healthcare", "marketing", "finance", "project management"
    }
    work_keywords = {"experience", "worked", "employed", "intern", "company", "organization"}

    skills_pattern = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)

    skills, education, experience = [], [], []
    lines = resume_text.split('\n')
    for line in lines:
        lower = line.strip().lower()
        if skills_pattern.search(lower):
            skills.append(line.strip())
        if edu_pattern.search(lower):
            education.append(line.strip())
        if work_pattern.search(lower):
            experience.append(line.strip())
    return skills + education + experience

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.markdown("<h1>📝 Resume Matching and Candidate Ranking</h1>", unsafe_allow_html=True)

# Session State
for key in ["text_blocks", "file_names", "processed_files", "submitted", "reset_uploader"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["text_blocks", "file_names"] else set() if key == "processed_files" else False if key == "submitted" else 0

# Reset Button
col1, col2, col3 = st.columns(3)
with col3:
    if st.button("🔁 Reset Application"):
        st.session_state.text_blocks = []
        st.session_state.file_names = []
        st.session_state.processed_files = set()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1
        st.rerun()

# Upload Job Description
jd_key = f"jd_uploader_{st.session_state.reset_uploader}"
resume_key = f"resume_uploader_{st.session_state.reset_uploader}"

jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type="pdf", key=jd_key)
jd = ""
if jd_file is not None:
    try:
        reader = PdfReader(jd_file)
        jd = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        st.success("✅ Job Description extracted successfully." if jd else "⚠️ No extractable text found.")
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

# Upload Resumes
uploaded_files = st.file_uploader("📑 Upload Multiple Resumes (PDF format)", type="pdf", accept_multiple_files=True, key=resume_key)
if uploaded_files and not st.session_state.submitted:
    st.markdown("### 📄 Extracted Resume Texts:")
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if text:
                st.session_state.text_blocks.append(text)
                st.session_state.file_names.append(file.name)
                st.session_state.processed_files.add(file.name)
                st.success(f"✅ Extracted: {file.name}")

# Submit
if not st.session_state.submitted:
    if st.button("🚀 Submit"):
        if not st.session_state.text_blocks:
            st.warning("⚠️ Please upload at least one valid resume.")
        elif not jd:
            st.warning("⚠️ Please upload a job description.")
        else:
            st.session_state.submitted = True
            st.rerun()

# Results
if st.session_state.submitted and jd:
    st.markdown("## ✅ Submission Complete!")
    resumes = st.session_state.text_blocks
    processed_resumes = [extract_skills_edu_exp(preprocess_resume(r)) for r in resumes]

    embeddings = sbert_model.encode(processed_resumes)
    jd_embedding = sbert_model.encode([jd])
    similarities = cosine_similarity(jd_embedding, embeddings)[0]

    predicted_labels = model.predict(embeddings)
    predicted_categories = tokenizer.inverse_transform(predicted_labels)

    df = pd.DataFrame({
        "Resume File": st.session_state.file_names,
        "Index": list(range(len(resumes))),
        "Score": similarities,
        "Predicted Category": predicted_categories
    })

    df["Rank"] = df["Score"].rank(ascending=False, method='first').astype(int)
    df = df.sort_values(by="Rank")

    top_resume_embedding = embeddings[df.iloc[0]["Index"]].reshape(1, -1)
    top_category = tokenizer.inverse_transform(model.predict(top_resume_embedding))[0]
    st.success(f"🎯 Predicted Job Category for Best Match: **{top_category}**")

    st.markdown("### 📊 Resume Analysis Results:")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], width=700)


    st.markdown("### 📊 Detailed Results")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)



