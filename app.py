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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and label encoder
model = joblib.load("resume_classification_model.pkl")
tokenizer = joblib.load("resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit page config and style
st.set_page_config(page_title="Resume Analyzer", layout="wide", page_icon="📄")
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2E7D32; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stDataFrame thead { background-color: #2E7D32; color: white; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>📋 Resume Matching & Candidate Ranking</h1>", unsafe_allow_html=True)

# Initialize session state
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

# Reset functionality
col1, col2, col3 = st.columns(3)
with col3:
    if st.button("🔁 Reset"):
        st.session_state.text_blocks.clear()
        st.session_state.file_names.clear()
        st.session_state.processed_files.clear()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1
        st.rerun()

# Define unique keys
jd_key = f"jd_uploader_{st.session_state.reset_uploader}"
resume_key = f"resume_uploader_{st.session_state.reset_uploader}"

# Job description input
st.subheader("📌 Upload Job Description (PDF)")
jd_file = st.file_uploader("Job Description", type="pdf", key=jd_key)
jd = ""
if jd_file:
    reader = PdfReader(jd_file)
    jd = "\n".join(p.extract_text() for p in reader.pages if p.extract_text()).strip()
    if jd:
        st.success("✅ JD uploaded successfully.")
    else:
        st.warning("⚠️ No readable text found in the file.")

# Resume input
st.subheader("📎 Upload Resumes (PDF, Multiple Allowed)")
uploaded_files = st.file_uploader("Candidate Resumes", type="pdf", accept_multiple_files=True, key=resume_key)

if uploaded_files and not st.session_state.submitted:
    st.markdown("### 🔍 Extracting Text from Resumes:")
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            text = "\n".join(p.extract_text() for p in PdfReader(file).pages if p.extract_text())
            if text.strip():
                st.session_state.text_blocks.append(text.strip())
                st.session_state.file_names.append(file.name)
                st.session_state.processed_files.add(file.name)
                st.success(f"📝 Processed: {file.name}")
            else:
                st.warning(f"⚠️ Could not extract text from {file.name}")

# Submit for processing
if not st.session_state.submitted:
    if st.button("🚀 Submit"):
        if not jd.strip():
            st.warning("📄 Please upload a job description.")
        elif not st.session_state.text_blocks:
            st.warning("📎 Please upload at least one resume.")
        else:
            st.session_state.submitted = True
            st.rerun()

# Preprocessing
def preprocess_resume(resume_text):
    text = re.sub(r'[^\w\s\d]', '', resume_text.lower())
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

def extract_skills_edu_exp(text):
    keywords = {
        "skills", "technologies", "tools", "languages", "certifications",
        "bachelor", "master", "phd", "education", "university",
        "experience", "company", "worked", "intern", "employment"
    }
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    return [line.strip() for line in text.split('\n') if pattern.search(line)]

# Process and rank resumes
if st.session_state.submitted and jd:
    with st.spinner("🧠 Analyzing resumes..."):
        resumes = st.session_state.text_blocks
        filenames = st.session_state.file_names
        processed = [extract_skills_edu_exp(preprocess_resume(res)) for res in resumes]
        embeddings = sbert_model.encode(processed)
        jd_embedding = sbert_model.encode([jd])
        scores = cosine_similarity(jd_embedding, embeddings)[0]
        preds = tokenizer.inverse_transform(model.predict(embeddings))

        results = pd.DataFrame({
            "Resume File": filenames,
            "Score": scores,
            "Predicted Category": preds
        })
        results["Rank"] = results["Score"].rank(method="first", ascending=False).astype(int)
        results["Score"] = results["Score"].round(3)
        results.sort_values("Rank", inplace=True)

    st.success("✅ Analysis Complete!")
    top_row = results.iloc[0]
    st.markdown(f"""
    ### 🏆 Best Match
    - **Resume:** `{top_row['Resume File']}`
    - **Predicted Category:** `{top_row['Predicted Category']}`
    - **Similarity Score:** `{top_row['Score']}`
    """)

    with st.expander("📊 Full Results Table"):
        st.dataframe(results[['Resume File', 'Score', 'Rank', 'Predicted Category']], use_container_width=True)

