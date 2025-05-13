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

def preprocess_resume(resume_text):
    """
    Preprocess resume text through the following steps:
    1. Load resume data from CSV files.
    2. Clean and standardize text:
       - Convert to lowercase
       - Remove numbers and punctuation
       - Tokenize text
       - Remove stop words
       - Apply lemmatization
    """

    # Step 3: Convert text to lowercase
    text = resume_text.lower()

    # Step 4: Remove punctuation and numbers
    text = re.sub(r'[^\w\s]|[\d]', '', text)

    # Step 5: Tokenize text into words
    tokens = word_tokenize(text)

    # Step 6 & 7: Remove stop words and apply lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    return ' '.join(cleaned_tokens)
    
# Function to extract skill, education, and work experience-related lines from resume text
def extract_skills_edu_exp(resume_text):
    """Function to extract skill, education, and work experience-related lines from resume text"""
    # Define a set of skill-related keywords
    skills_keywords = {
        "skills", "skill", "technical skills", "soft skills", "tools", "technologies", "technology",
        "frameworks", "libraries", "platforms", "languages", "certifications", "methodologies",
        "programming", "databases", "cloud", "devops", "analytics", "testing tools", "networking"
    }

    # Define a comprehensive set of education-related keywords
    education_keywords = {
        # General education terms
        "bachelor", "bachelors", "master", "masters", "phd", "degree", "degrees", "university", "college",
        "graduate", "postgraduate", "undergraduate", "school", "education", "certification", "certifications", "diploma",

        # Common degree/program abbreviations
        "b.tech", "be", "bsc", "bca", "bba", "ba", "mba", "mca", "m.tech", "me", "msc", "ms", "pgdm", "pg", "ug", "llb", 
        "llm",

        # Data/Tech-specific
        "data science", "machine learning", "artificial intelligence", "computer science", "cs",
        "information technology", "it", "software engineering", "cybersecurity", "network security",
        "devops", "cloud computing", "aws", "azure", "gcp", "database", "big data", "hadoop", "etl",
        "python", "java", "dotnet", ".net", "blockchain", "web development", "web designing", "ui ux",
        "software testing", "automation testing", "qa", "full stack", "frontend", "backend",

        # Business/Management
        "business administration", "business analytics", "operations management", "operations",
        "human resources", "hr", "marketing", "finance", "accounting", "commerce", "economics",
        "entrepreneurship", "strategy", "organizational behavior", "project management", "pmp",

        # Engineering branches
        "mechanical engineering", "civil engineering", "electrical engineering", "electronics",
        "electronics and communication", "ece", "instrumentation", "engineering",

        # Legal/Advocate
        "law", "legal studies", "advocate", "llb", "llm", "jurisprudence", "judicial", "legal",

        # Arts & Health
        "liberal arts", "fine arts", "performing arts", "visual arts", "design", "health sciences",
        "healthcare", "nursing", "medicine", "fitness", "physical education", "nutrition",

        # Additional soft skill / professional-related
        "communication", "leadership", "pmo", "analyst", "consultant", "trainer", "coach"
    }

    # Define work-related keywords
    work_keywords = {
        "experience", "exprience", "worked", "employed", "company", "organization", "intern", "internship"
    }

    # Compile regex patterns for matching categories
    skills_pattern = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)

    # Lists to store matched lines
    skills = []
    education = []
    experience = []

    # Split text into lines to process individually
    lines = resume_text.split('\n')

    for line in lines:
        sentence = line.strip()
        lower = sentence.lower()

        # Match and collect skill-related lines
        if skills_pattern.search(lower):
            skills.append(sentence)

        # Match and collect education-related lines
        if edu_pattern.search(lower):
            education.append(sentence)

        # Match and collect work experience-related lines
        if work_pattern.search(lower):
            experience.append(sentence)

    # Return results as a Pandas Series (suitable for applying to DataFrames)
    return skills + education +  experience

# Page config
st.set_page_config(page_title="Multi Resume Analyser", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📝 Resume Matching And Candidate Ranking</h1>", unsafe_allow_html=True)

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
    st.session_state.reset_uploader = 0  # used to reset file uploaders

# Reset Button
col1, col2, col3 = st.columns(3)
with col3:
    if st.button("🔁 Reset Application"):
        st.session_state.text_blocks = []
        st.session_state.file_names = []
        st.session_state.processed_files = set()
        st.session_state.submitted = False
        st.session_state.reset_uploader += 1  # triggers re-render of uploaders
        st.rerun()

# Dynamic keys for resetting
jd_uploader_key = f"jd_uploader_{st.session_state.reset_uploader}"
resume_uploader_key = f"resume_uploader_{st.session_state.reset_uploader}"

# Job Description upload
jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf", key=jd_uploader_key)
jd = ""
if jd_file is not None:
    try:
        reader = PdfReader(jd_file)
        jd_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                jd_text += page_text + "\n"
        jd = jd_text.strip()
        if jd:
            st.success("✅ Job Description extracted successfully.")
        else:
            st.warning("⚠️ No extractable text found in the PDF.")
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

# Resume upload
uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (PDF format)",
    type="pdf",
    accept_multiple_files=True,
    key=resume_uploader_key
)

# Extract resume text
if uploaded_files and not st.session_state.submitted:
    st.markdown("### 📄 Extracted Resumes from PDF:")
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
                st.success(f"✅ Extracted text from: {uploaded_file.name}")

# Submit button
if not st.session_state.submitted:
    if st.button("Submit"):
        if len(st.session_state.text_blocks) == 0:
            st.warning("Please upload at least one valid PDF with text.")
        elif not jd.strip():
            st.warning("Please enter a job description.")
        else:
            st.session_state.submitted = True
            st.rerun()

# Results
if st.session_state.submitted and jd:
    st.markdown("## ✅ Submission Complete!")

    resumes = st.session_state.text_blocks
    file_names = st.session_state.file_names
    processed_resumes = []
    for resume in resumes:
        processed_temp_resume = preprocess_resume(resume)
        processed_resumes.append(extract_skills_edu_exp(processed_temp_resume))
        
    # SBERT embeddings
    embeddings = sbert_model.encode(processed_resumes)
    jd_embedding = sbert_model.encode([jd])

    # Similarity calculation
    similarities = cosine_similarity(jd_embedding, embeddings)[0]

    # Predict categories
    predicted_labels = model.predict(embeddings)
    predicted_categories = tokenizer.inverse_transform(predicted_labels)

    # Create DataFrame
    df = pd.DataFrame({
        "Resume File": file_names,
        "Index": list(range(len(resumes))),
        "Score": similarities,
        "Predicted Category": predicted_categories
    })

    # Ranking
    df["Rank"] = df["Score"].rank(ascending=False, method='first').astype(int)
    df = df.sort_values(by="Rank")

    # Best match category
    top_resume_embedding = embeddings[df.iloc[0]["Index"]].reshape(1, -1)
    top_pred = model.predict(top_resume_embedding)
    top_category = tokenizer.inverse_transform(top_pred)[0]
    st.success(f"🎯 Predicted Job Category for Best Match: **{top_category}**")

    # Show results
    st.markdown("### 📊 Resume Analysis Results:")
    st.dataframe(df[['Resume File', 'Score', 'Rank', 'Predicted Category']], width=700)
