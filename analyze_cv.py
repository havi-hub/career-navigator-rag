import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pypdf import PdfReader
from pydantic import BaseModel

load_dotenv()

parser = argparse.ArgumentParser(description="Analyze a resume against job descriptions.")
parser.add_argument(
    "resume",
    nargs="?",
    default="data/my_resume/havi_resume.pdf",
    help="Path to the resume PDF (default: data/my_resume/havi_resume.pdf)",
)
args = parser.parse_args()

# Load FAISS index
index_path = "faiss_index"
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Extract resume text
reader = PdfReader(args.resume)
resume_text = "".join(page.extract_text() or "" for page in reader.pages)

# Find most similar job
results = vector_store.similarity_search(resume_text, k=1)
if not results:
    print("No similar job description found.")
    exit(1)

matched_job = results[0]
print(f"Matched job: {matched_job.metadata.get('title', 'Unknown')}")
print("-" * 80)


class CareerAnalysis(BaseModel):
    matching_skills: list[str]
    missing_skills: list[str]
    rewritten_summary: str


llm = ChatOpenAI(model="gpt-4o").with_structured_output(CareerAnalysis)

prompt = f"""You are an expert career coach and resume analyst.

Resume:
---
{resume_text}
---

Job Description:
---
{matched_job.page_content}
---

Return exactly:
- matching_skills: top 3 skills the resume shares with the job
- missing_skills: top 3 skills or gaps missing from the resume
- rewritten_summary: a professional summary paragraph tailored to this job
"""

analysis: CareerAnalysis = llm.invoke(prompt)

print("Matching Skills:")
for i, skill in enumerate(analysis.matching_skills, 1):
    print(f"  {i}. {skill}")

print("\nMissing Skills:")
for i, skill in enumerate(analysis.missing_skills, 1):
    print(f"  {i}. {skill}")

print("\nRewritten Summary:")
print(analysis.rewritten_summary)
