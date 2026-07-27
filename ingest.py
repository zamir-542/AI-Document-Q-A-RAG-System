from pypdf import PdfReader
pdf_path = r"C:\Users\zamir\Desktop\projects\AI to Docs Project\ResumeZamir.pdf"
docs = PdfReader(pdf_path)
text = ""
for page in docs.pages:
    text += page.extract_text()
#print(text)
CHUNK=1000
OVERLAP=400

def chunk_text(text,chunk_size,chunk_overlap):
    chunks=[]
    for i in range(0,len(text),chunk_size-chunk_overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks
    
chunks = chunk_text(text,CHUNK,OVERLAP)
print(f"The number of tokens is: {len(chunks)}")
    
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks).tolist() 
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_or_create_collection(name="docs_collection")

collection.add(
    embeddings = vectors,
    documents = chunks,
    ids = [f"doc{i}" for i in range(len(chunks))]
)    
