from sentence_transformers import SentenceTransformer
import chromadb
model = SentenceTransformer("all-MiniLM-L6-v2")
vector_db = chromadb.PersistentClient(path="vectordb")
collection = vector_db.get_collection(name="docs_collection")
question=input("Enter your question: ")
query_vector = model.encode(question).tolist()
results = collection.query(
    query_embeddings = [query_vector],
    n_results = 5
)
#print(results["ids"])
context="\n".join(results["documents"][0])
#print(context)
prompt=f"""context: {context}
question: {question}
"""
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
client=genai.Client()
response=client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Answer to the question"
                }
            }
        }
    )
)
print("\n",response.text)
