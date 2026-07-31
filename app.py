import os
import fitz
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Groq API Key
# -------------------------
load_dotenv()

client = Groq(
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
)


# -------------------------
# Load Embedding Model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Read PDF
# -------------------------
pdf_path = "documents/Resume.pdf.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

doc.close()


# -------------------------
# Split into Chunks
# -------------------------
chunk_size = 300

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])


print("Total Chunks:", len(chunks))


# -------------------------
# Create Embeddings
# -------------------------
embeddings = model.encode(chunks)

embedding_array = np.array(embeddings).astype("float32")


# -------------------------
# Create FAISS Index
# -------------------------
index = faiss.IndexFlatL2(
    embedding_array.shape[1]
)

index.add(embedding_array)


print("FAISS Index Created!")
print("Total Chunks in Index:", index.ntotal)



# -------------------------
# User Question Loop
# -------------------------
while True:

    question = input("\nAsk a question (type exit to stop): ")

    if question.lower() == "exit":
        break


    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # Retrieve top 3 relevant chunks
    distance, index_result = index.search(
        question_embedding,
        k=3
    )


    best_chunk = "\n".join(
        [chunks[i] for i in index_result[0]]
    )


    print("\nRetrieved Context:\n")
    print(best_chunk)



    # -------------------------
    # Ask Groq LLM
    # -------------------------
    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content":
                """
                You are a helpful AI assistant.

                Answer ONLY using the provided document context.

                If the answer is not available in the context,
                reply:
                "I couldn't find that information in the document."
                """
            },

            {
                "role": "user",
                "content": f"""
                Context:
                {best_chunk}

                Question:
                {question}
                """
            }
        ]
    )


    print("\nAnswer:\n")

    print(
        response.choices[0].message.content
    )