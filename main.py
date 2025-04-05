import qdrant_client
import requests
import json


QDRANT_URL = "https://example.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "<<your qdrant api key>>"
API_KEY = "<<your deepseek api key>>"

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

prompt = """
What tools should I need to use to build a web service using vector embeddings for search？
"""

client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


client.set_model("BAAI/bge-base-en-v1.5")

client.add(
    collection_name="knowledge-base",
    # The collection is automatically created if it doesn't exist.
    documents=[
        "Qdrant is a vector database & vector similarity search engine. It deploys as an API service providing search for the nearest high-dimensional vectors. With Qdrant, embeddings or neural network encoders can be turned into full-fledged applications for matching, searching, recommending, and much more!",
        "Docker helps developers build, share, and run applications anywhere — without tedious environment configuration or management.",
        "PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.",
        "MySQL is an open-source relational database management system (RDBMS). A relational database organizes data into one or more data tables in which data may be related to each other; these relations help structure the data. SQL is a language that programmers use to create, modify and extract data from the relational database, as well as control user access to the database.",
        "NGINX is a free, open-source, high-performance HTTP server and reverse proxy, as well as an IMAP/POP3 proxy server. NGINX is known for its high performance, stability, rich feature set, simple configuration, and low resource consumption.",
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        "SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for semantic textual similar, semantic search, or paraphrase mining.",
        "The cron command-line utility is a job scheduler on Unix-like operating systems. Users who set up and maintain software environments use cron to schedule jobs (commands or shell scripts), also known as cron jobs, to run periodically at fixed times, dates, or intervals.",
    ]
)

def query_deepseek(prompt):
    data = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'stream': False
    }

    response = requests.post("https://api.deepseek.com/chat/completions", headers=HEADERS, data=json.dumps(data))

    if response.ok:
        result = response.json() 
        print(result['choices'][0]['message']['content'])  
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# query_deepseek(prompt)

results = client.query(
    collection_name="knowledge-base",
    query_text=prompt,
    limit=3,
)

context = "\n".join(r.document for r in results)

metaprompt = f"""
You are a software architect. 
Answer the following question using the provided context. 
If you can't find the answer, do not pretend you know it, but answer "I don't know".

Question: {prompt.strip()}

Context: 
{context.strip()}

Answer:
"""

def rag(question: str, n_points: int = 3) -> str:
    results = client.query(
        collection_name="knowledge-base",
        query_text=question,
        limit=n_points,
    )

    context = "\n".join(r.document for r in results)

    metaprompt = f"""
    You are a software architect. 
    Answer the following question using the provided context. 
    If you can't find the answer, do not pretend you know it, but only answer "I don't know".
    
    Question: {question.strip()}
    
    Context: 
    {context.strip()}
    
    Answer:
    """

    return query_deepseek(metaprompt)

rag("What can the stack for a web api look like?")
