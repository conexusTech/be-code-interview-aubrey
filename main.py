
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceHub
from functools import lru_cache

# Load environment variables from the .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Access the API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the HuggingFaceHub model with the API token
model = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.1, "max_length": 20}, huggingfacehub_api_token=api_token)


# Define request and response data models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# LRU cache to store responses for repeat queries
@lru_cache(maxsize=100)
def cached_response(query: str) -> str:
    # Apply basic prompt engineering
    prompt = f"Answer this in one sentence: {query}"
    return model(prompt)

# API endpoint to generate response
@app.post("/query", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    query = request.query
    try:
        response_text = cached_response(query)
        # response_text = response_text.split("\n")[0]

        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while generating the response.")

# To run the app: uvicorn main:app --reload
