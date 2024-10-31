from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@lru_cache(maxsize=100)
def cached_response(prompt: str) -> str:
    context = "You are a helpful assistant."
    full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
    
    input_ids = tokenizer.encode(full_prompt + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

@app.post("/query", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    try:
        response_text = cached_response(request.query)
        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while generating the response.")