import os
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "amansomvanshi36/nagrik-setu-lora")

app = FastAPI(
    title="Nagrik Setu AI",
    version="1.0",
    description="AI civic complaint classification API"
)

class Complaint(BaseModel):
    title: str
    description: str
    latitude: float | None = None
    longitude: float | None = None

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading Qwen base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    print("Loading Nagrik Setu LoRA...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("Nagrik Setu AI model loaded successfully.")

@app.get("/")
def home():
    return {
        "message": "Nagrik Setu AI is running",
        "model": "Qwen2.5-3B-Instruct + Nagrik Setu LoRA"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "service": "Nagrik Setu AI",
        "model": "Qwen2.5-3B-Instruct + LoRA"
    }

@app.post("/classify")
def classify_complaint(complaint: Complaint):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="AI model is still loading")

    prompt = f"""Classify this civic complaint for Nagrik Setu.

Title: {complaint.title}
Description: {complaint.description}
Latitude: {complaint.latitude}
Longitude: {complaint.longitude}

Return ONLY valid JSON with exactly these keys:
category
sub_category
department
is_emergency
verification_status
confidence
reason
recommended_action
"""

    messages = [
        {
            "role": "system",
            "content": "You are Nagrik Setu, an AI civic complaint classifier. Return only valid JSON."
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    start = result_text.find("{")
    end = result_text.rfind("}")

    if start == -1 or end == -1:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "AI did not return valid JSON",
                "raw_output": result_text
            }
        )

    try:
        return json.loads(result_text[start:end + 1])
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "AI returned invalid JSON",
                "raw_output": result_text
            }
        )
