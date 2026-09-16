# Nagrik Setu AI

FastAPI service for Nagrik Setu complaint classification.

## Endpoints
- GET `/`
- GET `/health`
- POST `/classify`

## Model
Qwen/Qwen2.5-3B-Instruct + Nagrik Setu LoRA adapter.

## Render
Build with Docker and expose port 10000.
