## Setup Instructions

### Production / Web App (Render)
1. Push this repo and deploy with the included `Dockerfile`
2. The `Dockerfile` runs `uvicorn app:app --host 0.0.0.0 --port 8000` and `EXPOSE`s 8000
3. Note: the DB contains the `.npy`/image data — do not remove them

### Development
`pip install -r requirements-dev.txt`
`uvicorn app:app --reload`
