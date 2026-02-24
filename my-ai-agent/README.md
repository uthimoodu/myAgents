# AI Agent — Full Stack Deployment Guide

## Project Structure
```
project/
├── backend/
│   ├── main.py            ← FastAPI app (deploy to Railway)
│   ├── requirements.txt
│   ├── Procfile
│   └── .env.example
└── frontend/
    └── index.html         ← Chat UI (deploy to Vercel)
```

---

## Step 1 — Get API Keys

| Service     | URL                          | Free Tier |
|-------------|------------------------------|-----------|
| Groq        | https://console.groq.com     | ✅ Yes    |
| Tavily      | https://tavily.com           | ✅ Yes    |
| WeatherAPI  | https://weatherapi.com       | ✅ Yes    |

---

## Step 2 — Deploy Backend to Railway

1. Go to https://railway.app and create a new project
2. Connect your GitHub repo (push the `backend/` folder)
3. Railway will auto-detect the `Procfile` and run:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Add these environment variables in Railway dashboard → Variables:
   ```
   WEATHER_API_KEY=your_key
   TAVILY_API_KEY=your_key
   GROQ_API_KEY=your_key
   ```
5. After deploy, copy your Railway URL e.g.:
   ```
   https://your-app.up.railway.app
   ```
6. Test it:
   ```
   curl https://your-app.up.railway.app/health
   ```

---

## Step 3 — Connect Frontend to Backend

Open `frontend/index.html` and update line:
```javascript
const BACKEND_URL = "https://your-railway-app.up.railway.app";
```
Replace with your actual Railway URL.

---

## Step 4 — Deploy Frontend to Vercel

1. Go to https://vercel.com and create a new project
2. Upload the `frontend/` folder OR connect GitHub repo
3. Vercel auto-deploys `index.html` — no build step needed
4. Your frontend is live at:
   ```
   https://your-app.vercel.app
   ```

---

## API Endpoints

| Method | Endpoint  | Description              |
|--------|-----------|--------------------------|
| GET    | /         | Root health check        |
| GET    | /health   | Railway health check     |
| POST   | /chat     | Send a question to agent |

### POST /chat Example
```json
Request:
{ "question": "What is the weather in London?" }

Response:
{
  "question": "What is the weather in London?",
  "answer": "The current weather in London is...",
  "tools_used": ["get_weather"]
}
```

---

## Local Development

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in browser and set:
```javascript
const BACKEND_URL = "http://localhost:8000";
```
