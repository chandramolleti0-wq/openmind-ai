# OpenMind AI — web-connected knowledge assistant starter

This project gives you:
- A Python FastAPI server with your own API-key authentication
- Best-effort public web search without a paid search API key
- Website text extraction and a refresh endpoint
- PDF/TXT/MD uploads
- Local TF-IDF retrieval over saved knowledge
- A basic web interface and interactive API docs

## Important limitation
This starter's answer engine is retrieval-based, not a generative LLM. It returns relevant source snippets and links. That makes it deployable on modest free hosting without pretending that a large model will fit there. To get fluent AI-generated answers, add a small open-source model in a separate inference environment or run it locally. Free hosting has limits and can sleep; scheduled updates should use an external scheduler (for example GitHub Actions) or a platform cron service.

## Run on Windows
1. Install Python 3.11 or 3.12.
2. Extract the ZIP and open this folder in VS Code.
3. In the VS Code terminal:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
4. Open http://127.0.0.1:8000. The first-run API key is printed in the terminal and saved in `data/FIRST_RUN_API_KEY.txt`. Keep it private.
5. Open http://127.0.0.1:8000/docs for API testing.

## Use the API
Send the key in the HTTP header:
`Authorization: Bearer YOUR_KEY`

Ask:
```powershell
$headers = @{ Authorization = "Bearer YOUR_KEY" }
$body = @{ question = "Latest developments in AI"; web_search = $true; knowledge_search = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

Add a URL to the knowledge base:
```powershell
$body = @{ url = "https://www.python.org/" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/knowledge/url" -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

Refresh saved website pages:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/knowledge/refresh" -Method Post -Headers $headers
```

Upload a PDF:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/knowledge/upload" -H "Authorization: Bearer YOUR_KEY" -F "file=@C:\path\to\notes.pdf"
```

## Deploy free (API/retrieval layer)
1. Create a GitHub repository and upload the project files. Do not upload `data/`, `.env`, or any API key.
2. Create a free web service at Render and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the host dashboard:
   - `OPENMIND_API_KEY`: generate a long random key locally and paste it privately
   - `DATA_DIR`: `data`
6. Deploy and test `/health` and `/docs`.

**Persistence warning:** Render's free web service filesystem is ephemeral and can lose the knowledge database on restart/redeploy. For a real persistent knowledge base, use an external database/object store or keep a local copy and re-index it. Free database offerings can also have expiration or size limits. See the host's current plan details before relying on it.

## Automatic refresh
The `/knowledge/refresh` endpoint refreshes URLs you've already added. It does not automatically crawl the whole internet. For scheduled refreshes, use a scheduler to call that endpoint with the secret header, or use GitHub Actions with the URL stored as a repository secret. Do not commit the key.

## Security
- Treat your API key like a password.
- Use HTTPS for public deployments.
- Don't put your secret key in client-side JavaScript for a public app; this demo does so only to make local testing easy. For a public multi-user site, put authenticated calls behind a backend session.
- Only retrieve pages you are allowed to access and respect site terms and robots policies.
- Add rate limits and quotas before sharing publicly.
