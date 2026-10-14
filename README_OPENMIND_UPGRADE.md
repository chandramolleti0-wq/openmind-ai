# OpenMind AI Upgrade

This bundle upgrades the supplied `app.py` while preserving the existing FastAPI + Phi-3 ONNX Runtime GenAI setup.

## Included improvements

- Overlapping text chunks for document retrieval, with ranked chunk snippets rather than entire documents.
- Persistent chat sessions stored locally in `data/chat_history.json`.
- Bounded `/agent/run` research workflow using read-only web search and saved-document search, followed by local-model synthesis.
- Basic request timing/error counters in `data/metrics.json` exposed through `/metrics`.
- New session-history endpoints: `/chat/sessions`, `/chat/{session_id}` and `DELETE /chat/{session_id}`.
- Corrected startup banner for port 8001.

The agent intentionally does **not** run shell commands, modify project files, or take other side-effecting actions. Web search availability depends on the installed `ddgs` package and network access. PDF extraction is text-based; scanned PDFs still require OCR.

## Apply safely

1. Stop the OpenMind server with `Ctrl+C` in the terminal where Uvicorn is running.
2. In your project folder, make a backup copy of the existing `app.py`, for example `app_before_upgrade.py`.
3. Replace your project's `app.py` with the included upgraded `app.py`.
4. Compare/merge `requirements.txt`; the supplied requirements preserve the packages used by the current app.
5. Keep your existing `models` and `data` folders. Do not delete `data/knowledge.json`, API key files, or model files.
6. Start the server from the project folder:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001 --log-level debug
   ```

7. Open `http://127.0.0.1:8001`.

## Test the new endpoints

Use your existing key from `data/FIRST_RUN_API_KEY.txt`. In PowerShell, set it for the current terminal (paste the key between the quotes):

```powershell
$env:OPENMIND_KEY = "PASTE_YOUR_LOCAL_API_KEY_HERE"
$headers = @{ "X-API-Key" = $env:OPENMIND_KEY }
```

List chat sessions:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/chat/sessions -Headers $headers
```

View metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/metrics -Headers $headers
```

Run a bounded research agent:

```powershell
$body = @{
  goal = "Summarize the key points about retrieval-augmented generation and list the sources"
  use_web = $true
  use_knowledge = $true
  max_steps = 3
  session_id = "research-test"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8001/agent/run -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

Chat sessions and metrics are stored locally in the `data` directory. Keep that directory private because it may contain your documents and chat content. The app is configured for local use at `127.0.0.1`; do not expose it to the public internet without additional authentication, deployment, and security review.
