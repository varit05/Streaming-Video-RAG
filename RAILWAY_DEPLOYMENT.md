# ✅ FINAL RAILWAY DEPLOYMENT GUIDE
## Production Verified - All Edge Cases Fixed

This guide contains every required step with no omissions. Follow exactly.

---

## 🚀 SERVICE 1: FASTAPI BACKEND

### ✅ Railway Dashboard Settings:
| Setting | Value |
|---|---|
| Name | `fastapi-api` |
| Memory | **8GB** |
| Disk | **10GB Volume mounted at /app/data** |
| Start Command | **SET THIS ONLY IN DASHBOARD NOT IN TOML:** |
| | `uvicorn api.main:app --host :: --port $PORT --workers 2 --timeout-keep-alive 600` |
| Healthcheck Path | `/health` |
| Public Network | ✅ Enabled |
| Always On | ✅ Enabled |
| Zero Downtime Deployments | ❌ Disabled |

### ✅ Environment Variables:
```env
ENVIRONMENT=production
WHISPER_MODEL=tiny
OPENAI_API_KEY=xxx
QDRANT_HOST=xxx
QDRANT_API_KEY=xxx
QDRANT_HTTPS=true
```

---

## 🚀 SERVICE 2: STREAMLIT FRONTEND

### ✅ Railway Dashboard Settings:
| Setting | Value |
|---|---|
| Name | `streamlit-ui` |
| Memory | **4GB** |
| Disk | Same shared 10GB Volume at /app/data |
| Start Command | **SET THIS ONLY IN DASHBOARD NOT IN TOML:** |
| | `streamlit run ui/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true --server.timeout=300 --server.maxUploadSize=95` |
| Healthcheck Path | `/` |
| Public Network | ✅ Enabled |
| Always On | ✅ Enabled |
| Zero Downtime Deployments | ❌ Disabled |

### ✅ Environment Variables:
```env
ENVIRONMENT=production
API_URL=https://<your-fastapi-domain>.up.railway.app
```

---

## 🔧 REQUIRED POST-DEPLOYMENT STEPS:

1.  **Open Railway Support Ticket**: Request to increase proxy timeout to 600 seconds for both services
2.  **First Deploy Order**: Deploy API service first, wait until it is fully running before deploying UI
3.  **Verify Health**: Check `/health` endpoint on API service returns 200 OK
4.  **Test End To End**: Upload a small 1 minute video to confirm full pipeline works

---

## ❌ NEVER DO THESE:
- ❌ Do NOT hardcode PORT in any config
- ❌ Do NOT set start command in railway.toml (Railway has bug with $PORT expansion)
- ❌ Do NOT use whisper model larger than tiny on standard instances
- ❌ Do NOT use more than 2 uvicorn workers
- ❌ Do NOT enable zero downtime deployments
- ❌ Do NOT forget to mount persistent volume

---

## ✅ Final Verification Checklist:

- [ ] Volume mounted successfully at /app/data
- [ ] API /health endpoint returns 200 OK
- [ ] Streamlit UI loads without errors
- [ ] You can navigate all tabs
- [ ] You can upload a small video file
- [ ] Transcription completes successfully
- [ ] Semantic search returns results
- [ ] Q&A functionality answers questions

This deployment will run reliably 24/7 on Railway.