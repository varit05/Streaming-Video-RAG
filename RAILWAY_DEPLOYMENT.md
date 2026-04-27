# Railway Production Deployment Guide

## Step-by-Step Deployment Instructions for Streaming-Video-RAG

---

## ✅ Pre-requisites

1. **Railway Account**: https://railway.app/
2. **Qdrant Cloud Account**: https://qdrant.tech/ (Recommended for production vector storage)
3. **OpenAI API Key** (or Anthropic / other LLM provider)
4. Git repository with this project

---

## 🚀 Deployment Steps

### 1. Create New Railway Project

1. Login to Railway Dashboard
2. Click **+ New Project**
3. Select **Deploy from GitHub repo**
4. Connect your GitHub account and select this repository
5. Select the branch you want to deploy (`main` / `master`)

---

### 2. Configure Railway Service

Railway will automatically detect the `Dockerfile` in your repository.

#### ⚙️ Required Service Configuration:

| Setting | Value |
|---------|-------|
| **Build Context** | Root directory |
| **Dockerfile Path** | `/Dockerfile` |
| **Port** | `8501` |
| **Healthcheck Path** | `/` |
| **Disk Size** | Minimum **10GB** (required for model downloads) |
| **Memory** | Minimum **4GB RAM** (8GB recommended for Whisper transcription) |

---

### 3. Set Production Environment Variables

Go to your Railway service → **Variables** tab and add:

```env
# ------------------------------
# Core Configuration
# ------------------------------
ENVIRONMENT=production
LOG_LEVEL=INFO

# ------------------------------
# API Configuration
# ------------------------------
API_HOST=0.0.0.0
API_PORT=8000

# ------------------------------
# LLM Provider
# ------------------------------
OPENAI_API_KEY=your-openai-api-key-here
# OR ANTHROPIC_API_KEY=your-anthropic-key

# ------------------------------
# Production Vector Store (Qdrant)
# ------------------------------
VECTOR_STORE_TYPE=qdrant
QDRANT_HOST=xxxxxx-xxxxx-xxxxx-xxxxx-xxxxxxxxx.us-east-1.aws.cloud.qdrant.io
QDRANT_PORT=6334
QDRANT_API_KEY=your-qdrant-cloud-api-key
QDRANT_HTTPS=true

# ------------------------------
# Database
# ------------------------------
DATABASE_URL=sqlite:///data/video_rag.db

# ------------------------------
# Transcription
# ------------------------------
WHISPER_MODEL=base
# Use 'small' / 'medium' if you have enough memory
```

> 💡 **Important**: Do NOT use ChromaDB on Railway for production. Qdrant Cloud is designed for managed production vector workloads.

---

### 4. Advanced Railway Settings

In your service **Settings** tab:

1. Enable **Public Network**
2. Generate a Railway domain (or configure custom domain)
3. Set **Start Command** (default from Dockerfile will work):
   ```bash
   streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
   ```

---

### 5. Deploy the Service

1. Click **Deploy**
2. Monitor build logs in Railway dashboard
3. First deployment will take ~5-8 minutes (dependency installation + ffmpeg setup)

---

### 6. Deploy the API Service (Optional)

To run the FastAPI backend as separate service:

1. Add another service in same Railway project
2. Same repository, same branch
3. Set **Custom Start Command**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. Set port `8000` and enable public network
5. Use same environment variables

---

## 📋 Post Deployment Checklist

- [ ] Service builds successfully
- [ ] Application loads on Railway domain
- [ ] Qdrant connection is established
- [ ] LLM API calls work correctly
- [ ] Video ingestion functionality tested
- [ ] Semantic search returns results
- [ ] Database is persisted correctly

---

## ⚡ Railway Optimizations

### Enable Build Cache
Add `railway.toml` file to your repository root:
```toml
[build]
  cache = true
  buildpacks = []
```

### Memory Optimizations
For best performance on Railway standard instances:
- Use `WHISPER_MODEL=base` for transcription
- Disable GPU acceleration
- Use OpenAI embeddings instead of local sentence-transformers
- Configure 8GB RAM instance for reliable operation

---

## 🔒 Security Best Practices

1. Never commit `.env` files to repository
2. All secrets are stored exclusively in Railway Variables
3. Use Railway's built-in environment encryption
4. Enable HTTPS only (default on Railway domains)
5. Restrict CORS origins in production

---

## 📊 Monitoring & Logs

- View realtime logs directly in Railway dashboard
- Enable Railway Metrics for performance monitoring
- Configure alerts for service downtime
- Monitor memory usage - this application is memory intensive

---

## ❌ Common Issues

| Problem | Solution |
|---------|----------|
| Build fails with out of memory | Upgrade service to 8GB RAM plan |
| Qdrant connection errors | Verify `QDRANT_HTTPS=true` is set |
| Whisper transcription crashes | Reduce whisper model size |
| Streamlit timeout | Increase Railway service timeout settings |

---

## ✅ Deployment Complete

Your Streaming Video RAG system will now be running on Railway at:
`https://your-service-name.up.railway.app`

For support see Railway documentation: https://docs.railway.app/