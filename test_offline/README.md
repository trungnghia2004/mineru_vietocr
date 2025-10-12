# Offline Docker Deployment Guide for MinerU VietOCR

This guide helps you build a Docker image with all models pre-downloaded for deployment on servers without internet access.

## 📋 Prerequisites

- **Machine with internet access** (for Phase 1 & 2)
- Docker installed on both machines
- Approximately 15-20 GB free disk space
- Python 3.8+ with `huggingface_hub` installed

## 🚀 Phase 1: Download Models (On Internet-Connected Machine)

### Step 1: Install HuggingFace Hub

```bash
pip install huggingface_hub
```

### Step 2: Download All Models

Navigate to the `test_offline` directory and run:

```bash
cd test_offline
python download_models.py
```

**Expected output:**
- Creates `models_cache/pipeline/` directory
- Downloads ~5-10 GB of models
- Shows completion message with directory structure

**What gets downloaded:**
- Layout detection model (YOLO)
- Formula detection model (MFD)
- Formula recognition model (MFR/Unimernet)
- OCR models (PaddleOCR)
- Table recognition models (SLANet, UNet)
- Table classification model
- Orientation classification model

## 🔨 Phase 2: Build Docker Image (On Internet-Connected Machine)

### Step 3: Verify Structure

Ensure your project structure looks like this:

```
mineru_vietocr/
├── test_offline/
│   ├── download_models.py
│   ├── Dockerfile.offline
│   ├── mineru.json
│   ├── models_cache/
│   │   └── pipeline/
│   │       └── models/
│   │           ├── Layout/
│   │           ├── MFD/
│   │           ├── MFR/
│   │           ├── OCR/
│   │           ├── TabRec/
│   │           ├── TabCls/
│   │           └── OriCls/
│   └── README.md
├── mineru/
├── requirements.txt
├── app.py
└── ... (other project files)
```

### Step 4: Build the Docker Image

From the **project root** (not inside test_offline):

```bash
docker build -f test_offline/Dockerfile.offline -t mineru-vietocr-offline:latest .
```

**Build time:** 10-30 minutes depending on your machine
**Image size:** ~10-15 GB

### Step 5: Save Docker Image to File

```bash
docker save mineru-vietocr-offline:latest -o mineru-vietocr-offline.tar
```

This creates a portable `.tar` file containing the entire Docker image with all models.

**File size:** ~10-15 GB (compressed might be smaller)

## 📦 Phase 3: Transfer to Offline Server

### Step 6: Transfer the Image

Transfer `mineru-vietocr-offline.tar` to your offline server using:
- USB drive
- SCP/SFTP (if accessible via internal network)
- Any other file transfer method

```bash
# Example using SCP (if internal network allows)
scp mineru-vietocr-offline.tar user@offline-server:/path/to/destination/
```

## 🖥️ Phase 4: Deploy on Offline Server

### Step 7: Load the Docker Image

On the offline server:

```bash
docker load -i mineru-vietocr-offline.tar
```

**Verify the image is loaded:**

```bash
docker images | grep mineru-vietocr-offline
```

### Step 8: Run the Container

**Basic run:**

```bash
docker run -d \
  --name mineru-vietocr \
  -p 8088:8088 \
  mineru-vietocr-offline:latest
```

**With volume mounting for outputs:**

```bash
docker run -d \
  --name mineru-vietocr \
  -p 8088:8088 \
  -v $(pwd)/output:/app/output \
  mineru-vietocr-offline:latest
```

**With GPU support (if available):**

```bash
docker run -d \
  --name mineru-vietocr \
  --gpus all \
  -p 8088:8088 \
  -v $(pwd)/output:/app/output \
  mineru-vietocr-offline:latest
```

## ✅ Verification

### Step 9: Check Container Logs

```bash
docker logs mineru-vietocr
```

**Look for:**
- ✅ `model_source: local` (confirms using local models)
- ✅ `DocAnalysis init, this may take some times......`
- ✅ `DocAnalysis init done!`
- ❌ NO download attempts or HuggingFace errors

### Step 10: Test the API

```bash
curl http://localhost:8088/health
```

Or test with a PDF file:

```bash
curl -X POST http://localhost:8088/process \
  -F "file=@sample.pdf" \
  -F "lang=vi"
```

## 🔧 Troubleshooting

### Models Not Found Error

**Symptom:** Error messages about missing models

**Solution:**
1. Check if models were copied correctly:
   ```bash
   docker exec mineru-vietocr ls -la /app/models/pipeline/models/
   ```
2. Verify environment variable:
   ```bash
   docker exec mineru-vietocr env | grep MINERU
   ```
   Should show: `MINERU_MODEL_SOURCE=local`

### Configuration File Not Found

**Symptom:** Warning about `mineru.json` not found

**Solution:**
```bash
docker exec mineru-vietocr cat /app/mineru.json
```
Should display the JSON configuration.

### Container Won't Start

**Check logs:**
```bash
docker logs mineru-vietocr
```

**Restart container:**
```bash
docker restart mineru-vietocr
```

### Out of Memory

**Symptom:** Container crashes or OOM errors

**Solution:** Increase Docker memory limit or use fewer workers:
```bash
docker run -d \
  --name mineru-vietocr \
  -p 8088:8088 \
  -e UVICORN_WORKERS=1 \
  --memory="8g" \
  mineru-vietocr-offline:latest
```

## 📊 Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8-16 GB |
| Disk Space | 15 GB | 25 GB |
| CPU Cores | 2 | 4+ |
| GPU (optional) | - | NVIDIA with 4GB+ VRAM |

## 🔄 Updating Models

To update models in the future:

1. On internet-connected machine, re-run:
   ```bash
   python download_models.py
   ```
2. Rebuild Docker image (Step 4)
3. Save new image (Step 5)
4. Transfer and load on offline server (Steps 6-7)

## 📝 Important Notes

1. **Environment Variable is Critical:** `MINERU_MODEL_SOURCE=local` must be set, or the app will try to download models at runtime.

2. **Path Consistency:** The paths in `mineru.json` must match where models are copied in the Dockerfile.

3. **Model Completeness:** Ensure ALL models are downloaded before building. Missing models will cause runtime errors.

4. **Docker Build Context:** Always build from the project root, not from inside `test_offline/`.

5. **Image Size:** The final image is large (~10-15 GB). Plan for sufficient transfer time and storage.

## 🆘 Support

If you encounter issues:

1. Check container logs: `docker logs mineru-vietocr`
2. Verify models are in the image: `docker exec mineru-vietocr ls -R /app/models/`
3. Test configuration: `docker exec mineru-vietocr cat /app/mineru.json`
4. Check environment: `docker exec mineru-vietocr env | grep MINERU`

## 📜 License

This deployment guide is part of the MinerU VietOCR project.

