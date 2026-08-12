# MinerU VietOCR Integration

A comprehensive document processing system that combines **MinerU** (a powerful PDF/document analysis framework) with **VietOCR** (Vietnamese OCR engine) to provide advanced document parsing, text extraction, and OCR capabilities with a modern web interface.

## 🚀 Features

### Core Functionality
- **PDF Document Processing**: Advanced PDF parsing with layout analysis, text extraction, and image processing
- **Multi-format Support**: Handles PDF, PNG, JPEG, JPG, WebP, and GIF files
- **Vietnamese OCR**: Specialized OCR engine optimized for Vietnamese text recognition
- **Layout Analysis**: Intelligent document structure detection including:
  - Text regions and paragraphs
  - Tables and formulas
  - Images and figures
  - Reading order detection
- **Web Interface**: Modern, responsive web UI with dark theme
- **API Endpoints**: RESTful API for programmatic access

### Advanced Features
- **Formula Recognition**: Mathematical formula detection and parsing
- **Table Extraction**: Structured table data extraction
- **Bounding Box Visualization**: Visual representation of detected regions
- **Multiple Output Formats**: Markdown, JSON, content lists, and annotated PDFs
- **Batch Processing**: Support for processing multiple documents
- **Label Studio Integration**: Tools for converting Label Studio annotations to VietOCR training format

## 📁 Project Structure

```
mineru_vietocr/
├── app.py                          # Main Flask web application
├── API.py                          # Alternative API implementation
├── test.py                         # API testing script
├── templates/
│   └── index.html                  # Web interface template
├── mineru/                         # MinerU document processing framework
│   ├── backend/pipeline/           # Core processing pipeline
│   ├── model/                      # AI models (layout, OCR, table, formula)
│   ├── utils/                      # Utility functions
│   └── cli/                        # Command-line interface
├── vietocr/                        # Vietnamese OCR engine
│   ├── vietocr/                    # Core OCR implementation
│   │   ├── model/                  # OCR models (Transformer, CNN)
│   │   ├── tool/                   # Prediction and training tools
│   │   └── loader/                 # Data loading utilities
│   └── setup.py                    # Package configuration
└── output/                         # Processed document outputs
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for better performance)
- 8GB+ RAM (16GB+ recommended)

### Dependencies

The project requires several key dependencies:

```bash
pip install -r requirements.txt
```

### Setup

1. **Clone the repository**:
   ```bash
   Sao chep ma nguon vao may va mo thu muc du an local.
   cd mineru_vietocr
   ```

2. **Install dependencies**:
    ```bash
    conda create -n <environment name> python=3.10
    pip install -r requirements.txt
    ```

2. **Download required models** (models will be downloaded automatically on first use):
   - Layout detection models
   - OCR models
   - Formula recognition models
   - Table extraction models

## 🚀 Usage

### Web Interface

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Access the interface**:
   Open your browser and navigate to `http://localhost:8000`

3. **Upload and process documents**:
   - Click "Chọn File" to select a PDF or image file
   - Click "Bắt đầu Xử lý" to start processing
   - View results in multiple formats (Markdown, JSON, etc.)

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
No authentication required for current implementation.

### Endpoints

#### 1. Process Document
**POST** `/process_pdf/`

Process a PDF or image file and extract text, tables, formulas, and layout information.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (PDF, PNG, JPEG, JPG, WebP, or GIF file)

**Response:**
```json
{
  "status": "success",
  "files": {
    "document_name": {
      "markdown": "# Document Title\n\nExtracted content...",
      "content_list": ["item1", "item2", "..."],
      "middle_json": {...},
      "model_output": {...},
      "annotated_pdf": "path/to/annotated.pdf",
      "original_pdf": "path/to/original.pdf"
    }
  },
  "message": "PDF processed successfully. Images are accessible via /output/ endpoints.",
  "download_base": "/download/",
  "static_base": "/output/"
}
```

**Example Usage:**
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/process_pdf/
```

#### 2. Download Processed Files
**GET** `/download/<path:rel_path>`

Download any processed file (PDFs, images, etc.).

**Parameters:**
- `rel_path`: Relative path to the file within the output directory

**Example Usage:**
```bash
curl -O http://localhost:8000/download/document_name/auto/document_name.pdf
```

#### 3. Serve Static Files (Images)
**GET** `/output/<path:filename>`

Access processed images and static assets.

**Parameters:**
- `filename`: Path to the image file within the output directory

**Example Usage:**
```bash
curl http://localhost:8000/output/document_name/auto/images/image1.jpg
```

#### 4. Health Check
**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

**Example Usage:**
```bash
curl http://localhost:8000/health
```

### API Usage Examples

#### Python
```python
import requests

# Process a document
url = "http://localhost:8000/process_pdf/"
files = {'file': open('document.pdf', 'rb')}

response = requests.post(url, files=files)
if response.status_code == 200:
    data = response.json()
    print("Markdown content:", data['files']['document_name']['markdown'])
    print("Download URL:", f"http://localhost:8000/download/{data['files']['document_name']['annotated_pdf']}")
```

#### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('document.pdf'));

axios.post('http://localhost:8000/process_pdf/', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Processed:', response.data);
})
.catch(error => {
  console.error('Error:', error.response.data);
});
```

#### cURL
```bash
# Process document
curl -X POST -F "file=@document.pdf" http://localhost:8000/process_pdf/

# Download processed PDF
curl -O http://localhost:8000/download/document_name/auto/document_name.pdf

# Access image
curl http://localhost:8000/output/document_name/auto/images/image1.jpg
```

### Error Responses

**400 Bad Request:**
```json
{
  "error": "No file part",
  "message": "No file part"
}
```

**404 Not Found:**
```json
{
  "error": "File not found",
  "message": "File not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Processing failed",
  "message": "Error details..."
}
```

## 🔧 Configuration

The system automatically configures models based on available hardware:
- **Device**: Auto-detects CUDA, CPU, or MPS
- **Language**: Defaults to Chinese ('ch') with Vietnamese support
- **Models**: Automatically downloads and caches required models

## 🛠️ Development Tools

### Testing
Test the API endpoints:
```bash
python test.py
```

## 🆘 Support

### Common Issues
1. **Model Download Failures**: Ensure stable internet connection
2. **Memory Issues**: Reduce batch size or use CPU mode
3. **CUDA Errors**: Check GPU compatibility and drivers

### Getting Help
- Check the local MinerU and VietOCR source bundled in this workspace.
- Open an issue for bugs or feature requests

## 📄 License

This project combines multiple open-source components:
- **MinerU**: Copyright (c) Opendatalab
- **VietOCR**: Apache 2.0 License
- **Integration Code**: See individual file headers
