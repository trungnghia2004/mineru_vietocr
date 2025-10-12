import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
import uvicorn

import sys
import pypdfium2 as pdfium
from mineru.cli.common import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

app = FastAPI(
    title="MinerU VietOCR API", 
    version="1.0.0",
    description="API for processing PDFs and images with OCR using MinerU and VietOCR",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory=UPLOAD_FOLDER), name="output")

# Templates
templates = Jinja2Templates(directory="templates")

class ProcessPdfLinkRequest(BaseModel):
    url: HttpUrl

def update_markdown_image_paths(markdown_content: str, base_url: str, pdf_name: str) -> str:
    """
    Update markdown content to use static file serving for images.
    Converts relative image paths to static file URLs.
    """
    if not markdown_content:
        return markdown_content
    
    # Pattern to match markdown image syntax: ![](path/to/image.jpg)
    image_pattern = r'!\[\]\(([^)]+)\)'
    
    def replace_image_path(match):
        image_path = match.group(1)
        # If it's already a full URL, don't modify it
        if image_path.startswith('http'):
            return match.group(0)
        
        # Convert relative path to static file URL
        # The image paths in markdown are like "images/filename.jpg"
        # But the actual files are at "{pdf_name}/auto/images/filename.jpg"
        if image_path.startswith('images/'):
            # Remove the "images/" prefix and add the full path
            image_filename = image_path[7:]  # Remove "images/" prefix
            static_url = f"/output/{pdf_name}/auto/images/{image_filename}"
            return f"![]({static_url})"
        
        # For other cases, assume it's a direct filename in the images directory
        static_url = f"/output/{pdf_name}/auto/images/{image_path}"
        return f"![]({static_url})"
    
    return re.sub(image_pattern, replace_image_path, markdown_content)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def _build_response_payload(output_files: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    response_data: Dict[str, Dict[str, str]] = {}
    for pdf_name, files in output_files.items():
        response_data[pdf_name] = {}
        for file_type, file_path in files.items():
            if file_type in ["markdown", "content_list", "middle_json", "model_output"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if file_type == "markdown":
                    content = update_markdown_image_paths(content, "", pdf_name)
                    return {"markdown": content}

                response_data[pdf_name][file_type] = content
            else:
                rel_path = os.path.relpath(file_path, UPLOAD_FOLDER)
                response_data[pdf_name][file_type] = rel_path

    return {
        "status": "success",
        "files": response_data,
        "message": "PDF processed successfully. Images are accessible via /output/ endpoints.",
        "download_base": "/download/",
        "static_base": "/output/"
    }

def _ensure_pdf_extension(path: str) -> str:
    if not path.lower().endswith(".pdf"):
        return f"{path}.pdf"
    return path

def _resolve_arxiv_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    if netloc.endswith("arxiv.org"):
        path = parsed.path
        if path.startswith("/abs/"):
            pdf_path = _ensure_pdf_extension(path.replace("/abs/", "/pdf/", 1))
            parsed = parsed._replace(path=pdf_path, query="", fragment="")
        elif path.startswith("/pdf/"):
            pdf_path = _ensure_pdf_extension(path)
            parsed = parsed._replace(path=pdf_path, query="", fragment="")
        return urlunparse(parsed)

    return url

def _extract_filename_from_headers(url: str, headers: Dict[str, str]) -> str:
    content_disposition = headers.get("content-disposition")
    if content_disposition:
        matches = re.findall('filename="?([^";]+)"?', content_disposition)
        if matches:
            return matches[-1]

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if filename:
        return filename
    return "downloaded.pdf"

from requests import get
def _download_pdf(url: str) -> Tuple[bytes, str]:
    normalized_url = url.strip()
    if not normalized_url:
        raise HTTPException(status_code=400, detail="URL must not be empty")

    resolved_url = _resolve_arxiv_pdf_url(normalized_url)

    response = get(resolved_url, timeout=60, verify=False)
    response.raise_for_status()

    pdf_bytes = response.content

    filename = _extract_filename_from_headers(resolved_url, response.headers)
    if not filename.lower().endswith(".pdf"):
        filename = f"{Path(filename).stem}.pdf"

    return pdf_bytes, filename

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No selected file")

        filename = file.filename
        if not filename.lower().endswith(tuple(pdf_suffixes + image_suffixes)):
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF or images are allowed.")

        # Check file size
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

        file_suffix = f".{filename.rsplit('.', 1)[-1].lower()}"
        pdf_bytes = read_fn(file_bytes, file_suffix)
        pdf_file_name = Path(filename).stem

        output_files = process_pipeline(
            output_dir=UPLOAD_FOLDER,
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            parse_method="auto",
            p_formula_enable=True,
            p_table_enable=True,
            f_draw_layout_bbox=True,
            f_draw_span_bbox=True,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=True,
            f_dump_content_list=True,
            f_make_md_mode=MakeMode.MM_MD
        )

        return JSONResponse(content=_build_response_payload(output_files))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_pdf_link/")
async def process_pdf_link(payload: ProcessPdfLinkRequest):
    try:
        pdf_bytes, filename = _download_pdf(str(payload.url))
        pdf_file_name = Path(filename).stem

        output_files = process_pipeline(
            output_dir=UPLOAD_FOLDER,
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[read_fn(pdf_bytes, ".pdf")],
            p_lang_list=["ch"],
            parse_method="auto",
            p_formula_enable=True,
            p_table_enable=True,
            f_draw_layout_bbox=True,
            f_draw_span_bbox=True,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=True,
            f_dump_content_list=True,
            f_make_md_mode=MakeMode.MM_MD
        )

        return JSONResponse(content=_build_response_payload(output_files))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{rel_path:path}")
async def download_file(rel_path: str):
    try:
        full_path = os.path.join(UPLOAD_FOLDER, rel_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(UPLOAD_FOLDER)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(full_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Static files are now handled by the mounted StaticFiles
# This endpoint is no longer needed as FastAPI's StaticFiles mount handles it

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8088)