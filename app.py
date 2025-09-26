import os
import re
from pathlib import Path
from flask import Flask, request, send_from_directory, jsonify, abort, render_template
from werkzeug.utils import secure_filename

import sys
import pypdfium2 as pdfium
from mineru.cli.common import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def update_markdown_image_paths(markdown_content, base_url, pdf_name):
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
            static_url = f"{base_url}/output/{pdf_name}/auto/images/{image_filename}"
            return f"![]({static_url})"
        
        # For other cases, assume it's a direct filename in the images directory
        static_url = f"{base_url}/output/{pdf_name}/auto/images/{image_path}"
        return f"![]({static_url})"
    
    return re.sub(image_pattern, replace_image_path, markdown_content)

@app.route('/', methods=['GET'])
def serve_index():
    return render_template('index.html')

@app.route('/process_pdf/', methods=['POST'])
def process_pdf():
    try:
        if 'file' not in request.files:
            abort(400, description="No file part")

        file = request.files['file']
        if file.filename == '':
            abort(400, description="No selected file")

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(tuple(pdf_suffixes + image_suffixes)):
            abort(400, description="Invalid file format. Only PDF or images are allowed.")

        file_bytes = file.read()
        file_suffix = f".{filename.rsplit('.', 1)[-1].lower()}"
        pdf_bytes = read_fn(file_bytes, file_suffix)
        pdf_file_name = Path(filename).stem

        output_dir = app.config['UPLOAD_FOLDER']

        output_files = process_pipeline(
            output_dir=output_dir,
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

        # Get base URL for static file serving
        base_url = request.url_root.rstrip('/')
        
        response_data = {}
        for pdf_name, files in output_files.items():
            response_data[pdf_name] = {}
            for file_type, file_path in files.items():
                if file_type in ["markdown", "content_list", "middle_json", "model_output"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Update markdown to use static file URLs for images
                    if file_type == "markdown":
                        content = update_markdown_image_paths(content, base_url, pdf_name)
                    
                    response_data[pdf_name][file_type] = content
                else:
                    rel_path = os.path.relpath(file_path, output_dir)
                    response_data[pdf_name][file_type] = rel_path

        return jsonify({
            "status": "success",
            "files": response_data,
            "message": "PDF processed successfully. Images are accessible via /output/ endpoints.",
            "download_base": "/download/",
            "static_base": "/output/"
        })

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

@app.route('/download/<path:rel_path>')
def download_file(rel_path):
    try:
        output_dir = app.config['UPLOAD_FOLDER']
        full_path = os.path.join(output_dir, rel_path)
        
        if not os.path.exists(full_path):
            abort(404, description="File not found")
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(output_dir)):
            abort(403, description="Access denied")
        
        return send_from_directory(output_dir, rel_path)
    
    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

@app.route('/output/<path:filename>')
def serve_static_file(filename):
    """
    Serve static files (images) from the output directory.
    This is equivalent to FastAPI's StaticFiles mount.
    """
    try:
        output_dir = app.config['UPLOAD_FOLDER']
        full_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(full_path):
            abort(404, description="File not found")
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(output_dir)):
            abort(403, description="Access denied")
        
        return send_from_directory(output_dir, filename)
    
    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)