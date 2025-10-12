import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure these variables
API_URL = "http://127.0.0.1:8088/process_pdf/"
PDF_FOLDER = "./input"  # Change this to your folder path
MAX_WORKERS = 5  # Number of concurrent requests (adjust based on your needs)

def process_pdf_file(file_path, api_url):
    """Process a single PDF file by posting it to the API."""
    file_name = Path(file_path).name
    start_time = time.time()
    
    try:
        with open(file_path, 'rb') as pdf_file:
            files = {'file': pdf_file}
            print(f"→ Sending: {file_name}")
            response = requests.post(api_url, files=files, timeout=300)  # 5 min timeout
            response.raise_for_status()
            elapsed = time.time() - start_time
            return True, response.json(), elapsed, file_name
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        return False, str(e), elapsed, file_name
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Unexpected error: {str(e)}", elapsed, file_name

def process_all_pdfs(folder_path, api_url, max_workers=MAX_WORKERS):
    """Process all PDF files in the specified folder concurrently."""
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get all PDF files
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Using {max_workers} concurrent workers")
    print(f"{'='*60}\n")
    
    overall_start = time.time()
    results = []
    
    # Process PDF files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_pdf_file, pdf_file, api_url): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Process completed tasks as they finish
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            pdf_file = future_to_file[future]
            
            try:
                success, result, elapsed, file_name = future.result()
                
                if success:
                    print(f"✓ [{completed}/{len(pdf_files)}] Success: {file_name} ({elapsed:.2f}s)")
                    # print(f"  Response: {result}")
                else:
                    print(f"✗ [{completed}/{len(pdf_files)}] Failed: {file_name} ({elapsed:.2f}s)")
                    print(f"  Error: {result}")
                
                results.append({
                    'file': file_name,
                    'success': success,
                    'result': result,
                    'elapsed': elapsed
                })
            except Exception as e:
                print(f"✗ [{completed}/{len(pdf_files)}] Exception: {pdf_file.name}")
                print(f"  Error: {str(e)}")
                results.append({
                    'file': pdf_file.name,
                    'success': False,
                    'result': str(e),
                    'elapsed': 0
                })
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total_processing_time = sum(r['elapsed'] for r in results)
    avg_time = total_processing_time / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:           {len(results)}")
    print(f"Successful:            {successful}")
    print(f"Failed:                {len(results) - successful}")
    print(f"Overall time:          {overall_elapsed:.2f}s")
    print(f"Avg processing time:   {avg_time:.2f}s per file")
    print(f"Throughput:            {len(results) / overall_elapsed:.2f} files/second")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_all_pdfs(PDF_FOLDER, API_URL)