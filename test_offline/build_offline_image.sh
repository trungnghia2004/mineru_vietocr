#!/bin/bash

# Script to automate the offline Docker image build process
# Run this on a machine with internet access

set -e  # Exit on error

echo "======================================================================"
echo "  MinerU VietOCR - Offline Docker Image Builder"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if we're in the right directory
if [ ! -f "download_models.py" ]; then
    echo -e "${RED}Error: download_models.py not found!${NC}"
    echo "Please run this script from the test_offline directory."
    exit 1
fi

# Step 2: Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found!${NC}"
    echo "Please install Python 3 to continue."
    exit 1
fi

# Step 3: Check if huggingface_hub is installed
echo -e "${YELLOW}Checking for huggingface_hub...${NC}"
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# Step 4: Download models
echo ""
echo -e "${GREEN}Step 1: Downloading models from HuggingFace...${NC}"
echo "----------------------------------------------------------------------"
python3 download_models.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Model download failed!${NC}"
    exit 1
fi

# Step 5: Check if models were downloaded
if [ ! -d "models_cache/pipeline/models" ]; then
    echo -e "${RED}Error: Models directory not found!${NC}"
    echo "Expected: models_cache/pipeline/models"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Models downloaded successfully!${NC}"
echo ""

# Step 6: Build Docker image
echo -e "${GREEN}Step 2: Building Docker image...${NC}"
echo "----------------------------------------------------------------------"
echo "This may take 10-30 minutes depending on your machine..."
echo ""

cd ..  # Go to project root
docker build -f test_offline/Dockerfile.offline -t mineru-vietocr-offline:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Docker build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Docker image built successfully!${NC}"
echo ""

# Step 7: Save Docker image
echo -e "${GREEN}Step 3: Saving Docker image to tar file...${NC}"
echo "----------------------------------------------------------------------"
docker save mineru-vietocr-offline:latest -o mineru-vietocr-offline.tar

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to save Docker image!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Docker image saved successfully!${NC}"
echo ""

# Final summary
IMAGE_SIZE=$(du -h mineru-vietocr-offline.tar | cut -f1)

echo "======================================================================"
echo -e "${GREEN}  BUILD COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo "Docker image: mineru-vietocr-offline:latest"
echo "Tar file: mineru-vietocr-offline.tar"
echo "File size: $IMAGE_SIZE"
echo ""
echo "Next steps:"
echo "  1. Transfer mineru-vietocr-offline.tar to your offline server"
echo "  2. On the offline server, run:"
echo "     docker load -i mineru-vietocr-offline.tar"
echo "  3. Start the container:"
echo "     docker run -d --name mineru-vietocr -p 8088:8088 mineru-vietocr-offline:latest"
echo ""
echo "For detailed instructions, see test_offline/README.md"
echo ""

