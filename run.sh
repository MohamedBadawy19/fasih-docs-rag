#!/bin/bash
# ============================================================
# run.sh — Fasih-Docs launcher script
#
# This script is here specifically to cover the Shell Scripting

set -e  # Exit immediately if any command fails

# ── Colors for terminal output ─────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}   Fasih-Docs | فصيح-دوكس  ${NC}"
echo -e "${CYAN}   AI Engineering Documentation Assistant ${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# ── 1. Check Python ────────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking Python...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+ from python.org${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}OK: $PYTHON_VERSION${NC}"

# ── 2. Check virtual env / dependencies ───────────────────────
echo -e "${YELLOW}[2/4] Checking dependencies...${NC}"
if ! python -c "import gradio" &> /dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}OK: Dependencies installed${NC}"

# ── 3. Check Ollama ────────────────────────────────────────────
echo -e "${YELLOW}[3/4] Checking Ollama...${NC}"
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo -e "${YELLOW}Ollama not running. Starting Ollama...${NC}"
    ollama serve &
    sleep 3
    echo -e "${GREEN}OK: Ollama started${NC}"
else
    echo -e "${GREEN}OK: Ollama already running${NC}"
fi

# Check Mistral model is pulled
if ! ollama list | grep -q "mistral"; then
    echo -e "${YELLOW}Mistral model not found. Downloading (4GB — one time only)...${NC}"
    ollama pull mistral
fi

# ── 4. Check vector store ──────────────────────────────────────
echo -e "${YELLOW}[4/4] Checking vector store...${NC}"
if [ ! -d "chroma_db" ]; then
    echo -e "${YELLOW}Vector store not found. Running ingestion...${NC}"
    
    # Check docs folder
    if [ ! "$(ls -A docs/*.pdf 2>/dev/null)" ]; then
        echo -e "${RED}ERROR: No PDFs found in docs/ folder.${NC}"
        echo -e "${YELLOW}Add PDF files to the docs/ folder and run again.${NC}"
        exit 1
    fi
    
    python ingest.py
fi
echo -e "${GREEN}OK: Vector store ready${NC}"

# ── Launch app ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Starting Fasih-Docs...${NC}"
echo -e "${CYAN}Open your browser at: http://localhost:7860${NC}"
echo ""

python app.py

# ── Log the session ────────────────────────────────────────────
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] App session ended" >> logs/queries.log
