#!/bin/bash
# .scripts/install_torch.sh
# Auto-detects GPU architecture and installs the correct PyTorch build

set -e

echo "=========================================="
echo "  PyTorch Auto-Installer (GPU-aware)"
echo "=========================================="

get_torch_index_url() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "cpu"
        return
    fi

    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

    if [ -z "$COMPUTE_CAP" ]; then
        echo "cu121"
        return
    fi

    echo "Detected compute capability: sm_${COMPUTE_CAP}" >&2

    case "$COMPUTE_CAP" in
        120|121)  # Blackwell (RTX 5070, 5080, 5090)
            echo "cu132"
            ;;
        89|90)   # Ada Lovelace (RTX 4060-4090) / Hopper
            echo "cu124"
            ;;
        86|87)   # Ampere (RTX 3060-3090, A100)
            echo "cu121"
            ;;
        80)      # Ampere (A100, older)
            echo "cu121"
            ;;
        75)      # Turing (RTX 2060-2080)
            echo "cu121"
            ;;
        *)
            echo "cu121"
            ;;
    esac
}

# Detect
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "No GPU detected")
CUDA_TAG=$(get_torch_index_url)

echo "GPU:        $GPU_NAME"
echo "PyTorch:    https://download.pytorch.org/whl/${CUDA_TAG}"
echo "=========================================="

# Uninstall existing torch
echo "Removing existing PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install correct version
INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
if [ "$CUDA_TAG" = "cpu" ]; then
    INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

echo "Installing PyTorch for ${CUDA_TAG}..."
pip install --no-cache-dir torch torchvision --index-url "$INDEX_URL"

# Try torchaudio (not available for all CUDA versions)
echo "Attempting to install torchaudio..."
pip install --no-cache-dir torchaudio --index-url "$INDEX_URL" 2>/dev/null || \
    echo "⚠️  torchaudio not available for ${CUDA_TAG}, skipping (not required for RL training)"

# Verify
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute cap:     sm_{cap[0]}{cap[1]}')
    try:
        t = torch.zeros(1).cuda()
        print(f'Kernel test:     ✅ PASSED')
    except RuntimeError as e:
        print(f'Kernel test:     ❌ FAILED - {e}')
        exit(1)
"

echo "=========================================="
echo "  ✅ PyTorch installation complete"
echo "=========================================="