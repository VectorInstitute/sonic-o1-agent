#!/bin/bash
#SBATCH --job-name=sonic_agent
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sonic_agent_%j.out
#SBATCH --mail-user=your.email@institution.edu

# Load required modules (adjust based on your cluster)
module load cuda/12.1
module load python/3.10
module load ffmpeg/6.0

# Project paths - ADJUST THESE FOR YOUR SETUP
PROJ=/path/to/your/project
AGENT_DIR=${PROJ}/sonic-o1-agent
JOB_TMP=${PROJ}/tmp/job_${SLURM_JOB_ID}

# Create temporary directories
mkdir -p "${JOB_TMP}"/{segments,cache,hf,torch,vllm}
mkdir -p "${JOB_TMP}/cache/torch/kernels"
chmod -R u+rwx "${JOB_TMP}"

# Set environment variables
export TMPDIR=${JOB_TMP}/segments
export SCRATCH_DIR=${JOB_TMP}
export XDG_CACHE_HOME=${JOB_TMP}/cache
export HF_HOME=${JOB_TMP}/hf
export HUGGINGFACE_HUB_CACHE=${JOB_TMP}/hf/hub
export TRANSFORMERS_CACHE=${JOB_TMP}/hf
export TORCH_HOME=${JOB_TMP}/torch
export VLLM_CACHE_ROOT=${JOB_TMP}/vllm

# vLLM settings
export VLLM_USE_SHM_BROADCAST=0

# Activate virtual environment (uncomment if using one)
# source ${AGENT_DIR}/.venv/bin/activate

cd ${AGENT_DIR}

echo '========================================='
echo 'Sonic O1 Agent - Interactive Processing'
echo '========================================='
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo '========================================='

# Run the agent - ADJUST VIDEO/AUDIO PATHS
python scripts/run_agent.py \\
  --config configs/agent_config.yaml \\
  --video /path/to/your/video.mp4 \\
  --audio /path/to/your/audio.m4a \\
  --query "Summarize the key points discussed in this video" \\
  --stream

echo '========================================='
echo 'Processing completed!'
echo '========================================='

# Cleanup temporary files
rm -rf "${JOB_TMP}"
