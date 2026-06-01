#!/bin/bash
#SBATCH --job-name=sonic_agent
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sonic_agent_%j.out
#SBATCH --mail-user=your.email@institution.edu

# Load cluster modules (adjust for your site)
module load cuda/12.1
module load python/3.10
module load ffmpeg/6.0

# Project paths — adjust for your setup
PROJ=/path/to/your/project
AGENT_DIR=${PROJ}/sonic-o1-agent
JOB_TMP=${PROJ}/tmp/job_${SLURM_JOB_ID}

mkdir -p "${JOB_TMP}"/{segments,cache,uv,hf,torch,vllm}
mkdir -p "${JOB_TMP}/cache/torch/kernels"
chmod -R u+rwx "${JOB_TMP}"

export TMPDIR=${JOB_TMP}/segments
export SCRATCH_DIR=${JOB_TMP}
export XDG_CACHE_HOME=${JOB_TMP}/cache
export UV_CACHE_DIR=${JOB_TMP}/uv
export HF_HOME=${JOB_TMP}/hf
export HUGGINGFACE_HUB_CACHE=${JOB_TMP}/hf/hub
export TRANSFORMERS_CACHE=${JOB_TMP}/hf
export TORCH_HOME=${JOB_TMP}/torch
export VLLM_CACHE_ROOT=${JOB_TMP}/vllm
export VLLM_USE_SHM_BROADCAST=0

cd "${AGENT_DIR}"

echo '========================================='
echo 'Sonic O1 Agent'
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo '========================================='

uv run sonic-o1-agent \
  --config configs/agent_config.yaml \
  --video /path/to/your/video.mp4 \
  --audio /path/to/your/audio.m4a \
  --query "Summarize the key points discussed in this video" \
  --all-features \
  --verbose \
  --stream \
  --output data/outputs/result.json

echo '========================================='
echo 'Processing completed!'
echo '========================================='

rm -rf "${JOB_TMP}"
