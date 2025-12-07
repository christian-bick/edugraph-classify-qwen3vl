FROM ghcr.io/ggml-org/llama.cpp:server

ADD https://huggingface.co/christian-bick/Qwen3-VL-4B-EduGraph-Q4_K_M-GGUF/resolve/main/qwen3-vl-4b-edugraph-q4_k_m.gguf /model.gguf

# 3. Set Environment Variables so we don't need complex CLI flags
ENV LLAMA_ARG_MODEL=/model.gguf
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080
ENV LLAMA_ARG_N_GPU_LAYERS=0
ENV LLAMA_ARG_CTX_SIZE=4096
