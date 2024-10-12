FROM runpod/base:0.6.1-cuda12.2.0

# Environment variable to differentiate between development and production
ARG ENVIRONMENT=development
ENV ENVIRONMENT=${ENVIRONMENT}

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends aria2 git-lfs unzip ffmpeg libegl1-mesa libegl1 && \
    rm -rf /var/lib/apt/lists/*

# Clone the Pyramid Flow repository 
RUN git clone https://github.com/jy0205/Pyramid-Flow /content/Pyramid-Flow 

# Download required model files
RUN  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/causal_video_vae/config.json -d /content/model/causal_video_vae -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/causal_video_vae/diffusion_pytorch_model.bin -d /content/model/causal_video_vae -o diffusion_pytorch_model.bin && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/diffusion_transformer_384p/config.json -d /content/model/diffusion_transformer_384p -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/diffusion_transformer_384p/diffusion_pytorch_model.bin -d /content/model/diffusion_transformer_384p -o diffusion_pytorch_model.bin && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/diffusion_transformer_768p/config.json -d /content/model/diffusion_transformer_768p -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/diffusion_transformer_768p/diffusion_pytorch_model.bin -d /content/model/diffusion_transformer_768p -o diffusion_pytorch_model.bin && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/text_encoder/model.safetensors -d /content/model/text_encoder -o model.safetensors && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/text_encoder_2/config.json -d /content/model/text_encoder_2 -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/text_encoder_2/model.safetensors -d /content/model/text_encoder_2 -o model.safetensors && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/text_encoder_3/config.json -d /content/model/text_encoder_3 -o config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/text_encoder_3/model-00001-of-00002.safetensors -d /content/model/text_encoder_3 -o model-00001-of-00002.safetensors && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/text_encoder_3/model-00002-of-00002.safetensors -d /content/model/text_encoder_3 -o model-00002-of-00002.safetensors && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/text_encoder_3/model.safetensors.index.json -d /content/model/text_encoder_3 -o model.safetensors.index.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer/merges.txt -d /content/model/tokenizer -o merges.txt && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer/vocab.json -d /content/model/tokenizer -o vocab.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_2/merges.txt -d /content/model/tokenizer_2 -o merges.txt && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_2/special_tokens_map.json -d /content/model/tokenizer_2 -o special_tokens_map.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_2/tokenizer_config.json -d /content/model/tokenizer_2 -o tokenizer_config.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_2/vocab.json -d /content/model/tokenizer_2 -o vocab.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_3/special_tokens_map.json -d /content/model/tokenizer_3 -o special_tokens_map.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/resolve/main/tokenizer_3/spiece.model -d /content/model/tokenizer_3 -o spiece.model && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_3/tokenizer.json -d /content/model/tokenizer_3 -o tokenizer.json && \
     aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/vdo/pyramid-flow-sd3/raw/main/tokenizer_3/tokenizer_config.json -d /content/model/tokenizer_3 -o tokenizer_config.json

COPY builder/requirements.txt /requirements.txt

# Install Python dependencies if building for production
RUN if [ "$ENVIRONMENT" = "production" ]; then \
    python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r /requirements.txt; \
    fi

# Copy all files from src to the root directory
COPY ./src/ /

# Final CMD based on environment
CMD if [ "$ENVIRONMENT" = "development" ]; then \
    /start.sh; \
    else \
    python3.10 -u /handler.py; \
    fi
