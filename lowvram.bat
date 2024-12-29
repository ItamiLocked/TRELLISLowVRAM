@echo off
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
set ATTN_BACKEND=xformers
python low_vram_app.py
pause
