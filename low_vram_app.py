import os
import gradio as gr
from gradio_litmodel3d import LitModel3D
import torch
import numpy as np
import imageio
from typing import Tuple, Dict, Any  # Add this line
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# Force using xformers attention
os.environ["ATTN_BACKEND"] = "xformers"

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Memory Management
def cleanup_memory():
    """Force cleanup CUDA memory"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def patch_pipeline(pipeline):
    """Patch pipeline to use less memory"""
    for model in pipeline.models.values():
        if hasattr(model, 'use_checkpoint'):
            model.use_checkpoint = True
        if hasattr(model, 'use_fp16'):
            model.use_fp16 = True
            if hasattr(model, 'convert_to_fp16'):
                model.convert_to_fp16()
    return pipeline

def low_vram_generate(
    image: Image.Image,
    seed: int = 0,
    ss_steps: int = 8,
    ss_guidance: float = 5.0,
    slat_steps: int = 8,
    slat_guidance: float = 2.0
) -> tuple[str, str]:  # Changed from Tuple to tuple
    """Generate 3D model with low VRAM settings"""
    cleanup_memory()
    
    # Process image
    processed_image = pipeline.preprocess_image(image)
    
    # Generate with reduced settings
    outputs = pipeline.run(
        processed_image,
        seed=seed,
        formats=["mesh"],  # Only generate mesh to save memory
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_steps,
            "cfg_strength": ss_guidance,
        },
        slat_sampler_params={
            "steps": slat_steps,
            "cfg_strength": slat_guidance,
        },
    )
    
    cleanup_memory()
    
    # Render preview with reduced quality
    video = render_utils.render_video(
        outputs['mesh'][0], 
        num_frames=20,  # Reduced frames
        resolution=256  # Lower resolution
    )['normal']
    
    cleanup_memory()
    
    # Save video
    video_path = os.path.join('outputs', 'preview.mp4')
    imageio.mimsave(video_path, video, fps=15)
    
    # Export mesh
    mesh_path = os.path.join('outputs', 'model.glb')
    mesh = outputs['mesh'][0]
    mesh.export(mesh_path)
    
    cleanup_memory()
    return mesh_path, video_path

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("## TRELLIS 3D Generation (Low VRAM Version)")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            seed = gr.Slider(0, MAX_SEED, value=0, label="Seed")
            
            with gr.Accordion("Advanced Settings", open=False):
                ss_steps = gr.Slider(1, 20, value=8, step=1, label="Structure Steps")
                ss_guidance = gr.Slider(0, 10, value=5.0, step=0.1, label="Structure Guidance")
                slat_steps = gr.Slider(1, 20, value=8, step=1, label="Detail Steps")
                slat_guidance = gr.Slider(0, 10, value=2.0, step=0.1, label="Detail Guidance")
            
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            model_output = gr.Model3D(label="3D Model")
            video_output = gr.Video(label="Preview")
    
    generate_btn.click(
        fn=low_vram_generate,
        inputs=[
            input_image,
            seed,
            ss_steps,
            ss_guidance,
            slat_steps,
            slat_guidance
        ],
        outputs=[model_output, video_output]
    )

if __name__ == "__main__":
    # Enable memory optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Load and patch pipeline
    print("Loading pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline = patch_pipeline(pipeline)
    pipeline.cuda()
    cleanup_memory()
    
    # Launch interface
    demo.launch(
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )