import torch
from PIL import Image
import trimesh
from trellis import TrellisImageTo3DPipeline

def run_low_vram(
    pipeline,
    image,
    num_samples=1,
    seed=42,
    resolution=512,  # Reduced from 1024
    texture_size=512,  # Reduced from 1024
    fill_holes_resolution=512,  # Reduced from 1024
    num_views=50,  # Reduced from 100
    sparse_structure_sampler_params={
        "steps": 50,  # Reduced number of sampling steps
    },
    slat_sampler_params={
        "steps": 50,  # Reduced number of sampling steps
    }
):
    """
    Run Trellis pipeline with reduced VRAM usage.
    """
    # Preprocess image
    if isinstance(image, str):
        image = Image.open(image)
    image = pipeline.preprocess_image(image)
    
    # Get conditioning info
    cond = pipeline.get_cond([image])
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Sample sparse structure with reduced parameters
    coords = pipeline.sample_sparse_structure(
        cond, 
        num_samples,
        sparse_structure_sampler_params
    )
    
    # Sample structured latent
    slat = pipeline.sample_slat(
        cond,
        coords,
        slat_sampler_params
    )
    
    # Decode with reduced parameters
    return pipeline.decode_slat(
        slat,
        formats=['mesh', 'gaussian']  # Reduced number of formats
    )

def main():
    # Load the model
    print("Loading Trellis model...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("stabilityai/TrellisStableDiffusion")
    pipeline.cuda()  # Move to GPU
    
    # Load your image
    image_path = "your_image.jpg"  # Replace with your image path
    print(f"Processing image: {image_path}")
    
    # Run the low VRAM version
    results = run_low_vram(
        pipeline=pipeline,
        image=image_path,
        resolution=512,  # Adjust based on your VRAM
        texture_size=512,
        num_views=50
    )
    
    # Save the results
    if 'mesh' in results:
        print("Saving mesh...")
        mesh = results['mesh']
        mesh.export("output_mesh.glb")
    
    if 'gaussian' in results:
        print("Gaussian representation generated...")
        # Handle gaussian output if needed
        
    print("Processing complete!")

if __name__ == "__main__":
    main()