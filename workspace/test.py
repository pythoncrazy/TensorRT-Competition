from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch 

model_id = "stabilityai/stable-diffusion-2"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline="stable_diffusion_tensorrt_txt2img",
    # revision="fp16",
    torch_dtype=torch.float16,
    scheduler=scheduler,
)

# re-use cached folder to save ONNX models and TensorRT Engines
cache_dir = os.path.basename(model_id) + '_cached_trt'
os.makedirs(cache_dir, exist_ok=True)
print(f'cached dir: {cache_dir}')
pipe.set_cached_folder(
    cache_dir,
    revision="fp16",
)

pipe = pipe.to("cuda")

prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"
image = pipe(prompt).images[0]
image.save("tensorrt_mt_fuji.png")