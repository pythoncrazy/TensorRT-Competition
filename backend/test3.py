import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16,custom_pipeline="stable_diffusion_tensorrt_txt2img",)

prompt = "a photograph of an astronaut riding a horse"
pipe.to("cuda")
image = pipe(prompt).images[0]
image.save(f"astronaut_rides_horse.png")