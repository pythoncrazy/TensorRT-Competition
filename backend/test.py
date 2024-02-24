import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
torch.cuda.empty_cache()

# Use the DDIMScheduler scheduler here instead
# scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
#                                             subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                custom_pipeline="stable_diffusion_tensorrt_txt2img",
                                                variant='fp16',
                                                torch_dtype=torch.float16,
                                                )

# re-use cached folder to save ONNX models and TensorRT Engines
pipe.set_cached_folder("stabilityai/stable-diffusion-2-1", revision='fp16',)
pipe = pipe.to("cuda")

prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"
image = pipe(prompt,height = 128, width = 128).images[0]
image.save('tensorrt_mt_fuji.png')
