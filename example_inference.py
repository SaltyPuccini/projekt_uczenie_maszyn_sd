from diffusers import AutoPipelineForText2Image
import torch
import gc

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to_cuda()
prompt = ("A pixel art illustration showcasing the top view of a fantasy game town, featuring vibrant colors, intricate details and medieval vibes")


gc.collect()
torch.cuda.empty_cache()

image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=5, guidance_scale=0.0).images[0]
image.save("sd-turbo-reference.png")
