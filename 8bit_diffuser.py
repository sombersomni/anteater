import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.to(device)
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.to(device),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
decoder_output.show()