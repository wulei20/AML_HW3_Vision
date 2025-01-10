import torch
from diffusers import StableDiffusionPipeline
import argparse
import os

BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"

def generate_from_raw_model(prompt, img_path, seed = None, guidance_scale = None):
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
    pipe.to("cuda")
    if seed:
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(int(seed))
        image = pipe(prompt, generator=generator).images[0]
    elif guidance_scale:
        image = pipe(prompt, guidance_scale=guidance_scale).images[0]
    else:
        image = pipe(prompt=prompt).images[0]
    image.save(img_path)

# generate_from_raw_model("Totoro", "1.png")

def generate_from_lora(lora_model_path, prompt, img_path, seed = None, guidance_scale = None):
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(lora_model_path)
    pipe.to("cuda")

    if seed:
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(int(seed))
        print("seed:", seed)
        image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    elif guidance_scale:
        image = pipe(prompt, num_inference_steps=30, guidance_scale=guidance_scale).images[0]
    else:
        image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(img_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', action="store_true")
    parser.add_argument('-l', '--lora_model_path', default="finetune/lora/pokemon")
    parser.add_argument('-g', '--guidance_scale', default=None)
    parser.add_argument('-s', '--seed', default=None)
    parser.add_argument('-o', '--output_dir', default="result")
    args = parser.parse_args()
    while True:
        prompt = input("Please input prompt for generation:")
        if args.lora:
            output_path = os.path.join(args.output_dir, os.path.basename(args.lora_model_path),
                    f"{prompt}_s{args.seed}.png" if args.seed else f"{prompt}_g{args.guidance_scale}.png" if args.guidance_scale else f"{prompt}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Saving img with prompt '{prompt}' to path: {output_path}")
            if args.seed:
                generate_from_lora(args.lora_model_path, prompt, output_path, args.seed)
            elif args.guidance_scale:
                generate_from_lora(args.lora_model_path, prompt, output_path, guidance_scale=args.guidance_scale)
            else:
                generate_from_lora(args.lora_model_path, prompt, output_path)
        else:
            output_path = os.path.join(args.output_dir, "raw",
                    f"{prompt}_s{args.seed}.png" if args.seed else f"{prompt}_g{args.guidance_scale}.png" if args.guidance_scale else f"{prompt}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Saving img with prompt '{prompt}' to path: {output_path}")
            if args.seed:
                generate_from_raw_model(prompt, output_path, args.seed)
            elif args.guidance_scale:
                generate_from_raw_model(prompt, output_path, guidance_scale=args.guidance_scale)
            else:
                generate_from_raw_model(prompt, output_path)

if __name__ == "__main__":
    # generate_from_lora("finetune/lora/pokemon", "Totoro", "3.png", seed="0")
    main()