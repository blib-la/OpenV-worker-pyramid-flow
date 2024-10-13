import os, random, time, runpod, requests, string, hashlib, mimetypes, sys
import logging

import torch
from PIL import Image

# Adding the correct path for Pyramid-Flow
sys.path.append("/content/Pyramid-Flow")

from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

# Set the UploadThing API key from environment variables
UPLOADTHING_API_KEY = os.getenv("UPLOADTHING_API_KEY")

model_dtype, torch_dtype = "bf16", torch.bfloat16
logging.basicConfig(level=logging.INFO)

model = PyramidDiTForVideoGeneration(
    "/content/model",
    model_dtype,
    model_variant="diffusion_transformer_768p",
)
logging.info(f"Model loaded successfully from /content/model")

model.vae.to("cuda")
model.dit.to("cuda")
model.text_encoder.to("cuda")
model.vae.enable_tiling()

video_path = "/content/pyramid-flow-t2v.mp4"


def upload_file_to_uploadthing(file_path):
    """Uploads a file to UploadThing using a pre-signed URL."""
    file_name = os.path.basename(file_path)
    _, file_extension = os.path.splitext(file_name)

    # Generate random string for file name
    random_string = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    file_name = md5_hash + file_extension
    file_size = os.path.getsize(file_path)
    file_type, _ = mimetypes.guess_type(file_path)

    # Read file content
    with open(file_path, "rb") as file:
        file_content = file.read()

    # File info
    file_info = {"name": file_name, "size": file_size, "type": file_type}

    # Get presigned URL from UploadThing
    headers = {"x-uploadthing-api-key": UPLOADTHING_API_KEY}
    data = {"contentDisposition": "inline", "acl": "public-read", "files": [file_info]}
    presigned_response = requests.post(
        "https://api.uploadthing.com/v6/uploadFiles", headers=headers, json=data
    )
    presigned_response.raise_for_status()

    # Upload file using the presigned URL
    presigned = presigned_response.json()["data"][0]
    upload_url = presigned["url"]
    fields = presigned["fields"]
    files = {"file": file_content}
    upload_response = requests.post(upload_url, data=fields, files=files)
    upload_response.raise_for_status()

    # Return the file URL
    file_url = presigned["fileUrl"]
    return file_url


# Add this function to list directory contents
def list_directory_contents(path):
    contents = []
    for root, dirs, files in os.walk(path):
        for name in files:
            contents.append(os.path.join(root, name))
        for name in dirs:
            contents.append(os.path.join(root, name) + "/")
    return contents


@torch.inference_mode()
def generate(input):
    try:
        # List the contents of the /content/model directory
        model_contents = list_directory_contents("/content/model")
        print("Contents of /content/model:")
        for item in model_contents:
            print(item)

        values = input["input"]

        prompt = values["prompt"]
        num_inference_steps = values["num_inference_steps"]
        video_num_inference_steps = values["video_num_inference_steps"]
        width = values["width"]
        height = values["height"]
        temp = values["temp"]
        guidance_scale = values["guidance_scale"]
        video_guidance_scale = values["video_guidance_scale"]
        seed = values["seed"]
        fps = values["fps"]

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        torch.manual_seed(seed)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            frames = model.generate(
                prompt=prompt,
                num_inference_steps=[
                    num_inference_steps,
                    num_inference_steps,
                    num_inference_steps,
                ],
                video_num_inference_steps=[
                    video_num_inference_steps,
                    video_num_inference_steps,
                    video_num_inference_steps,
                ],
                height=height,
                width=width,
                temp=temp,  # temp=16: 5s, temp=31: 10s
                guidance_scale=guidance_scale,  # The guidance for the first frame
                video_guidance_scale=video_guidance_scale,  # The guidance for the other video latent
                output_type="pil",
                save_memory=False,
            )

        export_to_video(frames, video_path, fps=fps)

        # Upload the video to UploadThing
        video_url = upload_file_to_uploadthing(video_path)

        return {
            "video": video_url,
            "status": "DONE",
        }

    except Exception as e:
        return {"video": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


runpod.serverless.start({"handler": generate})
