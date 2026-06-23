import shutil

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image, ImageEnhance
import uuid
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None
txt2img_pipe = None
img2img_pipe = None

POSITIVE_PROMPT = (
    "pixel art style, low resolution photograph, slightly blurred image, soft focus, "
    "simple centered composition, single subject, natural lighting, "
    "realistic style, candid photo, subtle film grain, "
    "slightly compressed digital image, mild jpeg artifacts, "
    "minimal background detail, plain or softly blurred background, "
    "clean composition, shallow depth of field, "
    "natural colors, realistic textures"
)


def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    return pipe


def get_txt2img_pipe():
    global txt2img_pipe
    if txt2img_pipe is None:
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
            if device == "cuda"
            else torch.float32
        ).to(device)
    return txt2img_pipe


def get_img2img_pipe():
    global img2img_pipe
    if img2img_pipe is None:
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
            if device == "cuda"
            else torch.float32
        ).to(device)
    return img2img_pipe


def generate_image(prompt: str, path: str):
    full_prompt = f"{prompt}, {POSITIVE_PROMPT}"
    NEGATIVE_PROMPT = (
        "grid, graph paper, blueprint, sketchbook, notebook, ruler lines, "
        "texture background, pattern background, noisy background, "
        "gradient, shading, lighting, shadow, "
        "realistic, 3d, photo, render, "
        "text, watermark, logo, "
        "complex background, scenery, multiple objects"
    )
    try:
        with torch.inference_mode():
            result = get_txt2img_pipe()(
                prompt=full_prompt,
                negative_prompt= NEGATIVE_PROMPT,
                guidance_scale=8,
                num_inference_steps=30
            )
            image = result.images[0]
        image.save(path)
    except Exception as e:
        print("IMAGE GENERATION ERROR:", e)
        raise


def generate_from_image(
        input_image_path: str,
        prompt: str,
        output_path: str,
        strength: float = 0.4
):
    full_prompt = f"{prompt}, {POSITIVE_PROMPT}"
    NEGATIVE_PROMPT = (
        "grid, graph paper, blueprint, sketchbook, notebook, ruler lines, "
        "texture background, pattern background, noisy background, "
        "gradient, shading, lighting, shadow, "
        "realistic, 3d, photo, render, "
        "text, watermark, logo, "
        "complex background, scenery, multiple objects"
    )
    try:
        init_image = Image.open(input_image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        with torch.inference_mode():
            result = get_img2img_pipe()(
                prompt=full_prompt,
                image=init_image,
                strength=float(strength),
                guidance_scale=7.5,
                num_inference_steps=30,
                negative_prompt=NEGATIVE_PROMPT
            )
            image = result.images[0]
        image.save(output_path)
    except Exception as e:
        print("IMAGE TO IMAGE ERROR:", e)
        raise


def process_image(input_path, output_path, stitch_size=32, color_count=4, final_size=192):
    img = Image.open(input_path).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Color(img).enhance(1.3)

    img = img.resize((stitch_size, stitch_size), Image.LANCZOS)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=color_count)
    img = img.convert("RGB")
    img = img.resize((final_size, final_size), Image.NEAREST)
    img.save(output_path)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <script>
                function toggleModeFields() {
                    var mode = document.getElementById("modeSelect").value;
                    var img2imgFields = document.getElementsByClassName("img2img-only");
                    for (var i = 0; i < img2imgFields.length; i++) {
                        if (mode === "img2img") {
                            img2imgFields[i].style.display = "block";
                        } else {
                            img2imgFields[i].style.display = "none";
                        }
                    }
                }
                window.onload = toggleModeFields;
            </script>
        </head>
        <body style="text-align:center;font-family:Arial;padding:40px;">
            <h2>Knitting Image Generator</h2>
            <form action="/generate" method="post" enctype="multipart/form-data">
                <label>Mode</label>
                <br>
                <select name="mode" id="modeSelect" onchange="toggleModeFields()">
                    <option value="txt2img">Text → Image</option>
                    <option value="img2img">Image → Image</option>
                </select>
                <br><br>
                <input
                    name="prompt"
                    style="width:350px;padding:8px;"
                    placeholder="Enter prompt"
                />
                <br><br>
                <div class="img2img-only" style="display:none;">
                    <label>Upload Image (only for img2img)</label>
                    <br>
                    <input type="file" name="source_image">
                    <br><br>
                </div>
                <label>Stitch Resolution</label>
                <br>
                <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                    <input type="range" min="16" max="64" value="32" name="stitch_size" id="stitchRange"
                           oninput="document.getElementById('stitchVal').innerText = this.value"/>
                    <span id="stitchVal" style="font-weight:bold; width: 20px; text-align: left;">32</span>
                </div>
                <br>
                <label>Color Count</label>
                <br>
                <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                    <input type="range" min="2" max="8" value="4" name="color_count" id="colorRange"
                           oninput="document.getElementById('colorVal').innerText = this.value"/>
                    <span id="colorVal" style="font-weight:bold; width: 20px; text-align: left;">4</span>
                </div>
                <br>
                <label>Output Size</label>
                <br>
                <select name="final_size">
                    <option value="128">128px</option>
                    <option value="192" selected>192px</option>
                    <option value="256">256px</option>
                </select>
                <br><br>
                <div class="img2img-only" style="display:none;">
                    <label>Strength (img2img only)</label>
                    <br>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                        <input type="range" min="0.1" max="1.0" step="0.1" value="0.4" name="strength" id="strengthRange"
                               oninput="document.getElementById('strengthVal').innerText = this.value"/>
                        <span id="strengthVal" style="font-weight:bold; width: 20px; text-align: left;">0.4</span>
                    </div>
                    <br>
                </div>

                <button type="submit">Generate</button>
            </form>
        </body>
    </html>
    """


@app.post("/generate", response_class=HTMLResponse)
async def generate(
        prompt: str = Form(...),
        mode: str = Form("txt2img"),
        source_image: UploadFile = File(None),
        stitch_size: int = Form(32),
        color_count: int = Form(4),
        final_size: int = Form(192),
        strength: float = Form(0.4)
):
    try:
        file_id = str(uuid.uuid4())
        raw_filename = f"{file_id}_raw.png"
        processed_filename = f"{file_id}_processed.png"
        raw_path = f"{OUTPUT_DIR}/{raw_filename}"
        processed_path = f"{OUTPUT_DIR}/{processed_filename}"
        if mode == "img2img":
            if source_image is None or source_image.filename == "":
                raise Exception("Image required for img2img mode")
            uploaded_path = f"{OUTPUT_DIR}/{file_id}_input.png"
            with open(uploaded_path, "wb") as buffer:
                shutil.copyfileobj(source_image.file, buffer)
            generate_from_image(
                uploaded_path,
                prompt,
                raw_path,
                strength
            )
        else:
            generate_image(prompt, raw_path)
        process_image(
            raw_path,
            processed_path,
            stitch_size=stitch_size,
            color_count=color_count,
            final_size=final_size
        )
        return f"""
        <html>
            <body style="text-align:center;font-family:Arial;padding:40px;">
                <h2>Generation Results</h2>
                <h4>Prompt</h4>
                <p>{prompt}</p>
                <h4>Raw Image</h4>
                <img src="/file/{raw_filename}" width="300"/>
                <h4>Knitting Pattern</h4>
                <img src="/file/{processed_filename}" width="300"/>
                <br><br>
                <a href="/download/{processed_filename}">
                    <button>Download Image</button>
                </a>
                <br><br>
                <form action="/generate" method="post" enctype="multipart/form-data">
                    <input type="hidden" name="prompt" value="{prompt}">
                    <input type="hidden" name="mode" value="{mode}">
                    <input type="hidden" name="stitch_size" value="{stitch_size}">
                    <input type="hidden" name="color_count" value="{color_count}">
                    <input type="hidden" name="final_size" value="{final_size}">
                    <input type="hidden" name="strength" value="{strength}">
                    <button type="submit">Regenerate</button>
                </form>
                <br>
                <a href="/">Generate Another</a>
            </body>
        </html>
        """

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"""
        <html>
            <body style="text-align:center;font-family:Arial;">
                <h3 style="color:red;">Error Occurred</h3>
                <pre>{str(e)}</pre>
                <br><br>
                <a href="/">Go Back</a>
            </body>
        </html>
        """


@app.get("/file/{filename}")
def get_file(filename: str):
    return FileResponse(f"{OUTPUT_DIR}/{filename}")


@app.get("/download/{filename}")
def download_file(filename: str):
    path = f"{OUTPUT_DIR}/{filename}"
    return FileResponse(
        path,
        media_type="image/png",
        filename=filename
    )