from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
import uuid
import os
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None

POSITIVE_PROMPT = (
    "pixel art, knitting pattern, intarsia knitting, "
    "clean color separation, low detail, centered composition, "
    "bold shapes, high contrast, soft yarn texture"
)

def get_pipe():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    return pipe

def generate_image(prompt: str, path: str):
    full_prompt = f"{prompt}, {POSITIVE_PROMPT}"
    negative_prompt = (
        "photorealistic, blurry, text, watermark, "
        "high detail, realistic shading, noisy background"
    )
    try:
        with torch.inference_mode():
            result = get_pipe()(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=8,
                num_inference_steps=30
            )
            image = result.images[0]
        image.save(path)
    except Exception as e:
        print("IMAGE GENERATION ERROR:", e)
        raise

def process_image(
    input_path: str,
    output_path: str,
    stitch_size: int = 32,
    color_count: int = 4,
    final_size: int = 192
):
    img = Image.open(input_path)
    img = img.resize((stitch_size, stitch_size), Image.NEAREST)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=color_count)
    img = img.convert("RGB")
    img = img.resize((final_size, final_size), Image.NEAREST)
    img.save(output_path)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body style="text-align:center;font-family:Arial;padding:40px;">
            <h2>Knitting Image Generator</h2>
            <form action="/generate" method="post">
                <input
                    name="prompt"
                    style="width:350px;padding:8px;"
                    placeholder="Enter prompt"
                />
                <br><br>
                <label>Stitch Resolution</label>
                <br>
                <input
                    type="range"
                    min="16"
                    max="64"
                    value="32"
                    name="stitch_size"
                />
                <br><br>
                <label>Color Count</label>
                <br>
                <input
                    type="range"
                    min="2"
                    max="8"
                    value="4"
                    name="color_count"
                />
                <br><br>
                <label>Output Size</label>
                <br>
                <select name="final_size">
                    <option value="128">128px</option>
                    <option value="192" selected>192px</option>
                    <option value="256">256px</option>
                </select>
                <br><br>
                <button type="submit">
                    Generate
                </button>
            </form>
        </body>
    </html>
    """

@app.post("/generate", response_class=HTMLResponse)
def generate(
    prompt: str = Form(...),
    stitch_size: int = Form(32),
    color_count: int = Form(4),
    final_size: int = Form(192)
):

    try:
        file_id = str(uuid.uuid4())
        raw_filename = f"{file_id}_raw.png"
        processed_filename = f"{file_id}_processed.png"
        raw_path = f"{OUTPUT_DIR}/{raw_filename}"
        processed_path = f"{OUTPUT_DIR}/{processed_filename}"
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
                    <button>
                        Download Image
                    </button>
                </a>
                <br><br>
                <form action="/generate" method="post">
                    <input type="hidden" name="prompt" value="{prompt}">
                    <input type="hidden" name="stitch_size" value="{stitch_size}">
                    <input type="hidden" name="color_count" value="{color_count}">
                    <input type="hidden" name="final_size" value="{final_size}">
                    <button type="submit">
                        Regenerate
                    </button>
                </form>
                <br>
                <a href="/">
                    Generate Another
                </a>
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