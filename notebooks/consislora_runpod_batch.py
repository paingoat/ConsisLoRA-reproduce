# %% [markdown]
# # ConsisLoRA — RunPod batch (style + content train / infer)
#
# - **Source format:** [Jupytext](https://jupytext.readthedocs.io/) (percent) — generate `consislora_runpod_batch.ipynb` with `script/jupytext_to_ipynb.sh`.
# - **Hugging Face cache:** `HF_HOME` defaults to `/workspace/huggingface` on RunPod, else `~/.cache/huggingface` locally.
# - Trains 5 **style** LoRAs (two-stage, same as `script/train_style.sh`), 2 **content** LoRAs (`script/train_content.sh`), then runs **style-only** and **content×style** inference.
# - **Section order (A → C → B → D):** all training first, then load SDXL for inference. This avoids GPU memory spikes from keeping the full pipeline in memory while `accelerate launch` trains.
# - **LoRA output dirs (under repo root, created with `mkdir` before training):** `lora-weights/style/{name}/` (style stage 1), `lora-weights/style_retrain/{name}/` (style stage 2 — use this for inference), `lora-weights/content/{name}/`. Weight files are written by `train_consislora.py` when training finishes, not committed in git.
# - **Section E:** optional upload of the whole `lora-weights/` tree to [paingoat/ConsisLoRA-test](https://huggingface.co/paingoat/ConsisLoRA-test) using `HfApi.upload_folder` (see [Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)). Set `HF_TOKEN` in a local `.env` (see `.env.example`).

# %% [markdown]
# ## Config

# %%
import os
import subprocess
import sys
from pathlib import Path


def _get_repo_root() -> Path:
    """Jupyter has no `__file__` in a cell; script runs from `notebooks/` with `__file__`."""
    if "__file__" in globals():
        return Path(globals()["__file__"]).resolve().parent.parent
    p = Path.cwd().resolve()
    for up in (p, p.parent):
        if (up / "pipeline_demo.py").is_file() and (up / "train_consislora.py").is_file():
            return up
    raise FileNotFoundError(
        f"Could not find repo root (expected pipeline_demo.py). cwd={p}. "
        "Open the project folder or `cd` to the ConsisLoRA repo root, then re-run the cell."
    )


REPO_ROOT = _get_repo_root()
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm

# Local repo modules (must come after sys.path)
from pipeline_demo import StableDiffusionXLPipelineLoraGuidance
from utils import load_pil_image

# Hugging Face model cache (RunPod: use persistent / workspace volume)
if Path("/workspace").is_dir():
    _hf = Path("/workspace") / "huggingface"
    _hf.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_hf)
else:
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# Training / inference layout (same conventions as `script/`)
LORA_STYLE_STAGE1 = REPO_ROOT / "lora-weights" / "style"
LORA_STYLE_FINAL = REPO_ROOT / "lora-weights" / "style_retrain"
LORA_CONTENT = REPO_ROOT / "lora-weights" / "content"
DATA_STYLE = REPO_ROOT / "data_demo" / "style"
DATA_CONTENT = REPO_ROOT / "data_demo" / "content"
OUT_INFER = REPO_ROOT / "outputs" / "runpod_batch"

# Hugging Face Hub — section E (upload); token from `.env` as HF_TOKEN
HF_REPO_ID = "paingoat/ConsisLoRA-test"

STYLE_NAMES = ["rococo", "baroque", "monet", "shinkai", "ukyoe"]
CONTENT_NAMES = ["rio", "uit"]

# Prompts: English; append `, in the style of [v]` for style-LoRA-only inference (repo convention).
P1_BASE = (
    "Elegant woman wearing a lavish pastel gown, sitting gracefully on an ornate swing "
    "in a lush formal garden"
)
P2_BASE = (
    "Farm landscape in the Alps with a rustic wooden barn, rolling green hills, "
    "and snow-capped mountains in the background"
)

SEED = 29
MAX_TRAIN_STEPS_STYLE_STAGE1 = 1500
MAX_TRAIN_STEPS_STYLE_STAGE2 = 1000
MAX_TRAIN_STEPS_CONTENT = 1500
STYLE_LORA_SCALE = 1.0
INFER_STEPS = 30
GUIDANCE = 7.5

# %% [markdown]
# ## Helpers: grid + training commands

# %%
def show_grid_pil(
    images: list,
    nrows: int,
    ncols: int,
    title: str,
    cell_titles: list | None = None,
    figsize_per_cell=(3.0, 3.0),
):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows)
    )
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    for i, im in enumerate(images):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(im)
            ax.axis("off")
            if cell_titles and i < len(cell_titles):
                ax.set_title(cell_titles[i], fontsize=8)
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def _run_training(
    cmd: list,
    cwd: Path,
    *,
    title: str | None = None,
) -> None:
    """Run `accelerate launch ...` with unbuffered child stdout so `tqdm` in `train_consislora` streams in the notebook."""
    if title:
        print("\n" + "=" * 72, flush=True)
        print(f"  {title}", flush=True)
        print("=" * 72 + "\n", flush=True)
    print(">", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if "COLUMNS" not in env:
        env["COLUMNS"] = "120"
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[error] Command failed with exit code {e.returncode}", flush=True)
        raise
    if title:
        print(f"\n[ok] {title}\n", flush=True)


def build_sdxl_pipeline() -> StableDiffusionXLPipelineLoraGuidance:
    device = "cuda"
    dtype = torch.float16
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipelineLoraGuidance.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    return pipe


def train_style_two_stage(name: str) -> None:
    """Match `script/train_style.sh` for one style image (basename `name`, file `name.jpg`)."""
    image_path = DATA_STYLE / f"{name}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    out1 = LORA_STYLE_STAGE1 / name
    out2 = LORA_STYLE_FINAL / name
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)

    ck1500 = out1 / f"checkpoint-{MAX_TRAIN_STEPS_STYLE_STAGE1}"
    if not ck1500.is_dir():
        _run_training(
            [
                "accelerate",
                "launch",
                "train_consislora.py",
                "--instance_prompt=A [v]",
                f"--image_path={image_path}",
                f"--output_dir={out1}",
                "--start_x0_loss_steps=500",
                "--rank=64",
                "--lr=2e-4",
                "--second_lr=1e-4",
                f"--max_train_steps={MAX_TRAIN_STEPS_STYLE_STAGE1}",
                "--checkpointing_steps=500",
                "--resolution=1024",
                "--mixed_precision=fp16",
                "--use_8bit_adam",
                "--seed=0",
            ],
            REPO_ROOT,
            title=f"Style {name!r} — stage 1 (A [v], → {out1.relative_to(REPO_ROOT)})",
        )
    else:
        print(f"Skip stage-1 (found {ck1500}) for style={name}", flush=True)

    _run_training(
        [
            "accelerate",
            "launch",
            "train_consislora.py",
            "--instance_prompt=An image in the style of [v]",
            f"--image_path={image_path}",
            f"--output_dir={out2}",
            f"--content_lora_path={out1}",
            "--start_x0_loss_steps=0",
            "--rank=64",
            "--lr=1e-4",
            f"--max_train_steps={MAX_TRAIN_STEPS_STYLE_STAGE2}",
            "--checkpointing_steps=500",
            "--resolution=1024",
            "--noise_offset=0.03",
            "--mixed_precision=fp16",
            "--use_8bit_adam",
            "--seed=0",
        ],
        REPO_ROOT,
        title=f"Style {name!r} — stage 2 (style retrain, → {out2.relative_to(REPO_ROOT)})",
    )


def train_content(name: str) -> None:
    image_path = DATA_CONTENT / f"{name}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    out = LORA_CONTENT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_training(
        [
            "accelerate",
            "launch",
            "train_consislora.py",
            "--instance_prompt=A [c]",
            f"--image_path={image_path}",
            f"--output_dir={out}",
            "--start_x0_loss_steps=500",
            "--rank=64",
            "--lr=2e-4",
            "--second_lr=1e-4",
            f"--max_train_steps={MAX_TRAIN_STEPS_CONTENT}",
            "--checkpointing_steps=500",
            "--resolution=1024",
            "--mixed_precision=fp16",
            "--use_8bit_adam",
            "--seed=0",
        ],
        REPO_ROOT,
        title=f"Content {name!r} (A [c], → {out.relative_to(REPO_ROOT)})",
    )


# %% [markdown]
# ## A) Train all style LoRAs
#
# Runs two-stage training per `STYLE_NAMES` (see `script/train_style.sh`). The outer `tqdm` bar advances once per **style**; inside each `accelerate` run, `train_consislora.py` prints its own **step** `tqdm` to the cell output. `PYTHONUNBUFFERED=1` is set for the child process so that inner progress line updates appear live in Jupyter when possible.

# %%
for _style in tqdm(STYLE_NAMES, desc="[A] Style LoRAs", unit="style"):
    train_style_two_stage(_style)

# %% [markdown]
# ## C) Train content LoRAs (rio, uit)
#
# Run after **A** so the SDXL pipeline is not loaded yet (saves GPU memory for `accelerate`). Same progress display as in **A** (outer `tqdm` + step bar from the training script).

# %%
for _c in tqdm(CONTENT_NAMES, desc="[C] Content LoRAs", unit="run"):
    train_content(_c)

# %% [markdown]
# ## B) Style-only inference (each trained style × p1, p2)
#
# LoRA from `lora-weights/style_retrain/{name}/`. Prompt: *base* + `, in the style of [v]`. `lora_scaling = [0., 1.]` (see `README` / `inference_notebook.py`).

# %%
pipeline = build_sdxl_pipeline()

p1 = f"{P1_BASE}, in the style of [v]"
p2 = f"{P2_BASE}, in the style of [v]"
style_prompts = [("p1", p1), ("p2", p2)]

# 2 rows (p1, p2) x 5 cols (styles)
ncols = len(STYLE_NAMES)
nrows = len(style_prompts)
style_only_images: list[Image.Image] = []
cell_titles: list[str] = []
generator = torch.manual_seed(SEED)

for (_tag, _prompt) in style_prompts:
    for sname in STYLE_NAMES:
        lora = str(LORA_STYLE_FINAL / sname)
        if not Path(lora).is_dir() or not any(Path(lora).iterdir()):
            raise FileNotFoundError(f"Missing or empty style LoRA directory: {lora}")
        pipeline.unload_lora_checkpoint()
        pipeline.load_lora_checkpoint(
            content_image_lora_path=None, style_image_lora_path=lora
        )
        img = pipeline(
            prompt=_prompt,
            lora_scaling=[0.0, STYLE_LORA_SCALE],
            guidance_scale=GUIDANCE,
            num_inference_steps=INFER_STEPS,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]
        style_only_images.append(img)
        cell_titles.append(f"{_tag} | {sname}")

show_grid_pil(
    style_only_images,
    nrows,
    ncols,
    "Style LoRA only — row: p1 / p2; column: rococo, baroque, monet, shinkai, ukyoe",
    cell_titles=cell_titles,
    figsize_per_cell=(2.8, 2.6),
)
OUT_INFER.mkdir(parents=True, exist_ok=True)
for i, (im, ti) in enumerate(zip(style_only_images, cell_titles)):
    im.save(OUT_INFER / f"style_only_{i:02d}_{ti.replace(' | ', '_').replace(' ', '_')}.png")

# %% [markdown]
# ## D) Content × style inference (`a [c] in the style of [v]`)
#
# Rows: `rio`, `uit`. Columns: `STYLE_NAMES`. Reuses `pipeline` from **B**; if you only run **D**, this cell will build the pipeline for you.

# %%
if "pipeline" not in globals():
    pipeline = build_sdxl_pipeline()

CROSS_PROMPT = "a [c] in the style of [v]"
cross_images: list[Image.Image] = []
cross_titles: list[str] = []
gen2 = torch.manual_seed(SEED)

for cname in CONTENT_NAMES:
    cpath = str(LORA_CONTENT / cname)
    for sname in STYLE_NAMES:
        spath = str(LORA_STYLE_FINAL / sname)
        pipeline.unload_lora_checkpoint()
        pipeline.load_lora_checkpoint(
            content_image_lora_path=cpath, style_image_lora_path=spath
        )
        out = pipeline(
            prompt=CROSS_PROMPT,
            lora_scaling=[1.0, STYLE_LORA_SCALE],
            guidance_scale=GUIDANCE,
            num_inference_steps=INFER_STEPS,
            num_images_per_prompt=1,
            generator=gen2,
        ).images[0]
        cross_images.append(out)
        cross_titles.append(f"{cname} × {sname}")

show_grid_pil(
    cross_images,
    2,
    len(STYLE_NAMES),
    f'Content×style — "{CROSS_PROMPT}" (row: rio, uit; col: styles)',
    cell_titles=cross_titles,
    figsize_per_cell=(2.8, 2.6),
)
for i, (im, ti) in enumerate(zip(cross_images, cross_titles)):
    im.save(OUT_INFER / f"cross_{i:02d}_{ti.replace(' × ', '_x_')}.png")

# %% [markdown]
# ## (Optional) Reference thumbnails — training images
#
# For documentation / sanity check.

# %%
fig, axes = plt.subplots(2, max(len(STYLE_NAMES), len(CONTENT_NAMES)), figsize=(14, 4))
# Row 0: styles
for j, s in enumerate(STYLE_NAMES):
    p = DATA_STYLE / f"{s}.jpg"
    if p.is_file():
        axes[0, j].imshow(load_pil_image(str(p), target_size=384))
    axes[0, j].set_title(f"style: {s}", fontsize=8)
    axes[0, j].axis("off")
for j in range(len(STYLE_NAMES), axes.shape[1]):
    axes[0, j].axis("off")
# Row 1: content
for j, c in enumerate(CONTENT_NAMES):
    p = DATA_CONTENT / f"{c}.jpg"
    if p.is_file():
        axes[1, j].imshow(load_pil_image(str(p), target_size=384))
    axes[1, j].set_title(f"content: {c}", fontsize=8)
    axes[1, j].axis("off")
for j in range(len(CONTENT_NAMES), axes.shape[1]):
    axes[1, j].axis("off")
plt.suptitle("Reference images (data_demo)", fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## E) Upload `lora-weights` to Hugging Face Hub
#
# Pushes the entire local `lora-weights/` directory to **`paingoat/ConsisLoRA-test`** under remote path `lora-weights/`, using [`HfApi.upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder) (one commit). The model repo is created if it does not exist.
#
# **Auth:** create a file `.env` in the repository root (see `.env.example`, not committed — listed in `.gitignore`) with:
#
# ```env
# HF_TOKEN=hf_...
# ```
#
# Use a token with **write** access from [Hugging Face settings](https://huggingface.co/settings/tokens). For very large trees, consider [`upload_large_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder) in a separate script.

# %%
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(REPO_ROOT / ".env")
hf_token = os.environ.get("HF_TOKEN", "").strip()
if not hf_token:
    raise ValueError(
        "Missing HF_TOKEN. Add a `.env` file in the repo root (see .env.example) with a write token from "
        "https://huggingface.co/settings/tokens"
    )

_lora_root = REPO_ROOT / "lora-weights"
if not _lora_root.is_dir():
    raise FileNotFoundError(f"No folder to upload: {_lora_root}")
if not any(_lora_root.iterdir()):
    raise FileNotFoundError(f"Folder is empty (train first): {_lora_root}")

_api = HfApi(token=hf_token)
_api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
_commit = _api.upload_folder(
    folder_path=str(_lora_root),
    repo_id=HF_REPO_ID,
    repo_type="model",
    path_in_repo="lora-weights",
    commit_message="Upload lora-weights (ConsisLoRA batch)",
)
print(_commit)
print(f"Hub tree: https://huggingface.co/{HF_REPO_ID}/tree/main/lora-weights")
