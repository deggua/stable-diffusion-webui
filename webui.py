import argparse
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))

# use current directory as SD dir if it has related files, otherwise parent dir of script as stated in guide
sd_path = os.path.abspath('.') if os.path.exists('./ldm/models/diffusion/ddpm.py') else os.path.dirname(script_path)

# add parent directory to path; this is where Stable diffusion repo should be
path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion'),
    ('../../taming-transformers', 'taming', 'Taming Transformers')
]
for d, must_exist, what in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        sys.path.append(os.path.join(script_path, d))

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import gradio.utils
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin, ImageFilter, ImageOps
from torch import autocast
import mimetypes
import random
import math
import html
import time
import json
import traceback
from datetime import datetime
from enum import IntEnum
from typing import Optional
from collections import namedtuple
from contextlib import nullcontext
import signal
import tqdm
import re
import threading
import base64
from io import BytesIO

import k_diffusion.sampling
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\\|?*\n'
config_filename = "config.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(sd_path, "configs/stable-diffusion/v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=os.path.join(sd_path, "models/ldm/stable-diffusion-v1/model.ckpt"), help="path to checkpoint of model",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default='GFPGANv1.3.pth')
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings dirtectory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrficing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrficing a lot of speed for very low VRM usage")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="a workaround test; may help with speed in you use --lowvram")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
cmd_opts = parser.parse_args()

cpu = torch.device("cpu")
gpu = torch.device("cuda")
device = gpu if torch.cuda.is_available() else cpu
batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
queue_lock = threading.Lock()


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class State:
    interrupted = False
    job = ""

    def interrupt(self):
        self.interrupted = True


state = State()

if not cmd_opts.share:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

# TODO: remove this when they fix the bug
# replace bugged gradio function
def update(**kwargs) -> dict:
    kwargs["__type__"] = "update"
    return kwargs

gr.update = update

SamplerData = namedtuple('SamplerData', ['name', 'constructor'])
samplers = [
    *[SamplerData(x[0], lambda funcname=x[1]: KDiffusionSampler(funcname)) for x in [
        ('Euler', 'sample_euler'),
        ('Euler ancestral', 'sample_euler_ancestral'),
        ('LMS', 'sample_lms'),
        ('Heun', 'sample_heun'),
        ('DPM2', 'sample_dpm_2'),
        ('DPM2 a', 'sample_dpm_2_ancestral'),
    ] if hasattr(k_diffusion.sampling, x[1])],
    SamplerData('DDIM', lambda: VanillaStableDiffusionSampler(DDIMSampler)),
    SamplerData('PLMS', lambda: VanillaStableDiffusionSampler(PLMSSampler)),
]
samplers_for_img2img = [x for x in samplers if x.name != 'PLMS']

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    realesrgan_models = [
        RealesrganModelInfo(
            name="Real-ESRGAN 2x plus",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            netscale=2, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        ),
        RealesrganModelInfo(
            name="Real-ESRGAN 4x plus",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        ),
        RealesrganModelInfo(
            name="Real-ESRGAN 4x plus anime 6B",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        ),
    ]
    have_realesrgan = True
except Exception:
    print("Error importing Real-ESRGAN:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
    have_realesrgan = False

sd_upscalers = {
    "RealESRGAN": lambda img: upscale_with_realesrgan(img, 2, 0),
    "Lanczos": lambda img: img.resize((img.width * 2, img.height * 2), resample=LANCZOS),
    "None": lambda img: img
}


def gfpgan_model_path():
    places = [script_path, '.', os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models')]
    files = [cmd_opts.gfpgan_model] + [os.path.join(dirname, cmd_opts.gfpgan_model) for dirname in places]
    found = [x for x in files if os.path.exists(x)]

    if len(found) == 0:
        raise Exception("GFPGAN model not found in paths: " + ", ".join(files))

    return found[0]


def gfpgan():
    return GFPGANer(model_path=gfpgan_model_path(), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

def gfpgan_fix_faces(gfpgan_model, np_image):
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan_model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    return np_image

have_gfpgan = False
try:
    model_path = gfpgan_model_path()

    if os.path.exists(cmd_opts.gfpgan_dir):
        sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
    from gfpgan import GFPGANer

    have_gfpgan = True
except Exception:
    print("Error setting up GFPGAN:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)



class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args

    data = None
    data_labels = {
        "outdir_samples": OptionInfo("", "Output dictectory for images; if empty, defaults to two directories below"),
        "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output dictectory for txt2img images'),
        "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output dictectory for img2img images'),
        "outdir_extras_samples": OptionInfo("outputs/post-images", 'Output dictectory for images from post-processing'),
        "outdir_grids": OptionInfo("", "Output dictectory for grids; if empty, defaults to two directories below"),
        "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output dictectory for txt2img grids'),
        "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output dictectory for img2img grids'),
        "save_to_dirs": OptionInfo(False, "When writing images/grids, create a directory with name derived from the prompt"),
        "save_to_dirs_prompt_len": OptionInfo(10, "When using above, how many words from prompt to put into directory name", gr.Slider, {"minimum": 1, "maximum": 32, "step": 1}),
        "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button"),
        "samples_save": OptionInfo(True, "Save indiviual samples"),
        "samples_format": OptionInfo('png', 'File format for indiviual samples'),
        "grid_save": OptionInfo(True, "Save image grids"),
        "return_grid": OptionInfo(True, "Show grid in results for web"),
        "grid_format": OptionInfo('png', 'File format for grids'),
        "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
        "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
        "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
        "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
        "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),
        "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
        "font": OptionInfo("arial.ttf", "Font for image grids  that have text"),
        "prompt_matrix_add_to_start": OptionInfo(True, "In prompt matrix, add the variable combination of text to the start of the prompt, rather than the end"),
        "sd_upscale_upscaler_index": OptionInfo("RealESRGAN", "Upscaler to use for SD upscale", gr.Radio, {"choices": list(sd_upscalers.keys())}),
        "sd_upscale_overlap": OptionInfo(64, "Overlap for tiles for SD upscale. The smaller it is, the less smooth transition from one tile to another", gr.Slider, {"minimum": 0, "maximum": 256, "step": 16}),
        "enable_history": OptionInfo(True, "Enable saving prompt history and thumbnails for txt2img"),
        "history_thumbnail_res": OptionInfo(64, "Set the resolution (in pixels) of thumbnails for the History tab (restart and refresh required)", gr.Slider, {"minimum": 64, "maximum": 128, "step": 32}),
        # TODO: Remove this and make it part of the Syntactic prompts settings
        "enable_emphasis": OptionInfo(True, "Use (text) to make model pay more attention to text text and [text] to make it pay less attention"),
        "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
    }

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


module_in_gpu = None


def setup_for_low_vram(sd_model):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(gpu)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods
    def first_stage_model_encode_wrap(self, encoder, x):
        send_me_to_gpu(self, None)
        return encoder(x)

    def first_stage_model_decode_wrap(self, decoder, z):
        send_me_to_gpu(self, None)
        return decoder(z)

    # remove three big modules, cond, first_stage, and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = None, None, None
    sd_model.to(device)
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = stored

    # register hooks for those the first two models
    sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = lambda x, en=sd_model.first_stage_model.encode: first_stage_model_encode_wrap(sd_model.first_stage_model, en, x)
    sd_model.first_stage_model.decode = lambda z, de=sd_model.first_stage_model.decode: first_stage_model_decode_wrap(sd_model.first_stage_model, de, z)
    parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    if cmd_opts.medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)


def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False):
    if short_filename or prompt is None or seed is None:
        file_decoration = ""
    elif opts.save_to_dirs:
        file_decoration = f"-{seed}"
    else:
        file_decoration = f"-{seed}-{sanitize_filename_part(prompt)[:128]}"

    if extension == 'png' and opts.enable_pnginfo and info is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Parameters", info)
    else:
        pnginfo = None

    if opts.save_to_dirs and not no_prompt:
        words = re.findall(r'\w+', prompt or "")
        if len(words) == 0:
            words = ["empty"]

        dirname = " ".join(words[0:opts.save_to_dirs_prompt_len])
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    filecount = len(os.listdir(path))
    fullfn = "a.png"
    fullfn_without_extension = "a"
    for i in range(100):
        fn = f"{filecount:05}" if basename == '' else f"{basename}-{filecount:04}"
        fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
        fullfn_without_extension = os.path.join(path, f"{fn}{file_decoration}")
        if not os.path.exists(fullfn):
            break

    image.save(fullfn, quality=opts.jpeg_quality, pnginfo=pnginfo)

    target_side_length = 4000
    oversize = image.width > target_side_length or image.height > target_side_length
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > 4 * 1024 * 1024):
        ratio = image.width / image.height

        if oversize and ratio > 1:
            image = image.resize((target_side_length, image.height * target_side_length // image.width), LANCZOS)
        elif oversize:
            image = image.resize((image.width * target_side_length // image.height, target_side_length), LANCZOS)

        image.save(f"{fullfn_without_extension}.jpg", quality=opts.jpeg_quality, pnginfo=pnginfo)

    if opts.save_txt and info is not None:
        with open(f"{fullfn_without_extension}.txt", "w", encoding="utf-8") as file:
            file.write(info + "\n")

def sanitize_filename_part(text):
    return text.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]


def plaintext_to_html(text, klass=None):
    if klass is None:
        text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
    else:
        text = "".join([f"<p class=\"{klass}\">{html.escape(x)}</p>\n" for x in text.split('\n')])
    return text


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    now = tile_w - overlap  # non-overlap width
    noh = tile_h - overlap

    cols = math.ceil((w - overlap) / now)
    rows = math.ceil((h - overlap) / noh)

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = row * noh

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = col * now

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


def draw_grid_annotations(im, width, height, hor_texts, ver_texts):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype(opts.font, fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = width * 3 // 4 if len(ver_texts) > 0 else 0

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result


def draw_prompt_matrix(im, width, height, all_prompts):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)

    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]

    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    return draw_grid_annotations(im, width, height, hor_texts, ver_texts)


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[GridAnnotation(x_label(x))] for x in xs]

    for y in ys:
        for x in xs:
            res.append(cell(x, y))

    grid = image_grid(res, rows=len(ys))
    grid = draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    return grid


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return wrap_gradio_call(f)


def wrap_gradio_call(func):
    def f(*args, **kwargs):
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            print("Error completing request", file=sys.stderr)
            print("Arguments:", args, kwargs, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

            res = [None, '', f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t

        # last item is always HTML
        res[-1] = res[-1] + f"<p class='performance'>Time taken: {elapsed:.2f}s</p>"

        state.interrupted = False

        return tuple(res)

    return f


class StableDiffusionModelHijack:
    ids_lookup = {}
    word_embeddings = {}
    word_embeddings_checksums = {}
    fixes = None
    comments = []
    dir_mtime = None

    def load_textual_inversion_embeddings(self, dirname, model):
        mt = os.path.getmtime(dirname)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        tokenizer = model.cond_stage_model.tokenizer

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = torch.load(path)
            param_dict = data['string_to_param']
            if hasattr(param_dict, '_parameters'):
                param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1].reshape(768)
            self.word_embeddings[name] = emb
            self.word_embeddings_checksums[name] = f'{const_hash(emb) & 0xffff:04x}'

            ids = tokenizer([name], add_special_tokens=False)['input_ids'][0]

            first_id = ids[0]
            if first_id not in self.ids_lookup:
                self.ids_lookup[first_id] = []
            self.ids_lookup[first_id].append((ids, name))

        for fn in os.listdir(dirname):
            try:
                process_file(os.path.join(dirname, fn), fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} text inversion embeddings.")

    def hijack(self, m):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings

        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)


class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.max_length = wrapped.max_length
        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def forward(self, text):
        self.hijack.fixes = []
        self.hijack.comments = []
        remade_batch_tokens = []
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id
        maxlen = self.wrapped.max_length - 2
        used_custom_terms = []

        cache = {}
        batch_tokens = self.wrapped.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        batch_multipliers = []
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes, multipliers = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []
                multipliers = []
                mult = 1.0

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    possible_matches = self.hijack.ids_lookup.get(token, None)

                    mult_change = self.token_mults.get(token) if opts.enable_emphasis else None
                    if mult_change is not None:
                        mult *= mult_change
                    elif possible_matches is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                    else:
                        found = False
                        for ids, word in possible_matches:
                            if tokens[i:i + len(ids)] == ids:
                                fixes.append((len(remade_tokens), word))
                                remade_tokens.append(777)
                                multipliers.append(mult)
                                i += len(ids) - 1
                                found = True
                                used_custom_terms.append((word, self.hijack.word_embeddings_checksums[word]))
                                break

                        if not found:
                            remade_tokens.append(token)
                            multipliers.append(mult)

                    i += 1

                if len(remade_tokens) > maxlen - 2:
                    vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                    ovf = remade_tokens[maxlen - 2:]
                    overflowing_words = [vocab.get(int(x), "") for x in ovf]
                    overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))

                    self.hijack.comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen - 2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

            multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
            multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

            remade_batch_tokens.append(remade_tokens)
            self.hijack.fixes.append(fixes)
            batch_multipliers.append(multipliers)

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used custom terms: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.wrapped.transformer(input_ids=tokens)
        z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(np.array(batch_multipliers)).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


class EmbeddingsWithFixes(nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is not None:
            for fixes, tensor in zip(batch_fixes, inputs_embeds):
                for offset, word in fixes:
                    tensor[offset] = self.embeddings.word_embeddings[word]

        return inputs_embeds


class StableDiffusionProcessing:
    def __init__(self, outpath_samples=None, outpath_grids=None, prompt="", seed=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, prompt_matrix=False, use_GFPGAN=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, overlay_images=None, negative_prompt=None):
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.negative_prompt: str = (negative_prompt or "")
        self.seed: int = seed
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.prompt_matrix: bool = prompt_matrix
        self.use_GFPGAN: bool = use_GFPGAN
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params
        self.overlay_images = overlay_images
        self.paste_to = None

    def init(self):
        pass

    def sample(self, x, conditioning, unconditional_conditioning):
        raise NotImplementedError()


def p_sample_ddim_hook(sampler_wrapper, x_dec, cond, ts, *args, **kwargs):
    if sampler_wrapper.mask is not None:
        img_orig = sampler_wrapper.sampler.model.q_sample(sampler_wrapper.init_latent, ts)
        x_dec = img_orig * sampler_wrapper.mask + sampler_wrapper.nmask * x_dec

    return sampler_wrapper.orig_p_sample_ddim(x_dec, cond, ts, *args, **kwargs)


class VanillaStableDiffusionSampler:
    def __init__(self, constructor):
        self.sampler = constructor(sd_model)
        self.orig_p_sample_ddim = self.sampler.p_sample_ddim if hasattr(self.sampler, 'p_sample_ddim') else None
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)

        # existing code fails with cetin step counts, like 9
        try:
            self.sampler.make_schedule(ddim_num_steps=p.steps, verbose=False)
        except Exception:
            self.sampler.make_schedule(ddim_num_steps=p.steps+1, verbose=False)

        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(device), noise=noise)

        self.sampler.p_sample_ddim = lambda x_dec, cond, ts, *args, **kwargs: p_sample_ddim_hook(self, x_dec, cond, ts, *args, **kwargs)
        self.mask = p.mask
        self.nmask = p.nmask
        self.init_latent = p.init_latent

        samples = self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning)

        return samples


    def sample(self, p: StableDiffusionProcessing, x, conditioning, unconditional_conditioning):
        samples_ddim, _ = self.sampler.sample(S=p.steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x)
        return samples_ddim


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if batch_cond_uncond:
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            denoised = uncond + (cond - uncond) * cond_scale
        else:
            uncond = self.inner_model(x, sigma, cond=uncond)
            cond = self.inner_model(x, sigma, cond=cond)
            denoised = uncond + (cond - uncond) * cond_scale

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        return denoised


def extended_trange(*args, **kwargs):
    for x in tqdm.trange(*args, desc=state.job, **kwargs):
        if state.interrupted:
            break

        yield x


class KDiffusionSampler:
    def __init__(self, funcname):
        self.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)
        sigmas = self.model_wrap.get_sigmas(p.steps)
        noise = noise * sigmas[p.steps - t_enc - 1]

        xi = x + noise

        sigma_sched = sigmas[p.steps - t_enc - 1:]

        self.model_wrap_cfg.mask = p.mask
        self.model_wrap_cfg.nmask = p.nmask
        self.model_wrap_cfg.init_latent = p.init_latent

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        return self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False)

    def sample(self, p: StableDiffusionProcessing, x, conditioning, unconditional_conditioning):
        sigmas = self.model_wrap.get_sigmas(p.steps)
        x = x * sigmas[0]

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        def cb(d):
            n = d['i']
            img = d['denoised']

            x_samples_ddim = sd_model.decode_first_stage(img)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                image.save(f'a/{n}.png')

        samples_ddim = self.func(self.model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False)
        return samples_ddim

class Processed:
    def __init__(self, p: StableDiffusionProcessing, images, seed, info):
        self.images = images
        self.prompt = p.prompt
        self.seed = seed
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler = samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps

    def js(self):
        obj = {
            "prompt": self.prompt,
            "seed": int(self.seed),
            "width": self.width,
            "height": self.height,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
        }

        return json.dumps(obj)

class OutputInfo:
    def __init__(self, prompt: str, params: str, comments: str):
        self.prompt = prompt.strip()
        self.params = params.strip()
        self.comments = comments.strip()

    def __str__(self):
        return '\n'.join([self.prompt, self.params, self.comments])

    def html(self) -> str:
        return f'''
        {plaintext_to_html(self.prompt, "prompt-info")}<br>
        {plaintext_to_html(self.params, "params-info")}
        {plaintext_to_html(self.comments, "comments-info")}
        '''


def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    prompt = p.prompt
    model = sd_model

    assert p.prompt is not None
    torch_gc()

    seed = int(random.randrange(4294967294) if p.seed == -1 else p.seed)

    os.makedirs(p.outpath_samples, exist_ok=True)
    os.makedirs(p.outpath_grids, exist_ok=True)

    comments = []

    prompt_matrix_parts = []
    if p.prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1 << n)]

            if opts.prompt_matrix_add_to_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(", ".join(selected_prompts))

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")
    else:
        all_prompts = p.batch_size * p.n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    generation_params = {
        "Steps": p.steps,
        "Sampler": samplers[p.sampler_index].name,
        "CFG": p.cfg_scale,
        "Seed": seed,
        "GFPGAN": ("GFPGAN" if p.use_GFPGAN else None)
    }

    if p.extra_generation_params is not None:
        generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    def infotext():
        return OutputInfo(prompt, generation_params_text, "".join(["\n\n" + x for x in comments]))

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, model)

    output_images = []
    precision_scope = autocast if cmd_opts.precision == "autocast" else nullcontext
    ema_scope = (nullcontext if cmd_opts.lowvram else model.ema_scope)
    with torch.no_grad(), precision_scope("cuda"), ema_scope():
        p.init()

        for n in range(p.n_iter):
            if state.interrupted:
                break

            prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]

            uc = model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
            c = model.get_learned_conditioning(prompts)

            if len(model_hijack.comments) > 0:
                comments += model_hijack.comments

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], seeds=seeds)

            if p.n_iter > 1:
                state.job = f"Batch {n+1} out of {p.n_iter}"

            samples_ddim = p.sample(x=x, conditioning=c, unconditional_conditioning=uc)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.use_GFPGAN:
                    torch_gc()

                    gfpgan_model = gfpgan()
                    x_sample = gfpgan_fix_faces(gfpgan_model, x_sample)

                image = Image.fromarray(x_sample)

                if p.overlay_images is not None and i < len(p.overlay_images):
                    overlay = p.overlay_images[i]

                    if p.paste_to is not None:
                        x, y, w, h = p.paste_to
                        base_image = Image.new('RGBA', (overlay.width, overlay.height))
                        image = resize_image(1, image, w, h)
                        base_image.paste(image, (x, y))
                        image = base_image

                    image = image.convert('RGBA')
                    image.alpha_composite(overlay)
                    image = image.convert('RGB')

                if opts.samples_save and not p.do_not_save_samples:
                    save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=str(infotext()))

                output_images.append(image)

        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            return_grid = opts.return_grid

            if p.prompt_matrix:
                grid = image_grid(output_images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))

                try:
                    grid = draw_prompt_matrix(grid, p.width, p.height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                return_grid = True
            else:
                grid = image_grid(output_images, p.batch_size)

            if return_grid:
                output_images.insert(0, grid)

            if opts.grid_save:
                save_image(grid, p.outpath_grids, "grid", seed, prompt, opts.grid_format, info=str(infotext()), short_filename=not opts.grid_extended_filename)

    torch_gc()
    return Processed(p, output_images, seed, infotext())


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def init(self):
        self.sampler = samplers[self.sampler_index].constructor()

    def sample(self, x, conditioning, unconditional_conditioning):
        samples_ddim = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        return samples_ddim


def txt2img(
    prompt: str,
    negative_prompt: Optional[str],
    steps: int,
    sampler_index: int,
    use_GFPGAN: bool,
    prompt_matrix: bool,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    seed: int,
    height: int,
    width: int,
    code: str):

    p = StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=use_GFPGAN,
    )

    if code != '' and cmd_opts.allow_code:
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        display_result_data = [[], -1, ""]

        def display(imgs, s=display_result_data[1], i=display_result_data[2]):
            display_result_data[0] = imgs
            display_result_data[1] = s
            display_result_data[2] = i

        from types import ModuleType
        compiled = compile(code, '', 'exec')
        module = ModuleType("testmodule")
        module.__dict__.update(globals())
        module.p = p
        module.display = display
        exec(compiled, module.__dict__)

        processed = Processed(p, *display_result_data)
    else:
        processed = process_images(p)

    return processed.images, processed.js(), processed.info.html()

# TODO: make Save call this function, modify it a bit to work better in our implementation
def save_files(js_data, images):
    import csv

    os.makedirs(opts.outdir_save, exist_ok=True)

    filenames = []

    data = json.loads(js_data)

    with open("log/log.csv", "a", encoding="utf8", newline='') as file:
        import time
        import base64

        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename"])

        filename_base = str(int(time.time() * 1000))
        for i, filedata in enumerate(images):
            filename = filename_base + ("" if len(images) == 1 else "-" + str(i + 1)) + ".png"
            filepath = os.path.join(opts.outdir_save, filename)

            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            with open(filepath, "wb") as imgfile:
                imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

            filenames.append(filename)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0]])

    return '', '', plaintext_to_html(f"Saved: {filenames[0]}")


def get_crop_region(mask, pad=0):
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:,i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:,i] == 0).all():
            break
        crop_right += 1


    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


def fill(image, mask):
    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, mask=None, mask_blur=4, inpainting_fill=0, inpaint_full_res=True, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.image_mask = mask
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.mask = None
        self.nmask = None

    def init(self):
        self.sampler = samplers_for_img2img[self.sampler_index].constructor()
        crop_region = None

        if self.image_mask is not None:
            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur)).convert('L')

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = get_crop_region(np.array(mask), 64)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                self.image_mask = resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                self.mask_for_overlay = self.image_mask

            self.overlay_images = []


        imgs = []

        if not self.init_images or None in self.init_images:
            raise Exception('No input image provided')

        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, self.mask_for_overlay)

                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = resize_image(2, image, self.width, self.height)

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(device)

        self.init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))

        if self.image_mask is not None:
            latmask = self.image_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float64), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(device).type(sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(device).type(sd_model.dtype)

    def sample(self, x, conditioning, unconditional_conditioning):

        if self.mask is not None:
            if self.inpainting_fill == 2:
                x = x * self.mask + create_random_tensors(x.shape[1:], [self.seed + x + 1 for x in range(x.shape[0])]) * self.nmask
            elif self.inpainting_fill == 3:
                x = x * self.mask

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        return samples

class Img2Img_Modes(IntEnum):
    CLASSIC = 0
    INPAINT = 1
    LOOPBACK = 2
    UPSCALE = 3

def img2img(
    prompt: str,
    init_img,
    init_img_with_mask,
    steps: int,
    sampler_index: int,
    mask_blur: int,
    inpainting_fill: int,
    use_GFPGAN: bool,
    prompt_matrix,
    mode: int,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    denoising_strength: float,
    seed: int,
    height: int,
    width: int,
    resize_mode: int,
    upscaler_name: str,
    upscale_overlap: int,
    inpaint_full_res: bool):

    is_classic = mode == Img2Img_Modes.CLASSIC
    is_inpaint = mode == Img2Img_Modes.INPAINT
    is_loopback = mode == Img2Img_Modes.LOOPBACK
    is_upscale = mode == Img2Img_Modes.UPSCALE

    if is_inpaint:
        image = init_img_with_mask['image']
        mask = init_img_with_mask['mask']
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=use_GFPGAN,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        extra_generation_params={"DNS": denoising_strength}
    )

    if is_loopback:
        output_images, info = None, None
        history = []
        initial_seed = None
        initial_info = None

        for i in range(n_iter):
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            state.job = f"Batch {i + 1} out of {n_iter}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.init_images = [processed.images[0]]
            p.seed = processed.seed + 1
            p.denoising_strength = max(p.denoising_strength * 0.95, 0.1)
            history.append(processed.images[0])

        grid = image_grid(history, batch_size, rows=1)

        save_image(grid, p.outpath_grids, "grid", initial_seed, prompt, opts.grid_format, info=str(info), short_filename=not opts.grid_extended_filename)

        processed = Processed(p, history, initial_seed, initial_info)

    elif is_upscale:
        initial_seed = None
        initial_info = None

        upscaler = sd_upscalers.get(upscaler_name, next(iter(sd_upscalers.values())))
        img = upscaler(init_img)

        torch_gc()

        grid = split_grid(img, tile_w=width, tile_h=height, overlap=upscale_overlap)

        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_results = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / p.batch_size)
        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} in a total of {batch_count} batches.")

        for i in range(batch_count):
            p.init_images = work[i * p.batch_size:(i + 1) * p.batch_size]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images

        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index] if image_index<len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = combine_grid(grid)

        save_image(combined_image, p.outpath_grids, "grid", initial_seed, prompt, opts.grid_format, info=str(initial_info), short_filename=not opts.grid_extended_filename)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

    else:
        processed = process_images(p)

    return processed.images, processed.js(), processed.info.html()

def upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index):
    info = realesrgan_models[RealESRGAN_model_index]

    model = info.model()
    upsampler = RealESRGANer(
        scale=info.netscale,
        model_path=info.location,
        model=model,
        half=True
    )

    upsampled = upsampler.enhance(np.array(image), outscale=RealESRGAN_upscaling)[0]

    image = Image.fromarray(upsampled)
    return image

# Post Processing
def run_postprocessing(image, use_GFPGAN: bool, strength_GFPGAN: float, use_ESRGAN: bool, model_ESRGAN: int, scale_ESRGAN: float):
    torch_gc()

    if not image:
        raise Exception('No input image provided')

    image = image.convert("RGB")

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    if have_gfpgan and use_GFPGAN and strength_GFPGAN > 0:
        gfpgan_model = gfpgan()

        restored_img = gfpgan_fix_faces(gfpgan_model, np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if strength_GFPGAN < 1.0:
            res = Image.blend(image, res, strength_GFPGAN)

        image = res

    if have_realesrgan and use_ESRGAN and scale_ESRGAN != 1.0:
        image = upscale_with_realesrgan(image, scale_ESRGAN, model_ESRGAN)

    save_image(image, outpath, "", None, '', opts.samples_format, short_filename=True, no_prompt=True)

    return [image, '']


# ---- FRONTEND CODE STARTS HERE ---- #

# Image Info
# TODO: add embedded info to JPGs and add JPG parsing support
def run_image_info(image):
    info = ''
    for key, text in image.info.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip() + "\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return info

# Settings
opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

def run_settings(*args):
    for key, value in zip(opts.data_labels.keys(), args):
        opts.data[key] = value

    opts.save(config_filename)

    return plaintext_to_html(f'Settings saved @ {datetime.now().strftime("%I:%M:%S")}')


def create_setting_component(key):
    def fun():
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    info = opts.data_labels[key]
    t = type(info.default)

    if info.component is not None:
        item = info.component(label=info.label, value=fun, **(info.component_args or {}))
    elif t == str:
        item = gr.Textbox(label=info.label, value=fun, lines=1)
    elif t == int:
        item = gr.Number(label=info.label, value=fun)
    elif t == bool:
        item = gr.Checkbox(label=info.label, value=fun)
    else:
        raise Exception(f'bad options item type: {str(t)} for key {key}')

    return item

# this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
try:
    from transformers import logging
    logging.set_verbosity_error()
except Exception:
    pass

sd_config = OmegaConf.load(cmd_opts.config)
sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
sd_model = (sd_model if cmd_opts.no_half else sd_model.half())

if cmd_opts.lowvram or cmd_opts.medvram:
    setup_for_low_vram(sd_model)
else:
    sd_model = sd_model.to(device)

model_hijack = StableDiffusionModelHijack()
model_hijack.hijack(sd_model)

class HistoryEntry:
    def __init__(self, images: Image=None, description: str=None):
        self.images = images
        self.description = description

    class HistoryImage:
        def __init__(self, image: Image):
            self.image = image

        @staticmethod
        def make_thumbnail(image: Image) -> Image:
            maxres = opts.history_thumbnail_res
            if image.width >= image.height:
                thumbnail_dim = (maxres, image.height * maxres // image.width)
            else:
                thumbnail_dim = (image.width * maxres // image.height, maxres)

            return image.resize(thumbnail_dim, resample=Image.Resampling.BICUBIC)

        def html(self) -> str:
            thumb = self.make_thumbnail(self.image)
            buffer = BytesIO()

            thumb.save(buffer, format='JPEG', quality=60)
            img_b64 = 'data:image/jpeg;charset=utf-8;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_tag = f'<img id="history_img" src="{img_b64}"/>'
            img_div = f'<div id="history_thumb">{img_tag}</div>'

            return img_div

    class HistoryDescription:
        def __init__(self, description=None):
            self.description = description
        
        def html(self):
            return f'<div id="history_description">{self.description}</div>'

    def html(self):
        history_images = [HistoryEntry.HistoryImage(img) for img in self.images]
        thumbnails_contents = ''.join([x.html() for x in history_images])
        thumbnails = f'<div id="history_thumbnails">{thumbnails_contents}</div>'

        history_description = HistoryEntry.HistoryDescription(self.description).html()

        contents = thumbnails + history_description
        return f'<div id="history_row">{contents}</div>'

def save_to_history(images, html):
    outdir = opts.outdir or "outputs/"
    history_file_path = os.path.join(outdir, 'history.html')

    # Skip the grid image if return grids is on
    if opts.return_grid and len(images) > 1:
        entry = HistoryEntry(images=images[1:4], description=html)
    else:
        entry = HistoryEntry(images=images[:3], description=html)

    # TODO: provide some max size limit and overwrite when at max size
    # TODO(maybe): add timestamp
    # TODO(maybe): provide link to file on disk if file was saved
    with open(history_file_path, 'a', encoding='utf-8') as fd:
        fd.write(entry.html())

def read_history():
    outdir = opts.outdir or "outputs/"
    history_file_path = os.path.join(outdir, 'history.html')

    if os.path.exists(history_file_path):
        history_contents = open(history_file_path, 'r', encoding='utf-8').read()
        history_container = f'<div id="history" style="display: flex; flex-direction: column-reverse;">{history_contents}</div>'
        return history_container
    else:
        return plaintext_to_html('No history saved')

def erase_history():
    outdir = opts.outdir or "outputs/"
    history_file_path = os.path.join(outdir, 'history.html')
    os.remove(history_file_path)

    return plaintext_to_html('History erased')

def do_generate(
        mode: str,
        prompt: str,
        cfg: float,
        denoise: float,
        sampler_index: int,
        sampler_steps: int,
        batch_count: int,
        batch_size: int,
        input_img,
        resize_mode,
        image_height: int,
        image_width: int,
        custom_code: str,
        input_seed: int,
        prompt_matrix: bool,
        loopback: bool,
        upscale: bool,
        inpainting_mask_blur,
        inpainting_mask_content,
        inpainting_image,
        use_input_seed: bool):

    if mode == 'Text-to-Image' or mode == 'Image-to-Image':
        if batch_count <= 0:
            raise Exception("Batch count must be > 0")
        elif batch_size <= 0:
            raise Exception("Images per batch must be > 0")
        elif image_height % 64 != 0 or image_width % 64 != 0:
            raise Exception("Image dimensions must both be multiples of 64")

    if mode == 'Text-to-Image':
        images, js, html = txt2img(
            prompt=prompt,
            negative_prompt=None,
            steps=sampler_steps,
            sampler_index=sampler_index,
            use_GFPGAN=False,
            prompt_matrix=prompt_matrix,
            n_iter=batch_count,
            batch_size=batch_size,
            cfg_scale=cfg,
            seed=input_seed if use_input_seed else -1,
            height=image_height,
            width=image_width,
            code=custom_code)

        if opts.enable_history:
            save_to_history(images, html)

    elif mode == 'Image-to-Image' or mode == 'Inpainting':
        # TODO(maybe): move SD upscale to post-procesing
        # otherwise we need some complex UI handling to make it clear what options are mutually exclusive
        if mode == 'Image-to-Image':
            if upscale:
                img2img_mode = Img2Img_Modes.UPSCALE
            elif loopback:
                img2img_mode = Img2Img_Modes.LOOPBACK
            else:
                img2img_mode = Img2Img_Modes.CLASSIC
        elif mode == 'Inpainting':
            img2img_mode = Img2Img_Modes.INPAINT

        images, js, html = img2img(
            prompt=prompt,
            init_img=input_img,
            init_img_with_mask=inpainting_image,
            steps=sampler_steps,
            sampler_index=sampler_index,
            mask_blur=inpainting_mask_blur,
            inpainting_fill=inpainting_mask_content,
            use_GFPGAN=False,
            prompt_matrix=prompt_matrix,
            mode=img2img_mode,
            n_iter=batch_count,
            batch_size=batch_size,
            cfg_scale=cfg,
            denoising_strength=denoise,
            seed=input_seed if use_input_seed else -1,
            height=image_height,
            width=image_width,
            resize_mode=resize_mode,
            # TODO: make these selectable in the UI (not settings) once we move SD upscale to post-processing
            upscaler_name=opts.sd_upscale_upscaler_index,
            upscale_overlap=opts.sd_upscale_overlap,
            # TODO: add UI element for this
            inpaint_full_res=False)
    else:
        raise Exception('Invalid mode selected')

    return images, html



# TODO: separate frontend (UI, handlers, wrappers, etc) and backend (ML, image processing, etc) code into separate files
'''
--- UI CODE ---
'''

webui_css = open('webui.css', 'r').read()

if os.path.isfile('userstyle.css'):
    userstyle_css = open('userstyle.css', 'r', encoding='utf-8').read()
else:
    userstyle_css = ''

# Some History CSS needs to be dynamic to support custom settings
history_css = f"""
#history_img {{
    max-width: {opts.history_thumbnail_res}px;
    max-height: {opts.history_thumbnail_res}px;
}}

#history_thumbnails {{
    min-width: {int(3 * opts.history_thumbnail_res + 40)}px;
}}

#history_thumb {{
    min-width: {opts.history_thumbnail_res}px;
    min-height: {opts.history_thumbnail_res}px;
}}
"""

# TODO: check to make sure that their aren't alignment/size issues with the inpainting mask relative to the image if the image is not square
with gr.Blocks(css=webui_css + userstyle_css + history_css, analytics_enabled=False, title='Stable Diffusion WebUI') as demo:
    with gr.Tabs(elem_id='tabs'):
        # Stable Diffusion tab
        with gr.TabItem('Stable Diffusion', id='sd_tab'):
            with gr.Row(elem_id='prompt_row'):
                sd_prompt = gr.Textbox(elem_id='prompt_input', placeholder='A corgi wearing a top hat as an oil painting.', lines=1, max_lines=1, show_label=False)

            with gr.Row(elem_id='body').style(equal_height=False):
                # Left Column
                with gr.Column():
                    with gr.Row():
                        sd_image_width = gr.Number(label="Width", elem_id='img_width', value=512, precision=0)
                        sd_image_height = gr.Number(label="Height", elem_id='img_height', value=512, precision=0)

                    with gr.Row():
                        sd_batch_count = gr.Number(label='Batch count', precision=0, value=1)
                        sd_batch_size = gr.Number(label='Images per batch', precision=0, value=1)

                    with gr.Group():
                        sd_inpainting_image = gr.Image(show_label=False, source="upload", interactive=True, type="pil", tool="sketch", visible=False, elem_id='sd_inpaint_img')
                        sd_input_image = gr.Image(show_label=False, source="upload", interactive=True, type="pil", visible=False, elem_id='sd_input_img')

                        sd_resize_mode = gr.Dropdown(label="Resize mode", choices=["Stretch", "Scale and crop", "Scale and fill"], type="index", value="Stretch", visible=False)

                        with gr.Row(elem_id='mask_options'):
                            sd_inpainting_mask_content = gr.Dropdown(label='Masked content', choices=['Fill', 'Original', 'Latent noise', 'Latent nothing'], value='Fill', type="index", visible=False, elem_id='sd_inpainting_mask_content')
                            sd_inpainting_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False, elem_id='sd_inpainting_mask_blur')

                # Center Column
                with gr.Column():
                    sd_output_image = gr.Gallery(show_label=False, elem_id='output_gallery').style(grid=3)
                    sd_output_html = gr.HTML()

                # Right Column
                with gr.Column():
                    with gr.Group():
                        sd_sampling_method = gr.Dropdown(label='Sampling method', choices=[x.name for x in samplers], value=samplers[0].name, type="index")
                        sd_sampling_steps = gr.Slider(label="Sampling steps", value=18, minimum=1, maximum=100, step=1)

                    with gr.Group():
                        sd_cfg = gr.Slider(label='Prompt similarity (CFG)', value=8.0, minimum=1.0, maximum=15.0, step=0.5)
                        sd_denoise = gr.Slider(label='Denoising strength (DNS)', value=0.75, minimum=0.0, maximum=1.0, step=0.01, visible=False)

                    with gr.Group():
                        sd_use_input_seed = gr.Checkbox(label='Custom seed')
                        sd_input_seed = gr.Number(show_label=False, value=-1, precision=0, visible=False)
                        sd_matrix = gr.Checkbox(label='Create prompt matrix', value=False)
                        sd_loopback = gr.Checkbox(label='Output loopback', value=False, visible=False)
                        sd_upscale = gr.Checkbox(label='Stable diffusion upscale', value=False, visible=False)

            with gr.Row(elem_id='row_buttons'):
                sd_mode = gr.Dropdown(show_label=False, value='Text-to-Image', choices=['Text-to-Image', 'Image-to-Image', 'Inpainting'], elem_id='sd_mode')
                sd_save_image = gr.Button('Save', elem_id='sd_save_image').style(full_width=False)
                sd_generate = gr.Button('Generate', variant='primary', elem_id='sd_generate').style(full_width=False)

            with gr.Row():
                sd_custom_code = gr.Textbox(label="Generate script (Python)", visible=cmd_opts.allow_code, lines=1, elem_id='sd_script')

        # Post-Processing tab
        with gr.TabItem('Post-Processing'):
            with gr.Row():
                pp_input_image = gr.Image(show_label=False, source='upload', interactive=True, type='pil', elem_id='pp_input_img')
                pp_output_image = gr.Image(show_label=False, interactive=False, type='pil', elem_id='pp_output_img')

            with gr.Row():
                with gr.Column():
                    pp_gfpgan_enable = gr.Checkbox(label='GFPGAN', value=False, interactive=have_gfpgan)
                    pp_gfpgan_strength = gr.Slider(label='Strength', value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=False)
                with gr.Column():
                    pp_esrgan_enable = gr.Checkbox(label='ESRGAN', value=False, interactive=have_realesrgan)
                    pp_esrgan_model = gr.Dropdown(label='Model', choices=[x.name for x in realesrgan_models], value=realesrgan_models[0].name, type='index', interactive=False)
                    pp_esrgan_scale = gr.Slider(label='Scale', minimum=1.0, step=0.1, maximum=4.0, interactive=False)
                # TODO: add style transfer from https://www.tensorflow.org/tutorials/generative/style_transfer
            
            with gr.Row(elem_id='pp_buttons_row'):
                pp_stats = gr.HTML(elem_id='pp_stats')
                pp_submit = gr.Button('Submit', variant='primary', elem_id='pp_submit').style(full_width=False)

        # Image Info tab
        with gr.TabItem('Image Info'):
            with gr.Column():
                info_input_image = gr.Image(show_label=False, source='upload', interactive=True, type='pil', elem_id='info_img')
                info_html = gr.HTML()

        # History tab
        with gr.TabItem('History'):
            with gr.Row():
                hist_refresh = gr.Button('Refresh', elem_id='hist_refresh')
                hist_erase = gr.Button('Erase', elem_id='hist_erase')
            with gr.Row():
                hist_html = gr.HTML(value=read_history)

        # Settings tab
        with gr.TabItem('Settings'):
            sd_settings = [create_setting_component(key) for key in opts.data_labels.keys()]
            with gr.Row():
                sd_confirm_settings = gr.HTML(elem_id='sd_settings_html')
                sd_save_settings = gr.Button('Save', variant='primary', elem_id='sd_save_settings').style(full_width=False)


    # Tab - Stable Diffusion - Callbacks
    def mode_change(mode: str):
        is_img2img = (mode == 'Image-to-Image')
        is_txt2img = (mode == 'Text-to-Image')
        is_inpainting = (mode == 'Inpainting')

        return {
            sd_cfg: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_denoise: gr.update(visible=is_img2img or is_inpainting),
            # TODO: need to hide PLMS from smaplers if in an img2img mode
            sd_sampling_method: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_sampling_steps: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_batch_count: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_batch_size: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_resize_mode: gr.update(visible=is_img2img or is_inpainting),
            sd_image_height: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_image_width: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_custom_code: gr.update(visible=is_txt2img and cmd_opts.allow_code),
            sd_matrix: gr.update(visible=is_img2img or is_txt2img),
            sd_loopback: gr.update(visible=is_img2img),
            sd_upscale: gr.update(visible=is_img2img),
            sd_inpainting_mask_blur: gr.update(visible=is_inpainting),
            sd_inpainting_mask_content: gr.update(visible=is_inpainting),
            sd_input_image: gr.update(visible=is_img2img),
            sd_inpainting_image: gr.update(visible=is_inpainting),
        }

    sd_mode.change(
        fn=mode_change,
        inputs=[
            sd_mode,
        ],
        outputs=[
            sd_cfg,
            sd_denoise,
            sd_sampling_method,
            sd_sampling_steps,
            sd_batch_count,
            sd_batch_size,
            sd_input_image,
            sd_resize_mode,
            sd_image_height,
            sd_image_width,
            sd_custom_code,
            sd_matrix,
            sd_loopback,
            sd_upscale,
            sd_inpainting_mask_blur,
            sd_inpainting_mask_content,
            sd_inpainting_image
        ]
    )

    do_generate_args = dict(
        fn=wrap_gradio_gpu_call(do_generate),
        inputs=[
            sd_mode,
            sd_prompt,
            sd_cfg,
            sd_denoise,
            sd_sampling_method,
            sd_sampling_steps,
            sd_batch_count,
            sd_batch_size,
            sd_input_image,
            sd_resize_mode,
            sd_image_height,
            sd_image_width,
            sd_custom_code,
            sd_input_seed,
            sd_matrix,
            sd_loopback,
            sd_upscale,
            sd_inpainting_mask_blur,
            sd_inpainting_mask_content,
            sd_inpainting_image,
            sd_use_input_seed
        ],
        outputs=[
            sd_output_image,
            sd_output_html
        ]
    )

    sd_prompt.submit(**do_generate_args)
    sd_generate.click(**do_generate_args)

    sd_batch_count.submit(
        lambda value: value if value > 0 else 1,
        inputs=sd_batch_count,
        outputs=sd_batch_count
    )

    sd_batch_size.submit(
        lambda value: value if value > 0 else 1,
        inputs=sd_batch_size,
        outputs=sd_batch_size
    )

    sd_image_height.submit(
        lambda value : 64 * ((value + 63) // 64),
        inputs=sd_image_height,
        outputs=sd_image_height
    )

    sd_image_width.submit(
        lambda value : 64 * ((value + 63) // 64),
        inputs=sd_image_width,
        outputs=sd_image_width
    )

    sd_use_input_seed.change(
        lambda checked: gr.update(visible=checked),
        inputs=sd_use_input_seed,
        outputs=sd_input_seed
    )

    # Tab - Post-Processing - Callbacks
    pp_submit.click(
        wrap_gradio_gpu_call(run_postprocessing),
        inputs=[
            pp_input_image,
            pp_gfpgan_enable,
            pp_gfpgan_strength,
            pp_esrgan_enable,
            pp_esrgan_model,
            pp_esrgan_scale
        ],
        outputs=[
            pp_output_image,
            pp_stats
        ]
    )

    pp_esrgan_enable.change(
        lambda checked : {pp_esrgan_model: gr.update(interactive=checked), pp_esrgan_scale: gr.update(interactive=checked)},
        inputs=pp_esrgan_enable,
        outputs=[pp_esrgan_model, pp_esrgan_scale]
    )

    pp_gfpgan_enable.change(
        lambda checked : gr.update(interactive=checked),
        inputs=pp_gfpgan_enable,
        outputs=pp_gfpgan_strength
    )

    # Tab - History - Callbacks
    hist_refresh.click(
        read_history,
        inputs=None,
        outputs=hist_html
    )

    hist_erase.click(
        erase_history,
        inputs=None,
        outputs=hist_html
    )

    # Tab - Image Info - Callbacks
    info_input_image.change(
        run_image_info,
        inputs=info_input_image,
        outputs=info_html
    )

    # Tab - Settings - Callbacks
    sd_save_settings.click(
        fn=run_settings,
        inputs=sd_settings,
        outputs=sd_confirm_settings
    )

# Start the WebUI
demo.launch(share=cmd_opts.share)

'''
# TODO: use this eventually
interrupt.click(
    fn=lambda: state.interrupt(),
    inputs=[],
    outputs=[],
)

save.click(
    fn=wrap_gradio_call(save_files),
    inputs=[
        generation_info,
        gallery,
    ],
    outputs=[
        html_info,
        html_info,
        html_info,
    ]
)
'''