# nodes.py
from __future__ import annotations

import base64
import io
import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


SF_CONFIG = Dict[str, str]


def _first_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI IMAGE tensor to a PIL Image.
    Expected shape: [B, H, W, C] with values in [0, 1].
    Uses the first image in the batch.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image_tensor).__name__}")

    if image_tensor.ndim != 4 or image_tensor.shape[0] < 1:
        raise ValueError(f"Unexpected IMAGE tensor shape: {tuple(image_tensor.shape)}")

    arr = image_tensor[0].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale to limit the longer side to max_side (no upscale)."""
    if max_side <= 0:
        return img

    w, h = img.size
    longer = max(w, h)
    if longer <= max_side:
        return img

    scale = max_side / float(longer)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _encode_data_url(img: Image.Image, image_format: str, jpeg_quality: int) -> str:
    """
    Encode PIL Image as a data URL for the OpenAI-compatible image_url field.
    image_format: "JPEG" or "PNG"
    """
    fmt = (image_format or "JPEG").upper().strip()
    buffered = io.BytesIO()

    if fmt == "PNG":
        img.save(buffered, format="PNG", optimize=True)
        mime = "image/png"
    else:
        q = int(jpeg_quality)
        q = max(1, min(100, q))
        img_rgb = img.convert("RGB")
        img_rgb.save(buffered, format="JPEG", quality=q, optimize=True)
        mime = "image/jpeg"

    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class SiliconFlowLoader_AS:
    """Stores API settings and returns a config object."""

    RETURN_TYPES = ("SF_CONFIG",)
    RETURN_NAMES = ("sf_config",)
    FUNCTION = "load_config"
    CATEGORY = "SiliconFlow/AS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "Qwen/Qwen2-VL-72B-Instruct", "multiline": False}),
                "base_url": ("STRING", {"default": "https://api.siliconflow.cn/v1", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def load_config(self, model: str, base_url: str, api_key: str) -> Tuple[SF_CONFIG]:
        cfg: SF_CONFIG = {
            "model": (model or "").strip(),
            "base_url": (base_url or "").strip().rstrip("/"),
            "api_key": (api_key or "").strip(),
        }
        return (cfg,)


class SiliconFlowSampler_AS:
    """Sends an OpenAI-compatible chat/completions request (text or vision) and returns the response text."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "SiliconFlow/AS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sf_config": ("SF_CONFIG",),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "timeout_seconds": ("INT", {"default": 60, "min": 1, "max": 600}),
                "image_format": (["JPEG", "PNG"], {"default": "JPEG"}),
                "max_image_side": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "jpeg_quality": ("INT", {"default": 80, "min": 1, "max": 100}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str, image_data_url: Optional[str]) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        if image_data_url:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _extract_content(result_json: Dict[str, Any]) -> str:
        choices = result_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Unexpected response format: {result_json}")

        msg = choices[0].get("message", {})
        content = msg.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        if content is None:
            raise ValueError(f"Unexpected response format: {result_json}")

        return json.dumps(content, ensure_ascii=False, indent=2)

    def generate(
        self,
        sf_config: SF_CONFIG,
        system_prompt: str,
        user_prompt: str,
        seed: int,
        max_tokens: int,
        temperature: float,
        timeout_seconds: int,
        image_format: str,
        max_image_side: int,
        jpeg_quality: int,
        image: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        api_key = (sf_config or {}).get("api_key", "").strip()
        base_url = (sf_config or {}).get("base_url", "").strip().rstrip("/")
        model = (sf_config or {}).get("model", "").strip()

        if not api_key:
            return ("Error: API key is missing.",)
        if not base_url:
            return ("Error: base_url is missing.",)
        if not model:
            return ("Error: model is missing.",)

        image_data_url: Optional[str] = None
        if image is not None:
            try:
                pil_img = _first_image_to_pil(image)
                pil_img = _resize_max_side(pil_img, int(max_image_side))
                image_data_url = _encode_data_url(pil_img, image_format, int(jpeg_quality))
            except Exception as e:
                return (f"Error: Failed to encode image: {e}",)

        messages = self._build_messages(system_prompt, user_prompt, image_data_url)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        endpoint = f"{base_url}/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as response:
                response_text = response.read().decode("utf-8")
            result_json = json.loads(response_text)
            content = self._extract_content(result_json)
            return (content,)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return (f"API Error {e.code}: {body}",)
        except urllib.error.URLError as e:
            return (f"Network Error: {getattr(e, 'reason', e)}",)
        except socket.timeout:
            return (f"Timeout Error: request exceeded {int(timeout_seconds)} seconds.",)
        except json.JSONDecodeError as e:
            return (f"Error: Failed to parse JSON response: {e}",)
        except Exception as e:
            return (f"System Error: {e}",)
