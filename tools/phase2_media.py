from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _slugify(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    return slug[:64] or "asset"


def _safe_text(text: str) -> str:
    return (text or "").strip() or "Narration placeholder."


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_placeholders_enabled() -> bool:
    # Debug mode keeps legacy synthetic overlays for troubleshooting only.
    return _env_bool("PHASE2_DEBUG_PLACEHOLDERS", False)


def _strict_production_mode() -> bool:
    # Production mode fails fast when true media backends are unavailable.
    return _env_bool("PHASE2_STRICT_PRODUCTION", True)


def _burn_in_overlays() -> bool:
    return _env_bool("PHASE2_BURN_OVERLAYS", False)


def _scene_seed(scene_id: str, heading: str, index: int = 0) -> int:
    material = f"{scene_id}|{heading}|{index}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def _truncate_prompt(text: str, max_words: int = 60, max_chars: int = 380) -> str:
    words = text.strip().split()
    compact = " ".join(words[:max_words])
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip(" ,.;:-")
    return compact


def _temp_dir(prefix: str = "") -> Path:
    """Return a temp directory on D: (via TEMP_MEDIA_DIR) instead of C:\\Temp."""
    base = _env_str("TEMP_MEDIA_DIR", "")
    if base:
        p = Path(base)
    else:
        p = Path(__file__).resolve().parents[1] / "temp_media"
    p.mkdir(parents=True, exist_ok=True)
    if prefix:
        sub = p / prefix
        sub.mkdir(parents=True, exist_ok=True)
        return sub
    return p


def _expand_cmd(template: str, values: Dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _run_shell_cmd(template: str, values: Dict[str, Any], timeout_sec: int) -> Tuple[bool, str]:
    cmd = _expand_cmd(template, values)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_sec)
    if proc.returncode == 0:
        return True, cmd
    err = (proc.stderr or proc.stdout or "").strip()
    return False, f"{cmd}\n{err}"


def _fallback_tone_wav(path: Path, text: str, character_name: str, emotion: str) -> None:
    sample_rate = 16000
    duration_seconds = max(1.0, min(8.5, len(_safe_text(text)) * 0.055))
    total_samples = int(sample_rate * duration_seconds)

    seed = int(hashlib.md5(character_name.encode("utf-8")).hexdigest(), 16)
    base_freq = 170 + (seed % 170)
    shift = {
        "neutral": 0,
        "sad": -24,
        "happy": 16,
        "angry": 26,
        "calm": -10,
    }
    freq = max(80, base_freq + shift.get((emotion or "neutral").lower(), 0))

    amplitude = 12000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = bytearray()
        for idx in range(total_samples):
            t = idx / sample_rate
            envelope = 0.65 + 0.35 * math.sin(2.0 * math.pi * 1.9 * t)
            sample = int(amplitude * envelope * math.sin(2.0 * math.pi * freq * t))
            pcm.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(pcm))


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _convert_audio_to_wav(source_audio: Path, target_wav: Path) -> bool:
    target_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            _ffmpeg_exe(),
            "-y",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(target_wav),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return proc.returncode == 0 and target_wav.exists() and target_wav.stat().st_size > 1024
    except Exception:
        return False


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _edge_tts_to_wav(path: Path, text: str, character_name: str, emotion: str) -> bool:
    voice_pool = [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
    ]
    idx = int(hashlib.md5(character_name.encode("utf-8")).hexdigest(), 16) % len(voice_pool)
    voice = voice_pool[idx]

    emo = (emotion or "neutral").lower()
    rate = "+0%"
    pitch = "+0Hz"
    if emo in ("happy", "angry"):
        rate = "+8%"
        pitch = "+5Hz"
    elif emo in ("sad", "calm"):
        rate = "-8%"
        pitch = "-4Hz"

    async def _synth() -> Path:
        import edge_tts

        tmp_mp3 = path.with_suffix(".tmp.mp3")
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(str(tmp_mp3))
        return tmp_mp3

    try:
        mp3_path = _run_async(_synth())
        ok = _convert_audio_to_wav(mp3_path, path)
        try:
            if mp3_path.exists():
                mp3_path.unlink()
        except Exception:
            pass
        return ok
    except Exception:
        return False


def _is_valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        with wave.open(str(path), "rb"):
            return True
    except Exception:
        return False


def synthesize_voice_wav(path: Path, text: str, character_name: str, emotion: str = "neutral") -> Tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _safe_text(text)

    if _edge_tts_to_wav(path, line, character_name, emotion):
        return True, "edge-tts"

    try:
        import pyttsx3

        engine = pyttsx3.init()
        base_rate = 172
        if (emotion or "").lower() in ("happy", "angry"):
            base_rate = 184
        elif (emotion or "").lower() in ("sad", "calm"):
            base_rate = 156
        engine.setProperty("rate", base_rate)

        voices = engine.getProperty("voices")
        if voices:
            pick = int(hashlib.md5(character_name.encode("utf-8")).hexdigest(), 16) % len(voices)
            engine.setProperty("voice", voices[pick].id)

        engine.save_to_file(line, str(path))
        engine.runAndWait()
        if _is_valid_wav(path):
            return True, "pyttsx3"
    except Exception:
        pass

    _fallback_tone_wav(path, line, character_name, emotion)
    return True, "tone-fallback"


def _load_reference_image(reference_image: str, size: Tuple[int, int]):
    from PIL import Image, ImageDraw

    w, h = size
    ref = Path(reference_image) if reference_image else None
    if ref and ref.exists() and ref.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            img = Image.open(ref).convert("RGB")
            return img
        except Exception:
            pass

    # deterministic placeholder portrait if no valid image exists
    img = Image.new("RGB", (max(360, w // 2), max(360, h // 2)), (45, 52, 68))
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 40, 280, 240), fill=(210, 180, 160))
    draw.rectangle((120, 220, 240, 360), fill=(80, 105, 140))
    return img


def _scene_prompts(heading: str, style: str, reference_image: str) -> List[str]:
    ref_hint = ""
    if reference_image:
        ref_hint = f"reference={Path(reference_image).name}, "

    base = (
        "cinematic still, coherent setting, natural composition, no text, "
        "consistent character identity, stable facial traits"
    )
    heading_short = (heading or "Scene").strip()[:90]
    style_short = (style or "cinematic").strip()[:90]
    context = f"heading={heading_short}; style={style_short}; {ref_hint}".strip()
    shot_plan = [
        "establishing wide shot",
        "medium dialogue shot",
        "close reaction shot",
        "over-the-shoulder coverage",
        "insert detail shot",
        "tracking profile shot",
        "reverse angle dialogue shot",
        "final hold shot",
    ]
    keyframe_count = max(4, min(8, _env_int("PHASE2_STORYBOARD_KEYFRAMES", 6)))

    prompts: List[str] = []
    for i in range(keyframe_count):
        shot = shot_plan[i % len(shot_plan)]
        prompts.append(_truncate_prompt(f"{base}, {shot}, {context}"))
    return prompts


def _generate_scene_keyframes(
    scene_id: str,
    heading: str,
    style: str,
    reference_image: str,
    keyframe_dir: Path,
    width: int,
    height: int,
) -> Tuple[List[Path], str, List[str]]:
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    prompts = _scene_prompts(heading, style, reference_image)
    planned_keyframes = max(1, len(prompts))
    min_required = max(2, min(planned_keyframes, _env_int("PHASE2_MIN_KEYFRAMES", 3)))
    backend_errors: List[str] = []

    # Prefer ComfyUI for high-quality scene generation when configured.
    try:
        from tools.comfy_client import comfyui_configured, generate_character_image_via_comfyui

        if comfyui_configured():
            out_paths: List[Path] = []
            for i, prompt in enumerate(prompts, start=1):
                dest = keyframe_dir / f"{_slugify(scene_id)}_kf_{i:02d}.png"
                try:
                    produced = Path(
                        generate_character_image_via_comfyui(
                            prompt,
                            dest,
                            seed=_scene_seed(scene_id, heading, i),
                        )
                    )
                    if produced.exists():
                        out_paths.append(produced)
                except Exception as exc:
                    backend_errors.append(f"comfyui:kf_{i:02d}:{exc}")
            if len(out_paths) >= min_required:
                return out_paths, "comfyui", prompts
            backend_errors.append(
                f"comfyui:generated={len(out_paths)} planned={planned_keyframes} min_required={min_required}"
            )
    except Exception:
        pass

    # Next best: local SD diffusers backend.
    try:
        from tools.local_sd import generate_local_sd_image, local_sd_configured

        if local_sd_configured():
            old_w = os.getenv("SD_LOCAL_WIDTH")
            old_h = os.getenv("SD_LOCAL_HEIGHT")
            os.environ["SD_LOCAL_WIDTH"] = str(width)
            os.environ["SD_LOCAL_HEIGHT"] = str(height)
            try:
                out_paths = []
                for i, prompt in enumerate(prompts, start=1):
                    dest = keyframe_dir / f"{_slugify(scene_id)}_kf_{i:02d}.png"
                    try:
                        produced = Path(
                            generate_local_sd_image(
                                prompt,
                                dest,
                                seed=_scene_seed(scene_id, heading, i),
                            )
                        )
                        if produced.exists():
                            out_paths.append(produced)
                    except Exception as exc:
                        backend_errors.append(f"local-sd:kf_{i:02d}:{exc}")
                if len(out_paths) >= min_required:
                    return out_paths, "local-sd", prompts
                backend_errors.append(
                    f"local-sd:generated={len(out_paths)} planned={planned_keyframes} min_required={min_required}"
                )
            finally:
                if old_w is None:
                    os.environ.pop("SD_LOCAL_WIDTH", None)
                else:
                    os.environ["SD_LOCAL_WIDTH"] = old_w

                if old_h is None:
                    os.environ.pop("SD_LOCAL_HEIGHT", None)
                else:
                    os.environ["SD_LOCAL_HEIGHT"] = old_h
    except Exception:
        pass

    if _strict_production_mode() and not _debug_placeholders_enabled():
        detail = " | ".join(backend_errors[-6:]) if backend_errors else "no backend details"
        raise RuntimeError(
            "Scene generation failed: no production image backend produced keyframes. "
            f"Configure ComfyUI or Local SD. details={detail}"
        )

    return [], "fallback", prompts


def _collect_frames_from_dir(path: Path) -> List[Path]:
    out = sorted(path.glob("frame_*.png"))
    if out:
        return out
    out = sorted(path.glob("frame_*.jpg"))
    if out:
        return out
    return sorted(path.glob("frame_*.jpeg"))


def _run_motion_synthesis(
    scene_id: str,
    keyframe_dir: Path,
    frame_sequence_dir: Path,
    fps: int,
) -> Tuple[List[Path], str]:
    cmd = _env_str("MOTION_SYNTH_CMD", "")
    if not cmd:
        return [], ""

    out_dir = frame_sequence_dir / "_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, detail = _run_shell_cmd(
        cmd,
        {
            "scene_id": scene_id,
            "keyframe_dir": str(keyframe_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
            "fps": fps,
        },
        timeout_sec=max(60, _env_int("MOTION_SYNTH_TIMEOUT_SEC", 600)),
    )
    if not ok:
        raise RuntimeError(f"motion synthesis failed: {detail}")

    frames = _collect_frames_from_dir(out_dir)
    if not frames:
        raise RuntimeError("motion synthesis finished but produced no frames")
    return frames, "motion-synth"


def _run_temporal_interpolator(scene_id: str, frame_sequence_dir: Path) -> Tuple[List[Path], str]:
    rife_cmd = _env_str("RIFE_INTERP_CMD", "")
    film_cmd = _env_str("FILM_INTERP_CMD", "")
    cmd = rife_cmd or film_cmd
    backend = "rife" if rife_cmd else ("film" if film_cmd else "")
    if not cmd:
        if _strict_production_mode() and not _debug_placeholders_enabled():
            raise RuntimeError("No temporal interpolation backend configured (set RIFE_INTERP_CMD or FILM_INTERP_CMD)")
        return _collect_frames_from_dir(frame_sequence_dir), "none"

    out_dir = frame_sequence_dir / "_interpolated"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, detail = _run_shell_cmd(
        cmd,
        {
            "scene_id": scene_id,
            "input_dir": str(frame_sequence_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
        },
        timeout_sec=max(60, _env_int("INTERP_TIMEOUT_SEC", 600)),
    )
    if not ok:
        raise RuntimeError(f"temporal interpolation failed: {detail}")

    frames = _collect_frames_from_dir(out_dir)
    if not frames:
        raise RuntimeError("temporal interpolation finished but produced no frames")
    return frames, backend


def render_scene_frame_sequence(
    scene_id: str,
    heading: str,
    style: str,
    frame_sequence_dir: Path,
    frame_count: int,
    reference_image: str,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
) -> List[str]:
    from PIL import Image, ImageChops, ImageDraw

    frame_sequence_dir.mkdir(parents=True, exist_ok=True)
    frame_total = max(8, int(frame_count))
    draw_overlay = _burn_in_overlays() or _debug_placeholders_enabled()

    keyframe_dir = frame_sequence_dir / "_keyframes"
    keyframes, backend, prompts = _generate_scene_keyframes(
        scene_id=scene_id,
        heading=heading,
        style=style,
        reference_image=reference_image,
        keyframe_dir=keyframe_dir,
        width=width,
        height=height,
    )

    (frame_sequence_dir / "_scene_backend.txt").write_text(
        "backend=" + backend + "\n" + "\n".join(prompts) + "\n",
        encoding="utf-8",
    )

    if _strict_production_mode() and not _debug_placeholders_enabled() and backend == "fallback":
        raise RuntimeError(
            f"Scene {scene_id} rendered with fallback backend. "
            "Production mode requires local-sd or comfyui."
        )

    if keyframes:
        if not _debug_placeholders_enabled():
            try:
                motion_frames, motion_backend = _run_motion_synthesis(
                    scene_id=scene_id,
                    keyframe_dir=keyframe_dir,
                    frame_sequence_dir=frame_sequence_dir,
                    fps=fps,
                )
                (frame_sequence_dir / "_scene_backend.txt").write_text(
                    "backend=" + motion_backend + "\n" + "\n".join(prompts) + "\n",
                    encoding="utf-8",
                )
                return [str(p.resolve()) for p in motion_frames]
            except Exception:
                # Continue into local interpolation path if motion synthesis is unavailable.
                pass

        kf_images: List[Image.Image] = []
        for path in keyframes:
            img = Image.open(path).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.BICUBIC)
            kf_images.append(img)

        n_kf = len(kf_images)
        frame_paths: List[str] = []
        for idx in range(frame_total):
            t = idx / max(1, frame_total - 1)
            seg = t * max(1, n_kf - 1)
            lo = int(math.floor(seg))
            hi = min(n_kf - 1, lo + 1)
            alpha = max(0.0, min(1.0, seg - lo))

            if lo == hi:
                base = kf_images[lo].copy()
            else:
                base = Image.blend(kf_images[lo], kf_images[hi], alpha)

            # Add subtle camera motion so scenes do not feel static.
            motion = _apply_camera_motion(base.convert("RGBA"), idx, frame_total).convert("RGB")

            # Minor flicker/vignette for cinematic continuity.
            flicker = int(6 * math.sin(2.0 * math.pi * t))
            if flicker != 0:
                offset = Image.new("RGB", (width, height), (flicker, flicker, flicker))
                motion = ImageChops.add(motion, offset, scale=1.0, offset=0)

            if draw_overlay:
                draw = ImageDraw.Draw(motion)
                draw.rectangle((0, 0, width, 42), fill=(10, 10, 14))
                draw.text((14, 13), f"{scene_id.upper()}  |  {heading[:70]}", fill=(240, 240, 240))
                draw.rectangle((0, height - 28, width, height), fill=(16, 16, 22))
                draw.text((14, height - 21), style[:90], fill=(200, 205, 218))

            frame_path = frame_sequence_dir / f"frame_{idx + 1:04d}.png"
            motion.save(frame_path)
            frame_paths.append(frame_path)

        if not _debug_placeholders_enabled():
            interp_frames, interp_backend = _run_temporal_interpolator(scene_id, frame_sequence_dir)
            (frame_sequence_dir / "_scene_backend.txt").write_text(
                f"backend={backend}+{interp_backend}\n" + "\n".join(prompts) + "\n",
                encoding="utf-8",
            )
            return [str(p.resolve()) for p in interp_frames]

        return [str(p.resolve()) for p in frame_paths]

    from PIL import ImageFilter

    if _strict_production_mode() and not _debug_placeholders_enabled():
        raise RuntimeError(
            f"Scene {scene_id} has no generated keyframes; fallback rendering is blocked in production mode"
        )

    ref_img = _load_reference_image(reference_image, (width, height))
    if ref_img.width != ref_img.height:
        side = min(ref_img.width, ref_img.height)
        left = (ref_img.width - side) // 2
        top = (ref_img.height - side) // 2
        ref_img = ref_img.crop((left, top, left + side, top + side))

    bg = ref_img.resize((width, height), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(10))

    seed = int(hashlib.md5(f"{scene_id}:{heading}".encode("utf-8")).hexdigest(), 16)
    c1 = (40 + (seed % 70), 52 + ((seed >> 8) % 80), 88 + ((seed >> 16) % 90))
    c2 = (18 + ((seed >> 4) % 50), 22 + ((seed >> 12) % 60), 35 + ((seed >> 20) % 60))

    frame_paths: List[str] = []
    for idx in range(frame_total):
        t = idx / max(1, frame_total - 1)

        base = Image.new("RGB", (width, height), c2)
        grad = Image.new("RGB", (width, height), c1)
        mask = Image.new("L", (width, height))
        mask_draw = ImageDraw.Draw(mask)
        horizon = int(height * (0.25 + 0.08 * math.sin(2.0 * math.pi * t)))
        mask_draw.rectangle((0, 0, width, horizon), fill=180)
        base = Image.composite(grad, base, mask)

        # Blend in a cinematic blurred plate derived from reference image.
        base = Image.blend(base, bg, 0.42)

        scale = 0.64 + 0.08 * math.sin(2.0 * math.pi * t)
        face_size = int(min(width, height) * scale)
        portrait = ref_img.resize((face_size, face_size))

        x = int((width - face_size) * (0.40 + 0.18 * math.sin(2.0 * math.pi * t)))
        y = int((height - face_size) * (0.50 + 0.08 * math.cos(2.0 * math.pi * t)))
        base.paste(portrait, (x, y))

        if draw_overlay:
            draw = ImageDraw.Draw(base)
            draw.rectangle((0, 0, width, 44), fill=(10, 10, 14))
            draw.text((14, 14), f"{scene_id.upper()}  |  {heading[:70]}", fill=(240, 240, 240))
            draw.rectangle((0, height - 30, width, height), fill=(16, 16, 22))
            draw.text((14, height - 22), style[:90], fill=(200, 205, 218))

        frame_path = frame_sequence_dir / f"frame_{idx + 1:04d}.png"
        base.save(frame_path)
        frame_paths.append(str(frame_path.resolve()))

    return frame_paths


def apply_face_swap_to_sequence(frame_sequence_dir: Path, reference_image: str, scene_id: str) -> Tuple[int, str]:
    from PIL import Image, ImageDraw

    frame_paths = sorted(frame_sequence_dir.glob("frame_*.png"))
    if not frame_paths:
        return 0, str(frame_sequence_dir.resolve())

    swapped_dir = frame_sequence_dir / "face_swapped"
    swapped_dir.mkdir(parents=True, exist_ok=True)

    if _debug_placeholders_enabled():
        ref = _load_reference_image(reference_image, (320, 320)).resize((180, 180)).convert("RGBA")
        mask = Image.new("L", (180, 180), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((8, 8, 172, 172), fill=180)
        count = 0
        for idx, frame_path in enumerate(frame_paths):
            frame = Image.open(frame_path).convert("RGBA")
            wobble_x = int(4 * math.sin(2.0 * math.pi * idx / max(1, len(frame_paths))))
            wobble_y = int(3 * math.cos(2.0 * math.pi * idx / max(1, len(frame_paths))))
            pos_x = (frame.width - ref.width) // 2 + wobble_x
            pos_y = int(frame.height * 0.28) + wobble_y
            frame.paste(ref, (pos_x, pos_y), mask)
            out = swapped_dir / frame_path.name
            frame.convert("RGB").save(out)
            count += 1
        return count, str(swapped_dir.resolve())

    try:
        import cv2
    except Exception as exc:
        if _strict_production_mode():
            raise RuntimeError("OpenCV is required for production face swapping") from exc
        for frame_path in frame_paths:
            shutil.copyfile(frame_path, swapped_dir / frame_path.name)
        return 0, str(swapped_dir.resolve())

    if reference_image and Path(reference_image).exists():
        ref_bgr = cv2.imread(str(reference_image))
    else:
        ref_pil = _load_reference_image(reference_image, (512, 512)).resize((512, 512)).convert("RGB")
        ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)

    if ref_bgr is None:
        raise RuntimeError("Reference image could not be loaded for face swap")

    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be initialized")

    def _prep_gray(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            gray = cv2.equalizeHist(gray)
        except Exception:
            pass
        return gray

    def _detect(gray):
        # Try progressively more permissive settings to handle small/partial faces.
        for scale in (1.05, 1.10, 1.15):
            for neighbors in (3, 5):
                for ms in ((24, 24), (32, 32), (40, 40)):
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=neighbors,
                        minSize=ms,
                    )
                    if len(faces) > 0:
                        return faces
        return []

    ref_gray = _prep_gray(ref_bgr)
    ref_faces = _detect(ref_gray)
    if len(ref_faces) == 0:
        if _strict_production_mode():
            raise RuntimeError("No face detected in reference image for production face swap")
        for frame_path in frame_paths:
            shutil.copyfile(frame_path, swapped_dir / frame_path.name)
        return 0, str(swapped_dir.resolve())

    rx, ry, rw, rh = max(ref_faces, key=lambda f: int(f[2]) * int(f[3]))
    ref_crop = ref_bgr[ry : ry + rh, rx : rx + rw]

    mapped = 0
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        gray = _prep_gray(frame_bgr)
        faces = _detect(gray)
        if len(faces) == 0:
            out = swapped_dir / frame_path.name
            cv2.imwrite(str(out), frame_bgr)
            continue

        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        target = cv2.resize(ref_crop, (w, h), interpolation=cv2.INTER_CUBIC)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (max(8, w // 2 - 4), max(8, h // 2 - 4)), 0, 0, 360, 255, -1)

        center = (x + w // 2, y + h // 2)
        try:
            blended = cv2.seamlessClone(target, frame_bgr, mask, center, cv2.NORMAL_CLONE)
            frame_bgr = blended
            mapped += 1
        except Exception:
            pass

        out = swapped_dir / frame_path.name
        cv2.imwrite(str(out), frame_bgr)

    if mapped == 0 and _strict_production_mode():
        raise RuntimeError(f"No frame-level faces detected for scene {scene_id}")

    return mapped, str(swapped_dir.resolve())


def _find_frames(frame_sequence_dir: Path) -> List[Path]:
    png = sorted(frame_sequence_dir.glob("frame_*.png"))
    if png:
        return png
    jpg = sorted(frame_sequence_dir.glob("frame_*.jpg"))
    if jpg:
        return jpg
    jpeg = sorted(frame_sequence_dir.glob("frame_*.jpeg"))
    return jpeg


def _read_wav_envelope(audio_path: Path, frame_count: int) -> List[float]:
    if frame_count <= 0:
        return []

    wav_path = audio_path
    if wav_path.suffix.lower() != ".wav":
        tmp_wav = wav_path.with_suffix(".tmp_lipsync.wav")
        if _convert_audio_to_wav(wav_path, tmp_wav):
            wav_path = tmp_wav

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            if sampwidth != 2 or n_frames <= 0:
                return [0.35] * frame_count

            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            if audio.size == 0:
                return [0.35] * frame_count

            samples_per_chunk = max(1, int(audio.size / frame_count))
            env: List[float] = []
            for i in range(frame_count):
                s = i * samples_per_chunk
                e = min(audio.size, s + samples_per_chunk)
                if s >= e:
                    env.append(0.35)
                    continue
                chunk = audio[s:e]
                rms = float(np.sqrt(np.mean(np.square(chunk / 32768.0))))
                env.append(rms)

            m = max(env) if env else 1.0
            if m <= 1e-6:
                return [0.35] * frame_count

            normalized = [min(1.0, max(0.0, x / m)) for x in env]
            return [0.2 + 0.8 * x for x in normalized]
    except Exception:
        return [0.35] * frame_count
    finally:
        if wav_path.suffix == ".wav" and wav_path.name.endswith(".tmp_lipsync.wav"):
            try:
                wav_path.unlink()
            except Exception:
                pass


def _audio_duration_seconds(audio_path: Path) -> float:
    if not audio_path.exists():
        return 0.0

    wav_path = audio_path
    if wav_path.suffix.lower() != ".wav":
        tmp_wav = wav_path.with_suffix(".tmp_duration.wav")
        if _convert_audio_to_wav(wav_path, tmp_wav):
            wav_path = tmp_wav

    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            if rate <= 0 or n_frames <= 0:
                return 0.0
            return float(n_frames) / float(rate)
    except Exception:
        return 0.0
    finally:
        if wav_path.suffix == ".wav" and wav_path.name.endswith(".tmp_duration.wav"):
            try:
                wav_path.unlink()
            except Exception:
                pass


def _apply_camera_motion(frame, idx: int, total: int):
    from PIL import Image

    if total <= 1:
        return frame

    t = idx / float(total - 1)
    zoom = 1.0 + 0.07 * math.sin(2.0 * math.pi * t)
    zoom = max(1.0, zoom)
    new_w = int(frame.width * zoom)
    new_h = int(frame.height * zoom)
    scaled = frame.resize((new_w, new_h), Image.Resampling.BICUBIC)

    pan_x = int((new_w - frame.width) * (0.48 + 0.24 * math.sin(2.0 * math.pi * t)))
    pan_y = int((new_h - frame.height) * (0.42 + 0.18 * math.cos(2.0 * math.pi * t)))
    pan_x = min(max(0, pan_x), max(0, new_w - frame.width))
    pan_y = min(max(0, pan_y), max(0, new_h - frame.height))

    return scaled.crop((pan_x, pan_y, pan_x + frame.width, pan_y + frame.height))


def _run_external_lipsync(scene_id: str, audio_path: Path, frame_sequence_dir: Path, output_path: Path, fps: int) -> Tuple[bool, str]:
    backends = [
        ("wav2lip", _env_str("WAV2LIP_INFER_CMD", "")),
        ("sadtalker", _env_str("SADTALKER_INFER_CMD", "")),
    ]

    for backend_name, template in backends:
        if not template:
            continue
        ok, detail = _run_shell_cmd(
            template,
            {
                "scene_id": scene_id,
                "audio": str(audio_path.resolve()),
                "frames_dir": str(frame_sequence_dir.resolve()),
                "output": str(output_path.resolve()),
                "fps": fps,
            },
            timeout_sec=max(60, _env_int("LIPSYNC_TIMEOUT_SEC", 1200)),
        )
        if ok and output_path.exists() and output_path.stat().st_size > 2048:
            return True, backend_name
        if not ok:
            return False, f"{backend_name} failed: {detail}"

    if _strict_production_mode() and not _debug_placeholders_enabled():
        return False, "No Wav2Lip/SadTalker command configured (set WAV2LIP_INFER_CMD or SADTALKER_INFER_CMD)"
    return False, "no-external-lipsync-configured"


def motion_score_for_frames(frame_sequence_dir: Path) -> float:
    frame_paths = _find_frames(frame_sequence_dir)
    if len(frame_paths) < 2:
        return 0.0

    try:
        import cv2
    except Exception:
        return 0.0

    diffs: List[float] = []
    prev = None
    for path in frame_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev is not None and prev.shape == img.shape:
            diff = cv2.absdiff(prev, img)
            diffs.append(float(diff.mean()) / 255.0)
        prev = img

    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def sync_confidence_for_scene(audio_path: Path, frame_sequence_dir: Path, fps: int = 24) -> float:
    if not audio_path.exists():
        return 0.0
    duration = _audio_duration_seconds(audio_path)
    if duration <= 0.0:
        return 0.0

    frame_count = len(_find_frames(frame_sequence_dir))
    if frame_count <= 0:
        return 0.0

    visual_duration = frame_count / float(max(1, fps))
    drift = abs(duration - visual_duration)
    return max(0.0, 1.0 - (drift / max(duration, 1e-6)))


def write_scene_metrics(scene_id: str, metrics: Dict[str, Any]) -> str:
    qa_dir = Path(__file__).resolve().parents[1] / "outputs" / "qa_metrics"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{_slugify(scene_id)}_quality.json"
    payload = {
        "scene_id": scene_id,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path.resolve())


def _render_lipsync_frames(frame_paths: List[Path], audio_path: Path, out_dir: Path, target_count: int) -> List[Path]:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    total = max(1, int(target_count))
    envelope = _read_wav_envelope(audio_path, total)
    if not envelope:
        envelope = [0.35] * total

    rendered: List[Path] = []
    for idx in range(total):
        source_path = frame_paths[idx % len(frame_paths)]
        frame = Image.open(source_path).convert("RGBA")
        frame = _apply_camera_motion(frame, idx, total)
        draw = ImageDraw.Draw(frame, "RGBA")

        amp = envelope[idx] if idx < len(envelope) else 0.35
        w, h = frame.width, frame.height
        mouth_w = int(w * (0.06 + 0.04 * amp))
        mouth_h = int(h * (0.010 + 0.030 * amp))
        cx = int(w * 0.52)
        cy = int(h * 0.64)

        # Subtle lower-face shadow + animated mouth opening.
        draw.ellipse(
            (cx - mouth_w - 6, cy - mouth_h - 4, cx + mouth_w + 6, cy + mouth_h + 8),
            fill=(20, 8, 8, 72),
        )
        draw.ellipse(
            (cx - mouth_w, cy - mouth_h, cx + mouth_w, cy + mouth_h),
            fill=(120, 18, 28, 188),
        )
        draw.ellipse(
            (cx - mouth_w + 4, cy - max(2, mouth_h // 2), cx + mouth_w - 4, cy + max(2, mouth_h // 2)),
            fill=(34, 6, 8, 210),
        )

        out_path = out_dir / f"frame_{idx + 1:04d}.png"
        frame.convert("RGB").save(out_path)
        rendered.append(out_path)

    return rendered


def compose_scene_video(scene_id: str, audio_path: Path, frame_sequence_dir: Path, output_path: Path, fps: int = 24) -> Tuple[bool, str]:
    frame_paths = _find_frames(frame_sequence_dir)
    if not frame_paths:
        return False, "no frames found"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as iio
    except Exception as exc:
        return False, f"imageio import failed: {exc}"

    render_frames = frame_paths
    lipsync_backend = "no-lipsync"
    target_count = len(frame_paths)
    if audio_path.exists():
        duration_sec = _audio_duration_seconds(audio_path)
        if duration_sec > 0.0:
            target_count = max(len(frame_paths), int(math.ceil(duration_sec * max(6, fps))))
            target_count = min(1800, target_count)

    if audio_path.exists() and not _debug_placeholders_enabled():
        ok, backend = _run_external_lipsync(
            scene_id=scene_id,
            audio_path=audio_path,
            frame_sequence_dir=frame_sequence_dir,
            output_path=output_path,
            fps=max(6, fps),
        )
        if ok:
            return True, backend
        if _strict_production_mode():
            return False, backend

    if audio_path.exists() and _debug_placeholders_enabled():
        lipsync_dir = _temp_dir(f"{_slugify(scene_id)}_lipsync")
        try:
            render_frames = _render_lipsync_frames(frame_paths, audio_path, lipsync_dir, target_count)
            lipsync_backend = "debug-audio-driven-lipsync"
        except Exception:
            render_frames = frame_paths

    temp_video = _temp_dir() / f"{_slugify(scene_id)}_silent.mp4"
    try:
        writer = iio.get_writer(str(temp_video), fps=max(6, fps), codec="libx264")
        for fp in render_frames:
            writer.append_data(iio.imread(fp))
        writer.close()
    except Exception as exc:
        try:
            writer.close()
        except Exception:
            pass
        return False, f"video encode failed: {exc}"

    if not audio_path.exists():
        shutil.copyfile(temp_video, output_path)
        return True, lipsync_backend + "+video-only"

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(audio_path),
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 2048:
            return True, lipsync_backend + "+ffmpeg-mux"
    except Exception:
        pass

    shutil.copyfile(temp_video, output_path)
    return True, lipsync_backend + "+video-only"
