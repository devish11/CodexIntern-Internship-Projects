"""
README: Speech to Image Pipeline using MonsterAPI

This script converts speech from an audio file to an image using MonsterAPI's Whisper (speech-to-text) and SDXL (text-to-image) models.

Usage:
    python speech_to_image_monster.py --audio path/to/audio.mp3 --out output.png [--api_key YOUR_API_KEY]

    - --audio: Path to input audio file (supports mp3, m4a, wav, flac, ogg; assumes MonsterAPI accepts these directly).
    - --out: Output image file path (default: output.png).
    - --api_key: MonsterAPI key (or set env var MONSTER_API_KEY).
    - --dry_run: Run dry-run validation (file existence check only, no API calls).

Requirements:
    - Python 3.7+
    - pip install monsterapi requests

MonsterAPI offers Whisper and Stable Diffusion endpoints. See docs: https://monsterapi.ai/docs (commented out for brevity).

Notes:
    - Audio is sent directly to MonsterAPI (no conversion; ensure format is supported).
    - Default models: whisper-large-v3 for transcription, sdxl for image generation.
    - Image generation params: 512x512, 20 steps, no seed.
    - API key is masked in logs.
"""

import os
import sys
import argparse
import requests

# Constants
MONSTER_API_BASE = "https://api.monsterapi.ai/v1/generate"
WHISPER_MODEL = "whisper-large-v3"
SDXL_MODEL = "sdxl"
DEFAULT_OUT = "output.png"

def get_api_key(args):
    """Retrieve API key from args or env var, with masking."""
    key = args.api_key or os.getenv("MONSTER_API_KEY")
    if not key:
        raise ValueError("MonsterAPI key not provided. Use --api_key or set MONSTER_API_KEY env var.")
    return key

def mask_key(key):
    """Mask API key for logging."""
    return key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "*" * len(key)

def validate_audio(input_path):
    """
    Validate audio file existence and basic readability.

    Args:
        input_path (str): Path to input audio file.

    Raises:
        Exception: If validation fails.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if not os.path.isfile(input_path):
        raise ValueError(f"Path is not a file: {input_path}")
    # Basic check: try to open as binary
    try:
        with open(input_path, "rb") as f:
            f.read(1)  # Just check if readable
    except Exception as e:
        raise Exception(f"Audio file not readable: {e}")
    print(f"Audio file validated: {input_path}")

def transcribe_audio(audio_path, api_key):
    """
    Transcribe audio using MonsterAPI Whisper.

    Args:
        audio_path (str): Path to audio file.
        api_key (str): MonsterAPI key.

    Returns:
        str: Transcript text.

    Raises:
        Exception: If transcription fails.
    """
    print(f"Transcribing audio with model {WHISPER_MODEL}...")
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model": WHISPER_MODEL}
        response = requests.post(MONSTER_API_BASE, headers=headers, files=files, data=data)
    if response.status_code != 200:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
    result = response.json()
    transcript = result.get("text", "").strip()
    if not transcript:
        raise Exception("No transcript generated.")
    print(f"Transcript: {transcript}")
    return transcript

def generate_image(prompt, api_key, output_path):
    """
    Generate image from prompt using MonsterAPI SDXL.

    Args:
        prompt (str): Text prompt.
        api_key (str): MonsterAPI key.
        output_path (str): Path to save image.

    Returns:
        str: Path to saved image.

    Raises:
        Exception: If generation fails.
    """
    print(f"Generating image with model {SDXL_MODEL}...")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": SDXL_MODEL,
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "steps": 20,
        "seed": None  # Optional, set to int for reproducibility
    }
    response = requests.post(MONSTER_API_BASE, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Image generation failed: {response.status_code} - {response.text}")
    result = response.json()
    image_url = result.get("output", [None])[0]
    if not image_url:
        raise Exception("No image URL in response.")
    # Download image
    img_response = requests.get(image_url)
    if img_response.status_code != 200:
        raise Exception("Failed to download image.")
    with open(output_path, "wb") as f:
        f.write(img_response.content)
    print(f"Image saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Speech to Image using MonsterAPI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--api_key", help="MonsterAPI key")
    parser.add_argument("--dry_run", action="store_true", help="Dry-run: validate audio file only")
    args = parser.parse_args()

    try:
        api_key = get_api_key(args)
        print(f"Using API key: {mask_key(api_key)}")

        if args.dry_run:
            print("Dry-run mode: Validating audio file...")
            validate_audio(args.audio)
            print("Dry-run successful.")
            return

        # Validate audio
        validate_audio(args.audio)

        # Transcribe
        transcript = transcribe_audio(args.audio, api_key)

        # Generate image
        image_path = generate_image(transcript, api_key, args.out)

        # Output summary
        print(f"Models used: Transcription={WHISPER_MODEL}, Image={SDXL_MODEL}")
        print(f"Transcript: {transcript}")
        print(f"Image saved: {image_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()