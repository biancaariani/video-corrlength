"""
extract_video_snippets.py
─────────────────────────
Extract the last N seconds from every .mp4 file in a folder.

Output directory behaviour
──────────────────────────
If --output_folder is omitted the script automatically creates a subfolder
called  last{N}sec/  inside the input folder and uses that as the output dir.
If the target directory does not exist it is created on the fly.

CLI examples
────────────
# Auto output dir  →  video_data/last20sec/
python extract_video_snippets.py video_data

# Explicit output dir
python extract_video_snippets.py video_data video_data/my_clips --seconds 30
"""

import os
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def extract_last_seconds(input_folder: str,
                         output_folder: str | None = None,
                         n: int = 20) -> str:
    """
    Extract the last *n* seconds from every .mp4 file in *input_folder* and
    write the clips to *output_folder*.

    Parameters
    ----------
    input_folder : str
        Directory that contains the source .mp4 files.
    output_folder : str or None
        Destination directory.  If *None* (default), a subfolder named
        ``last{n}sec`` is created inside *input_folder* automatically.
    n : int
        Number of seconds to keep from the end of each clip (default 20).

    Returns
    -------
    str
        The resolved path of the output folder that was used.
    """
    # ── resolve output directory ──────────────────────────────────────────────
    if output_folder is None:
        output_folder = os.path.join(input_folder, f"last{n}sec")

    if not os.path.exists(output_folder):
        print(f"[INFO] Creating output directory: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
    else:
        print(f"[INFO] Output directory already exists: {output_folder}")

    # ── collect source files ──────────────────────────────────────────────────
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".mp4")
    ]

    if not video_files:
        print(f"[WARNING] No .mp4 files found in: {input_folder}")
        return output_folder

    # ── process each clip ─────────────────────────────────────────────────────
    for filename in sorted(video_files):
        input_path = os.path.join(input_folder, filename)

        try:
            with VideoFileClip(input_path) as clip:
                duration = clip.duration

                if duration < n:
                    print(
                        f"[SKIP] {filename}: duration {duration:.2f}s "
                        f"< requested {n}s"
                    )
                    continue

                start_time = duration - n

                # Compatible with MoviePy 1.x (.subclip) and 2.x (.subclipped)
                if hasattr(clip, "subclipped"):
                    subclip = clip.subclipped(start_time, duration)
                else:
                    subclip = clip.subclip(start_time, duration)

                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_last{n}sec{ext}"
                output_path = os.path.join(output_folder, output_filename)

                print(f"[WRITE] {output_path}")
                subclip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None          # suppress MoviePy progress bars
                )

        except Exception as exc:
            print(f"[ERROR] Failed to process {filename}: {exc}")

    return output_folder


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the last N seconds from every .mp4 in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing source .mp4 files.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",            # optional — auto-derived when omitted
        default=None,
        help=(
            "Destination folder for the extracted clips.  "
            "Defaults to <input_folder>/last{N}sec/ (created if absent)."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=20,
        help="Number of seconds to extract from the end of each clip.",
    )

    args = parser.parse_args()
    used_folder = extract_last_seconds(
        args.input_folder,
        args.output_folder,
        args.seconds,
    )
    print(f"\n[DONE] Clips saved to: {used_folder}")
