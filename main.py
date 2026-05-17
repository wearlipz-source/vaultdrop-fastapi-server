"""
VaultDrop yt-dlp FastAPI Server — Production Architecture v4.0
Supports 1000+ platforms: YouTube, Instagram, TikTok, Twitter/X, Facebook, Vimeo, and more.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import yt_dlp
import re
import os
import asyncio
import logging
import time
import subprocess
import tempfile
import shutil
import uuid
import threading

# ---------------------------------------------------------------------------
# Logging — structured, reduced noise
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vaultdrop")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="VaultDrop yt-dlp Server", version="4.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Concurrency semaphore — prevent CPU/memory spikes from too many yt-dlp calls
# ---------------------------------------------------------------------------
_YDL_SEMAPHORE = asyncio.Semaphore(3)  # max 3 concurrent yt-dlp executions

# ---------------------------------------------------------------------------
# CORS — open for Flutter mobile clients
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Platform whitelist — reject unknown/malicious domains
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS = re.compile(
    r"(youtube\.com|youtu\.be|tiktok\.com|vimeo\.com|instagram\.com|"
    r"twitter\.com|x\.com|facebook\.com|fb\.watch|twitch\.tv|"
    r"dailymotion\.com|soundcloud\.com|reddit\.com|bilibili\.com|"
    r"nicovideo\.jp|rumble\.com|odysee\.com|bitchute\.com|"
    r"streamable\.com|gfycat\.com|imgur\.com|v\.redd\.it|"
    r"clips\.twitch\.tv|vm\.tiktok\.com|m\.youtube\.com)",
    re.IGNORECASE,
)


def validate_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL — must start with http:// or https://")
    if len(url) > 2048:
        raise HTTPException(status_code=400, detail="URL too long")
    blocked = re.compile(r"(localhost|127\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)", re.IGNORECASE)
    if blocked.search(url):
        raise HTTPException(status_code=400, detail="Local/internal URLs are not allowed")
    if not ALLOWED_DOMAINS.search(url):
        raise HTTPException(
            status_code=400,
            detail="Unsupported platform. Supported: YouTube, TikTok, Vimeo, Instagram, Twitter/X, Facebook, Twitch, Dailymotion, SoundCloud, Reddit, Bilibili, and more.",
        )
    return url


# ---------------------------------------------------------------------------
# Shared yt-dlp options
# ---------------------------------------------------------------------------
COMMON_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "skip_download": True,
    "socket_timeout": 30,
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    },
}


def _best_thumbnail(info: dict) -> str | None:
    thumbnails = info.get("thumbnails", [])
    if thumbnails:
        return thumbnails[-1].get("url")
    return info.get("thumbnail")


def _detect_platform(info: dict, url: str) -> str:
    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    domain_map = {
        "youtube": "YouTube", "youtu": "YouTube",
        "instagram": "Instagram",
        "tiktok": "TikTok",
        "twitter": "Twitter/X", "x.com": "Twitter/X",
        "facebook": "Facebook",
        "vimeo": "Vimeo",
        "twitch": "Twitch",
        "dailymotion": "Dailymotion",
        "soundcloud": "SoundCloud",
        "reddit": "Reddit",
        "bilibili": "Bilibili",
    }
    for key, name in domain_map.items():
        if key in extractor or key in url.lower():
            return name
    return info.get("extractor_key") or "Unknown"


def _format_duration(seconds) -> str:
    if not seconds:
        return "0:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _human_filesize(size_bytes) -> str:
    if not size_bytes:
        return "Unknown"
    size_bytes = int(size_bytes)
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Temp file registry — track all temp dirs for cleanup on shutdown
# ---------------------------------------------------------------------------
_temp_dirs_lock = threading.Lock()
_active_temp_dirs: set[str] = set()


def _register_temp_dir(path: str):
    with _temp_dirs_lock:
        _active_temp_dirs.add(path)


def _cleanup_temp_dir(path: str):
    """Remove a temp directory and deregister it."""
    with _temp_dirs_lock:
        _active_temp_dirs.discard(path)
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"[CLEANUP] Removed temp dir: {path}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to remove temp dir {path}: {e}")


def _cleanup_stale_temp_dirs():
    """Clean up any temp dirs left over from previous runs (e.g. after crash)."""
    try:
        tmp_root = tempfile.gettempdir()
        for entry in os.scandir(tmp_root):
            if entry.is_dir() and entry.name.startswith("vaultdrop_"):
                age = time.time() - entry.stat().st_mtime
                if age > 3600:  # older than 1 hour
                    shutil.rmtree(entry.path, ignore_errors=True)
                    logger.info(f"[CLEANUP] Removed stale temp dir: {entry.path}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Stale cleanup error: {e}")


@app.on_event("startup")
async def startup_event():
    _cleanup_stale_temp_dirs()
    logger.info(f"[STARTUP] VaultDrop v4.0.0 started. ffmpeg={_ffmpeg_available()}")


def _merge_video_audio_server(
    video_url: str,
    audio_url: str,
    output_path: str,
    timeout_seconds: int = 600,
) -> tuple[bool, str]:
    """
    Use ffmpeg to merge separate video and audio streams into a single mp4.
    Returns (success: bool, error_message: str).

    Reliability guarantees:
    - Verifies output file exists AND has non-zero size
    - Kills zombie ffmpeg processes on timeout
    - Returns detailed error on failure
    """
    if not _ffmpeg_available():
        return False, "ffmpeg not available on server"

    proc = None
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_url,
            "-i", audio_url,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-f", "mp4",
            output_path,
        ]
        logger.info(f"[MERGE] Starting ffmpeg merge → {output_path}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Kill zombie ffmpeg process
            proc.kill()
            proc.communicate()  # drain pipes
            logger.error(f"[MERGE] ffmpeg timed out after {timeout_seconds}s — process killed")
            return False, f"ffmpeg timed out after {timeout_seconds}s"

        if proc.returncode != 0:
            err_tail = stderr.decode("utf-8", errors="replace")[-500:] if stderr else "unknown"
            logger.error(f"[MERGE] ffmpeg failed (rc={proc.returncode}): {err_tail}")
            return False, f"ffmpeg exit code {proc.returncode}"

        # Verify output file exists and has content
        if not os.path.exists(output_path):
            logger.error("[MERGE] ffmpeg succeeded but output file missing")
            return False, "output file not created"

        size = os.path.getsize(output_path)
        if size == 0:
            logger.error("[MERGE] ffmpeg succeeded but output file is empty")
            os.remove(output_path)
            return False, "output file is empty"

        logger.info(f"[MERGE] ffmpeg merge successful: {output_path} ({size:,} bytes)")
        return True, ""

    except FileNotFoundError:
        return False, "ffmpeg binary not found"
    except Exception as e:
        logger.error(f"[MERGE] Unexpected ffmpeg error: {e}")
        if proc and proc.poll() is None:
            try:
                proc.kill()
                proc.communicate()
            except Exception:
                pass
        return False, str(e)


def _extract_best_url(info: dict, formats: list) -> tuple[str | None, str, int | None, str | None, str | None]:
    """
    Extract the best direct downloadable URL from yt-dlp info.
    Returns (url, ext, filesize, quality_label, audio_url).
    audio_url is non-None when video and audio are separate streams.
    """
    ext = info.get("ext", "mp4")
    filesize = None
    quality_label = None
    download_url = None
    audio_url = None

    # 1. requested_formats — yt-dlp selected specific formats
    requested_formats = info.get("requested_formats")
    if requested_formats:
        # Try progressive (video+audio in one stream)
        for rf in requested_formats:
            if (rf.get("url")
                    and rf.get("vcodec") not in (None, "none")
                    and rf.get("acodec") not in (None, "none")):
                download_url = rf["url"]
                ext = rf.get("ext", "mp4")
                filesize = rf.get("filesize") or rf.get("filesize_approx")
                quality_label = rf.get("format_note") or (
                    f"{rf['height']}p" if rf.get("height") else None
                )
                logger.info("[DOWNLOAD] Using progressive requested_format (video+audio)")
                break

        # Separate video + audio streams
        if not download_url:
            video_rf = next(
                (rf for rf in requested_formats
                 if rf.get("url") and rf.get("vcodec") not in (None, "none")),
                None
            )
            audio_rf = next(
                (rf for rf in requested_formats
                 if rf.get("url") and rf.get("acodec") not in (None, "none")
                 and rf.get("vcodec") in (None, "none")),
                None
            )

            if video_rf:
                download_url = video_rf["url"]
                ext = video_rf.get("ext", "mp4")
                filesize = (video_rf.get("filesize") or video_rf.get("filesize_approx") or 0)
                quality_label = video_rf.get("format_note") or (
                    f"{video_rf['height']}p" if video_rf.get("height") else None
                )
                if audio_rf:
                    audio_url = audio_rf["url"]
                    audio_filesize = audio_rf.get("filesize") or audio_rf.get("filesize_approx") or 0
                    filesize = (filesize or 0) + audio_filesize
                    logger.info("[DOWNLOAD] Separate video+audio streams detected")
                else:
                    logger.warning("[DOWNLOAD] Video-only stream, no audio found")
            else:
                for rf in requested_formats:
                    if rf.get("url"):
                        download_url = rf["url"]
                        ext = rf.get("ext", "mp4")
                        filesize = rf.get("filesize")
                        break

    # 2. Top-level info["url"]
    if not download_url and info.get("url"):
        download_url = info["url"]
        ext = info.get("ext", "mp4")
        filesize = info.get("filesize") or info.get("filesize_approx")
        logger.info("[DOWNLOAD] Using info['url']")

    # 3. Search formats list
    if not download_url:
        for f in reversed(formats):
            if (f.get("url")
                    and f.get("vcodec") not in (None, "none")
                    and f.get("acodec") not in (None, "none")):
                download_url = f["url"]
                ext = f.get("ext", "mp4")
                filesize = f.get("filesize") or f.get("filesize_approx")
                quality_label = f.get("format_note") or (
                    f"{f['height']}p" if f.get("height") else None
                )
                break

        if not download_url:
            for f in reversed(formats):
                if f.get("url"):
                    download_url = f["url"]
                    ext = f.get("ext", "mp4")
                    filesize = f.get("filesize") or f.get("filesize_approx")
                    break

    return download_url, ext or "mp4", int(filesize) if filesize else None, quality_label, audio_url


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()


class QualityOption(BaseModel):
    label: str
    format_id: str
    filesize: int | None = None
    filesize_human: str | None = None
    ext: str = "mp4"
    resolution: str | None = None
    fps: int | None = None
    vcodec: str | None = None
    acodec: str | None = None
    tbr: float | None = None


class AnalyzeResponse(BaseModel):
    title: str
    thumbnail: str | None = None
    duration: int | None = None
    duration_label: str | None = None
    uploader: str | None = None
    platform: str
    original_url: str
    qualities: list[QualityOption]


class DownloadRequest(BaseModel):
    url: str
    format_id: str

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()


class DownloadResponse(BaseModel):
    downloadUrl: str
    title: str
    ext: str
    filesize: int | None = None
    thumbnail: str | None = None
    duration: int | None = None
    uploader: str | None = None
    qualityLabel: str | None = None
    audioUrl: str | None = None
    merged: bool = False


class ResolveRequest(BaseModel):
    url: str
    quality: str = "1080p"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
def _ydl_error_to_http(e: yt_dlp.utils.DownloadError) -> HTTPException:
    msg = str(e).lower()
    if "private" in msg:
        return HTTPException(422, "This video is private and cannot be downloaded.")
    if "age" in msg:
        return HTTPException(422, "Age-restricted video — cannot be downloaded without login.")
    if "not available" in msg or "unavailable" in msg:
        return HTTPException(422, "This video is not available in your region.")
    if "removed" in msg or "deleted" in msg:
        return HTTPException(422, "This video has been removed.")
    if "login" in msg or "sign in" in msg:
        return HTTPException(422, "This video requires login to download.")
    if "copyright" in msg:
        return HTTPException(422, "This video is blocked due to copyright.")
    if "no video formats" in msg:
        return HTTPException(422, "No downloadable video formats found for this URL.")
    if "live" in msg or "is live" in msg:
        return HTTPException(422, "Live streams cannot be downloaded.")
    if "geo" in msg or "blocked" in msg:
        return HTTPException(422, "This content is geo-blocked in the server region.")
    return HTTPException(422, f"Could not process video: {str(e)[:300]}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "VaultDrop yt-dlp Server",
        "version": "4.0.0",
        "timestamp": int(time.time()),
        "ffmpeg_available": _ffmpeg_available(),
    }


@app.get("/health")
def health():
    try:
        yt_dlp_version = yt_dlp.version.__version__
    except Exception:
        yt_dlp_version = "unknown"
    return {
        "status": "ok",
        "version": "4.0.0",
        "yt_dlp_version": yt_dlp_version,
        "ffmpeg_available": _ffmpeg_available(),
        "timestamp": int(time.time()),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("20/minute")
async def analyze_video(req: AnalyzeRequest, request: Request):
    url = validate_url(req.url)
    logger.info(f"[ANALYZE] {url}")

    ydl_opts = {
        **COMMON_YDL_OPTS,
        "format": "bestvideo+bestaudio/best",
    }

    async with _YDL_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: _run_ydl_extract(url, ydl_opts))

            if info.get("_type") == "playlist":
                entries = info.get("entries", [])
                if not entries:
                    raise HTTPException(422, "Playlist is empty")
                info = entries[0]

            formats = info.get("formats", [])
            platform = _detect_platform(info, url)
            logger.info(f"[ANALYZE] Platform={platform} Title={info.get('title', 'N/A')!r} Formats={len(formats)}")

            seen_labels: set[str] = set()
            qualities: list[QualityOption] = []

            video_formats = [
                f for f in formats
                if f.get("url")
                and f.get("vcodec") not in (None, "none")
                and f.get("acodec") not in (None, "none")
                and f.get("height")
            ]

            video_only = [
                f for f in formats
                if f.get("url")
                and f.get("vcodec") not in (None, "none")
                and f.get("acodec") in (None, "none")
                and f.get("height")
            ]

            candidate_formats = video_formats if video_formats else video_only
            candidate_formats.sort(key=lambda f: f.get("height", 0), reverse=True)

            for f in candidate_formats:
                height = f.get("height", 0)
                if not height:
                    continue
                label = f"{height}p"
                if label in seen_labels:
                    continue
                seen_labels.add(label)

                if f.get("acodec") in (None, "none"):
                    audio_formats = [
                        af for af in formats
                        if af.get("url")
                        and af.get("acodec") not in (None, "none")
                        and af.get("vcodec") in (None, "none")
                    ]
                    if audio_formats:
                        best_audio = max(audio_formats, key=lambda af: af.get("abr") or 0)
                        format_id = f"{f['format_id']}+{best_audio['format_id']}"
                        filesize = (f.get("filesize") or f.get("filesize_approx") or 0) + \
                                   (best_audio.get("filesize") or best_audio.get("filesize_approx") or 0)
                    else:
                        format_id = f["format_id"]
                        filesize = f.get("filesize") or f.get("filesize_approx")
                else:
                    format_id = f["format_id"]
                    filesize = f.get("filesize") or f.get("filesize_approx")

                qualities.append(QualityOption(
                    label=label,
                    format_id=format_id,
                    filesize=int(filesize) if filesize else None,
                    filesize_human=_human_filesize(filesize),
                    ext=f.get("ext", "mp4"),
                    resolution=f"{f.get('width', '?')}x{height}",
                    fps=int(f["fps"]) if f.get("fps") else None,
                    vcodec=f.get("vcodec"),
                    acodec=f.get("acodec"),
                    tbr=f.get("tbr"),
                ))

                if len(qualities) >= 6:
                    break

            # Audio-only option
            audio_formats = [
                f for f in formats
                if f.get("url")
                and f.get("acodec") not in (None, "none")
                and f.get("vcodec") in (None, "none")
            ]
            if audio_formats:
                best_audio = max(audio_formats, key=lambda f: f.get("abr") or 0)
                filesize = best_audio.get("filesize") or best_audio.get("filesize_approx")
                # Prefer mp3 > m4a > opus > webm for audio-only
                audio_ext = best_audio.get("ext", "m4a")
                if audio_ext not in ("mp3", "m4a", "aac", "opus"):
                    audio_ext = "m4a"
                qualities.append(QualityOption(
                    label="Audio",
                    format_id=best_audio["format_id"],
                    filesize=int(filesize) if filesize else None,
                    filesize_human=_human_filesize(filesize),
                    ext=audio_ext,
                    resolution="Audio only",
                    acodec=best_audio.get("acodec"),
                    tbr=best_audio.get("abr"),
                ))

            if not qualities:
                qualities.append(QualityOption(
                    label="Best",
                    format_id="bestvideo+bestaudio/best",
                    ext="mp4",
                    resolution="Best available",
                ))

            duration_secs = int(info["duration"]) if info.get("duration") else None
            logger.info(f"[ANALYZE] Returning {len(qualities)} quality options")

            return AnalyzeResponse(
                title=info.get("title") or "Unknown Title",
                thumbnail=_best_thumbnail(info),
                duration=duration_secs,
                duration_label=_format_duration(duration_secs),
                uploader=info.get("uploader") or info.get("channel") or "Unknown",
                platform=platform,
                original_url=url,
                qualities=qualities,
            )

        except HTTPException:
            raise
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"[ANALYZE] DownloadError: {e}")
            raise _ydl_error_to_http(e)
        except Exception as e:
            logger.error(f"[ANALYZE] Unexpected error: {e}", exc_info=True)
            raise HTTPException(500, f"Server error: {str(e)[:300]}")


def _run_ydl_extract(url: str, opts: dict) -> dict:
    """Run yt-dlp extract_info synchronously (called in thread pool)."""
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


@app.post("/download", response_model=DownloadResponse)
@limiter.limit("10/minute")
async def download_video(req: DownloadRequest, request: Request):
    """
    Resolve a direct download URL for a specific format_id.

    Strategy:
    1. If video+audio are in one stream → return direct URL (client downloads via Dio)
    2. If separate streams AND ffmpeg available → merge server-side into temp mp4,
       serve merged file via /merged/<token> endpoint
    3. If separate streams AND ffmpeg unavailable OR merge fails →
       return video URL only (audio missing) — client falls back gracefully
    """
    url = validate_url(req.url)
    format_id = req.format_id.strip()
    logger.info(f"[DOWNLOAD] url={url} format_id={format_id}")

    if not format_id:
        raise HTTPException(400, "format_id must not be empty")

    ydl_opts = {
        **COMMON_YDL_OPTS,
        "format": format_id,
    }

    async with _YDL_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: _run_ydl_extract(url, ydl_opts))

            if info.get("_type") == "playlist":
                entries = info.get("entries", [])
                if not entries:
                    raise HTTPException(422, "Playlist is empty")
                info = entries[0]

            formats = info.get("formats", [])
            logger.info(f"[DOWNLOAD] Title={info.get('title')!r} formats={len(formats)}")

            download_url, ext, filesize, quality_label, audio_url = _extract_best_url(info, formats)

            if not download_url:
                logger.error(f"[DOWNLOAD] No URL found for format_id={format_id}")
                raise HTTPException(422, "Could not extract a direct download URL for this video")

            # Guard: reject webpage URLs
            if any(x in download_url for x in ["youtube.com/watch", "youtu.be/", "tiktok.com/@"]):
                logger.error(f"[DOWNLOAD] Webpage URL returned: {download_url[:80]}")
                raise HTTPException(422, "Server returned a webpage URL instead of a stream URL. Try a different quality.")

            # --- Server-side ffmpeg merge if separate streams ---
            merged = False
            if audio_url and _ffmpeg_available():
                logger.info("[DOWNLOAD] Attempting server-side ffmpeg merge...")
                tmp_dir = tempfile.mkdtemp(prefix="vaultdrop_")
                _register_temp_dir(tmp_dir)
                merged_path = os.path.join(tmp_dir, f"merged_{uuid.uuid4().hex}.mp4")

                try:
                    merge_ok, merge_err = await loop.run_in_executor(
                        None,
                        lambda: _merge_video_audio_server(download_url, audio_url, merged_path)
                    )

                    if merge_ok:
                        # Store merged file for serving, schedule cleanup after 10 min
                        merged_token = uuid.uuid4().hex
                        _merged_files[merged_token] = {
                            "path": merged_path,
                            "tmp_dir": tmp_dir,
                            "created": time.time(),
                            "ext": "mp4",
                        }
                        # Schedule cleanup after 10 minutes
                        asyncio.create_task(_schedule_merged_cleanup(merged_token, delay=600))

                        merged_url = f"/merged/{merged_token}"
                        logger.info(f"[DOWNLOAD] Merge successful, token={merged_token}")
                        merged = True

                        return DownloadResponse(
                            downloadUrl=merged_url,
                            title=info.get("title") or "video",
                            ext="mp4",
                            filesize=os.path.getsize(merged_path),
                            thumbnail=_best_thumbnail(info),
                            duration=int(info["duration"]) if info.get("duration") else None,
                            uploader=info.get("uploader") or info.get("channel"),
                            qualityLabel=quality_label,
                            audioUrl=None,  # already merged
                            merged=True,
                        )
                    else:
                        logger.warning(f"[DOWNLOAD] ffmpeg merge failed: {merge_err} — falling back to video-only URL")
                        _cleanup_temp_dir(tmp_dir)

                except Exception as e:
                    logger.error(f"[DOWNLOAD] Merge exception: {e}")
                    _cleanup_temp_dir(tmp_dir)

            logger.info(f"[DOWNLOAD] OK ext={ext} filesize={filesize} merged={merged} url={download_url[:80]}...")

            return DownloadResponse(
                downloadUrl=download_url,
                title=info.get("title") or "video",
                ext=ext or "mp4",
                filesize=filesize,
                thumbnail=_best_thumbnail(info),
                duration=int(info["duration"]) if info.get("duration") else None,
                uploader=info.get("uploader") or info.get("channel"),
                qualityLabel=quality_label,
                audioUrl=audio_url,
                merged=merged,
            )

        except HTTPException:
            raise
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"[DOWNLOAD] DownloadError: {e}")
            raise _ydl_error_to_http(e)
        except Exception as e:
            logger.error(f"[DOWNLOAD] Unexpected error: {e}", exc_info=True)
            raise HTTPException(500, f"Server error: {str(e)[:300]}")


# ---------------------------------------------------------------------------
# Merged file serving — ephemeral endpoint for server-merged mp4 files
# ---------------------------------------------------------------------------
_merged_files: dict[str, dict] = {}


async def _schedule_merged_cleanup(token: str, delay: int):
    """Clean up merged temp file after delay seconds."""
    await asyncio.sleep(delay)
    entry = _merged_files.pop(token, None)
    if entry:
        _cleanup_temp_dir(entry["tmp_dir"])
        logger.info(f"[CLEANUP] Scheduled cleanup for token={token}")


from fastapi.responses import FileResponse as _FileResponse


@app.get("/merged/{token}")
async def serve_merged_file(token: str):
    """Serve a server-merged mp4 file by token (ephemeral — expires after 10 min)."""
    entry = _merged_files.get(token)
    if not entry:
        raise HTTPException(404, "Merged file not found or expired. Please re-download.")

    path = entry["path"]
    if not os.path.exists(path):
        _merged_files.pop(token, None)
        raise HTTPException(404, "Merged file missing on server.")

    return _FileResponse(
        path=path,
        media_type="video/mp4",
        filename=f"vaultdrop_merged_{token[:8]}.mp4",
    )


# ---------------------------------------------------------------------------
# Legacy /resolve endpoint — backward compatibility
# ---------------------------------------------------------------------------
def quality_to_format(quality: str) -> str:
    q = quality.lower().strip()
    if q == "audio":
        return "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"
    height_map = {
        "4k": 2160, "2160p": 2160, "1440p": 1440, "1080p": 1080,
        "720p": 720, "480p": 480, "360p": 360, "240p": 240,
    }
    height = height_map.get(q)
    if height:
        return (
            f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={height}]+bestaudio"
            f"/best[height<={height}]/best"
        )
    return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"


@app.post("/resolve")
@limiter.limit("20/minute")
async def resolve_video(req: ResolveRequest, request: Request):
    url = validate_url(req.url)
    fmt = quality_to_format(req.quality)
    logger.info(f"[RESOLVE] url={url} quality={req.quality}")
    ydl_opts = {**COMMON_YDL_OPTS, "format": fmt}

    async with _YDL_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: _run_ydl_extract(url, ydl_opts))

            if info.get("_type") == "playlist":
                entries = info.get("entries", [])
                if not entries:
                    raise HTTPException(422, "Playlist is empty")
                info = entries[0]

            formats = info.get("formats", [])
            download_url, ext, filesize, quality_label, audio_url = _extract_best_url(info, formats)

            if not download_url:
                raise HTTPException(422, "Could not extract a direct download URL for this video")

            logger.info(f"[RESOLVE] OK url={download_url[:80]}...")

            thumbnails = info.get("thumbnails", [])
            thumbnail = thumbnails[-1].get("url") if thumbnails else info.get("thumbnail")

            return {
                "downloadUrl": download_url,
                "title": info.get("title") or "video",
                "ext": ext or "mp4",
                "filesize": filesize,
                "thumbnail": thumbnail,
                "duration": int(info["duration"]) if info.get("duration") else None,
                "uploader": info.get("uploader") or info.get("channel"),
                "qualityLabel": quality_label,
                "audioUrl": audio_url,
            }

        except HTTPException:
            raise
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"[RESOLVE] DownloadError: {e}")
            raise _ydl_error_to_http(e)
        except Exception as e:
            logger.error(f"[RESOLVE] Unexpected error: {e}", exc_info=True)
            raise HTTPException(500, f"Server error: {str(e)[:300]}")
