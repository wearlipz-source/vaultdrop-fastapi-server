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
# Shared yt-dlp options (metadata-only, no download)
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
    # Force working YouTube player clients that don't require PO tokens
    "extractor_args": {
        "youtube": {
            "player_client": ["tv,ios,android_vr,web_embedded"],
            "player_skip": ["webpage"],
        }
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
# Server-side CDN URL guard — ensures we never accidentally return a raw CDN URL
# ---------------------------------------------------------------------------
_CDN_PATTERNS = [
    "googlevideo.com",
    "storage.googleapis.com",
    "youtube.com/videoplayback",
    "tiktokcdn.com",
    "fbcdn.net",
    "cdninstagram.com",
    "akamaized.net",
    "cloudfront.net",
    "twitch.tv/vod",
    "redd.it",
    "v.redd.it",
]


def _is_cdn_url(url: str) -> bool:
    """Return True if the URL looks like a temporary CDN stream URL."""
    lower = url.lower()
    return any(p in lower for p in _CDN_PATTERNS)


def _assert_railway_url(url: str, context: str = ""):
    """Raise HTTPException if url is a CDN URL instead of a /merged/ Railway URL."""
    if not url.startswith("/merged/"):
        if _is_cdn_url(url):
            logger.error(f"[GUARD] {context} CDN URL leaked: {url[:80]}")
            raise HTTPException(
                500,
                "Server configuration error: backend returned a CDN URL instead of a "
                "Railway-hosted /merged/ URL. Ensure the Railway deployment is running "
                "VaultDrop v4.0+ with server-side download enabled.",
            )


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


# ---------------------------------------------------------------------------
# Server-side full download via yt-dlp (actual file download, not CDN URL)
# ---------------------------------------------------------------------------
def _download_and_merge_server(
    url: str,
    format_id: str,
    output_path: str,
    timeout_seconds: int = 1800,
) -> tuple[bool, str, dict]:
    """
    Use yt-dlp to fully download video+audio and merge into a single mp4 on disk.
    yt-dlp handles the ffmpeg merge internally when format_id contains '+'.
    Returns (success: bool, error_message: str, info: dict).
    """
    info_holder = {}
    try:
        # outtmpl must NOT include the extension — yt-dlp appends it automatically.
        # If we pass "merged_TOKEN.mp4", yt-dlp creates "merged_TOKEN.mp4.mp4".
        # Strip any extension from output_path and let merge_output_format=mp4 handle it.
        base_path = output_path
        if base_path.endswith(".mp4"):
            base_path = base_path[:-4]

        # Detect platform for platform-specific options
        is_youtube = any(d in url.lower() for d in ["youtube.com", "youtu.be"])
        is_facebook = any(d in url.lower() for d in ["facebook.com", "fb.watch", "fb.com"])
        is_instagram = "instagram.com" in url.lower()
        is_twitter = any(d in url.lower() for d in ["twitter.com", "x.com"])

        # Build platform-specific HTTP headers
        http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        if is_facebook:
            http_headers["Referer"] = "https://www.facebook.com/"
            http_headers["Sec-Fetch-Dest"] = "document"
            http_headers["Sec-Fetch-Mode"] = "navigate"
            http_headers["Sec-Fetch-Site"] = "none"
        elif is_instagram:
            http_headers["Referer"] = "https://www.instagram.com/"
        elif is_twitter:
            http_headers["Referer"] = "https://twitter.com/"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": False,          # MUST be False — actually download the file
            "simulate": False,               # Explicitly disable simulate mode
            "format": format_id,
            "outtmpl": base_path + ".%(ext)s",  # yt-dlp fills in the correct extension
            "merge_output_format": "mp4",
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
            "socket_timeout": 60,
            "http_headers": http_headers,
        }

        # YouTube-specific: force working player clients that bypass bot detection / PO token requirement
        if is_youtube:
            ydl_opts["extractor_args"] = {
                "youtube": {
                    "player_client": ["tv,ios,android_vr,web_embedded"],
                    "player_skip": ["webpage"],
                }
            }
            ydl_opts["sleep_interval_requests"] = 1
            ydl_opts["sleep_interval"] = 2

        # Facebook-specific extractor args to improve download reliability
        elif is_facebook:
            ydl_opts["extractor_args"] = {
                "facebook": {
                    "webpage_url_basename": ["video"],
                }
            }

        logger.info(f"[SERVER-DL] Starting yt-dlp download: format={format_id} → {base_path}.mp4")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                info_holder.update(info)

        # yt-dlp saves the merged file as base_path + ".mp4" (due to merge_output_format=mp4)
        # but search the temp dir for any completed file in case the name differs.
        parent = os.path.dirname(base_path)
        stem = os.path.basename(base_path)

        # Priority 1: expected merged mp4
        expected_mp4 = base_path + ".mp4"
        if os.path.exists(expected_mp4) and os.path.getsize(expected_mp4) > 0:
            actual_path = expected_mp4
        else:
            # Priority 2: scan temp dir for any non-partial file matching our stem
            candidates = []
            try:
                for f in os.listdir(parent):
                    full = os.path.join(parent, f)
                    if (
                        os.path.isfile(full)
                        and not f.endswith(".part")
                        and not f.endswith(".ytdl")
                        and not f.endswith(".tmp")
                        and os.path.getsize(full) > 0
                    ):
                        candidates.append(full)
            except Exception as scan_err:
                logger.warning(f"[SERVER-DL] Dir scan error: {scan_err}")

            if not candidates:
                return False, "Downloaded file not found on disk after yt-dlp completed", info_holder

            # Pick the largest file (most likely the merged video)
            actual_path = max(candidates, key=os.path.getsize)

        size = os.path.getsize(actual_path)
        if size == 0:
            return False, "Downloaded file is empty (0 bytes)", info_holder

        logger.info(f"[SERVER-DL] Download complete: {actual_path} ({size:,} bytes)")

        # Normalise to the expected output_path (always .mp4)
        if actual_path != output_path:
            shutil.move(actual_path, output_path)
            logger.info(f"[SERVER-DL] Moved {actual_path} → {output_path}")

        return True, "", info_holder

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"[SERVER-DL] yt-dlp DownloadError: {e}")
        return False, str(e), info_holder
    except Exception as e:
        logger.error(f"[SERVER-DL] Unexpected error: {e}")
        return False, str(e), info_holder


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
    merged: bool = True


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
    Fully download and merge video+audio on the Railway server using yt-dlp + ffmpeg.
    Saves the merged mp4 locally and returns ONLY a Railway-hosted /merged/{token} URL.
    Flutter NEVER receives YouTube CDN URLs directly.
    """
    url = validate_url(req.url)
    format_id = req.format_id.strip()
    logger.info(f"[DOWNLOAD] url={url} format_id={format_id}")

    if not format_id:
        raise HTTPException(400, "format_id must not be empty")

    # First extract metadata (no download) to get title/thumbnail/duration
    meta_opts = {**COMMON_YDL_OPTS, "format": format_id}
    async with _YDL_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()

            # Step 1: Extract metadata
            logger.info(f"[DOWNLOAD] Extracting metadata for: {url}")
            try:
                info = await loop.run_in_executor(None, lambda: _run_ydl_extract(url, meta_opts))
                if info.get("_type") == "playlist":
                    entries = info.get("entries", [])
                    info = entries[0] if entries else {}
            except Exception as meta_err:
                logger.warning(f"[DOWNLOAD] Metadata extraction failed (non-fatal): {meta_err}")
                info = {}

            title = info.get("title") or "video"
            thumbnail = _best_thumbnail(info)
            duration = int(info["duration"]) if info.get("duration") else None
            uploader = info.get("uploader") or info.get("channel")

            # Step 2: Full server-side download + merge
            tmp_dir = tempfile.mkdtemp(prefix="vaultdrop_")
            _register_temp_dir(tmp_dir)
            merged_token = uuid.uuid4().hex
            output_path = os.path.join(tmp_dir, f"merged_{merged_token}.mp4")

            logger.info(f"[DOWNLOAD] Starting server-side download → {output_path}")

            ok, err, dl_info = await loop.run_in_executor(
                None,
                lambda: _download_and_merge_server(url, format_id, output_path)
            )

            if not ok:
                _cleanup_temp_dir(tmp_dir)
                logger.error(f"[DOWNLOAD] Server-side download failed: {err}")
                # Map yt-dlp errors to friendly messages
                err_lower = err.lower()
                if "private" in err_lower:
                    raise HTTPException(422, "This video is private and cannot be downloaded.")
                if "age" in err_lower:
                    raise HTTPException(422, "Age-restricted video — cannot be downloaded without login.")
                if "not available" in err_lower or "unavailable" in err_lower:
                    raise HTTPException(422, "This video is not available in your region.")
                if "removed" in err_lower or "deleted" in err_lower:
                    raise HTTPException(422, "This video has been removed.")
                if "login" in err_lower or "sign in" in err_lower:
                    raise HTTPException(422, "This video requires login to download.")
                if "copyright" in err_lower:
                    raise HTTPException(422, "This video is blocked due to copyright.")
                if "live" in err_lower or "is live" in err_lower:
                    raise HTTPException(422, "Live streams cannot be downloaded.")
                if "geo" in err_lower or "blocked" in err_lower:
                    raise HTTPException(422, "This content is geo-blocked in the server region.")
                raise HTTPException(500, f"Server-side download failed: {err[:300]}")

            # Use info from download if metadata extraction failed earlier
            if dl_info and not title or title == "video":
                title = dl_info.get("title") or title
            if dl_info and not thumbnail:
                thumbnail = _best_thumbnail(dl_info)
            if dl_info and not duration:
                duration = int(dl_info["duration"]) if dl_info.get("duration") else None
            if dl_info and not uploader:
                uploader = dl_info.get("uploader") or dl_info.get("channel")

            file_size = os.path.getsize(output_path)

            # Store merged file for serving, schedule cleanup after 30 minutes
            _merged_files[merged_token] = {
                "path": output_path,
                "tmp_dir": tmp_dir,
                "created": time.time(),
                "ext": "mp4",
            }
            asyncio.create_task(_schedule_merged_cleanup(merged_token, delay=1800))

            merged_url = f"/merged/{merged_token}"
            _assert_railway_url(merged_url, context="[DOWNLOAD]")
            logger.info(f"[DOWNLOAD] Ready: token={merged_token} size={file_size:,} bytes")

            return DownloadResponse(
                downloadUrl=merged_url,
                title=title,
                ext="mp4",
                filesize=file_size,
                thumbnail=thumbnail,
                duration=duration,
                uploader=uploader,
                qualityLabel=None,
                audioUrl=None,
                merged=True,
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
    """Serve a server-merged mp4 file by token (ephemeral — expires after 30 min)."""
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
        filename=f"vaultdrop_{token[:8]}.mp4",
    )


# ---------------------------------------------------------------------------
# Legacy /resolve endpoint — now also uses server-side download
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
    """
    Legacy endpoint — now performs full server-side download like /download.
    Returns /merged/{token} URL, never raw CDN URLs.
    """
    url = validate_url(req.url)
    fmt = quality_to_format(req.quality)
    logger.info(f"[RESOLVE] url={url} quality={req.quality} format={fmt}")

    async with _YDL_SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()

            # Extract metadata first
            meta_opts = {**COMMON_YDL_OPTS, "format": fmt}
            try:
                info = await loop.run_in_executor(None, lambda: _run_ydl_extract(url, meta_opts))
                if info.get("_type") == "playlist":
                    entries = info.get("entries", [])
                    info = entries[0] if entries else {}
            except Exception:
                info = {}

            title = info.get("title") or "video"
            thumbnail = _best_thumbnail(info)
            duration = int(info["duration"]) if info.get("duration") else None
            uploader = info.get("uploader") or info.get("channel")

            # Full server-side download
            tmp_dir = tempfile.mkdtemp(prefix="vaultdrop_")
            _register_temp_dir(tmp_dir)
            merged_token = uuid.uuid4().hex
            output_path = os.path.join(tmp_dir, f"merged_{merged_token}.mp4")

            ok, err, dl_info = await loop.run_in_executor(
                None,
                lambda: _download_and_merge_server(url, fmt, output_path)
            )

            if not ok:
                _cleanup_temp_dir(tmp_dir)
                raise HTTPException(500, f"Server-side download failed: {err[:300]}")

            if dl_info and (not title or title == "video"):
                title = dl_info.get("title") or title
            if dl_info and not thumbnail:
                thumbnail = _best_thumbnail(dl_info)

            file_size = os.path.getsize(output_path)

            _merged_files[merged_token] = {
                "path": output_path,
                "tmp_dir": tmp_dir,
                "created": time.time(),
                "ext": "mp4",
            }
            asyncio.create_task(_schedule_merged_cleanup(merged_token, delay=1800))

            merged_url = f"/merged/{merged_token}"
            _assert_railway_url(merged_url, context="[RESOLVE]")
            logger.info(f"[RESOLVE] Ready: token={merged_token} size={file_size:,} bytes")

            return {
                "downloadUrl": merged_url,
                "title": title,
                "ext": "mp4",
                "filesize": file_size,
                "thumbnail": thumbnail,
                "duration": duration,
                "uploader": uploader,
                "qualityLabel": req.quality,
                "audioUrl": None,
                "merged": True,
            }

        except HTTPException:
            raise
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"[RESOLVE] DownloadError: {e}")
            raise _ydl_error_to_http(e)
        except Exception as e:
            logger.error(f"[RESOLVE] Unexpected error: {e}", exc_info=True)
            raise HTTPException(500, f"Server error: {str(e)[:300]}")
