"""
VaultDrop yt-dlp FastAPI Server
Supports 1000+ platforms: YouTube, Instagram, TikTok, Twitter/X, Facebook, Vimeo, and more.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Deploy to Railway / Render / any VPS — see README.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import re

app = FastAPI(title="VaultDrop yt-dlp Server", version="1.0.0")

# Allow all origins so the Flutter app can call from any platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ResolveRequest(BaseModel):
    url: str
    quality: str = "1080p"


class ResolveResponse(BaseModel):
    downloadUrl: str
    title: str
    ext: str
    filesize: int | None = None
    thumbnail: str | None = None
    duration: int | None = None
    uploader: str | None = None
    qualityLabel: str | None = None


def quality_to_format(quality: str) -> str:
    """Convert a quality label to a yt-dlp format selector."""
    q = quality.lower().strip()

    if q == "audio":
        # Best audio only, prefer m4a/mp3
        return "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"

    # Map label to max height
    height_map = {
        "4k": 2160, "2160p": 2160,
        "1440p": 1440,
        "1080p": 1080,
        "720p": 720,
        "480p": 480,
        "360p": 360,
        "240p": 240,
    }
    height = height_map.get(q)

    if height:
        # Prefer a single progressive file (video+audio merged), fall back to best available
        return (
            f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={height}]+bestaudio"
            f"/best[height<={height}]"
            f"/best"
        )

    # Default: best quality
    return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"


@app.get("/")
def health_check():
    return {"status": "ok", "service": "VaultDrop yt-dlp Server"}


@app.post("/resolve", response_model=ResolveResponse)
def resolve_video(req: ResolveRequest):
    """
    Resolve a video page URL into a direct download URL.
    Supports YouTube, Instagram, TikTok, Twitter/X, Facebook, Vimeo, and 1000+ more.
    """
    url = req.url.strip()

    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL — must start with http/https")

    fmt = quality_to_format(req.quality)

    ydl_opts = {
        "format": fmt,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Return info dict without downloading
        "skip_download": True,
        # Some platforms need a browser-like user agent
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # If it's a playlist, take the first entry
            if info.get("_type") == "playlist":
                entries = info.get("entries", [])
                if not entries:
                    raise HTTPException(status_code=422, detail="Playlist is empty")
                info = entries[0]

            # Resolve the best format URL
            formats = info.get("formats", [])
            requested_formats = info.get("requested_formats")

            download_url = None
            ext = info.get("ext", "mp4")
            filesize = None
            quality_label = None

            if requested_formats:
                # yt-dlp selected a specific format (or merged pair)
                # For merged formats, use the video format URL (actual download handled by yt-dlp)
                # We return the direct URL of the best single-file format instead
                best = None
                for f in reversed(formats):
                    if f.get("url") and f.get("vcodec") != "none" and f.get("acodec") != "none":
                        best = f
                        break
                if best:
                    download_url = best["url"]
                    ext = best.get("ext", "mp4")
                    filesize = best.get("filesize") or best.get("filesize_approx")
                    quality_label = best.get("format_note") or best.get("height") and f"{best['height']}p"
                else:
                    # Fall back to the first requested format URL
                    download_url = requested_formats[0].get("url")
                    ext = requested_formats[0].get("ext", "mp4")
                    filesize = requested_formats[0].get("filesize")
            elif info.get("url"):
                download_url = info["url"]
                ext = info.get("ext", "mp4")
                filesize = info.get("filesize") or info.get("filesize_approx")
            else:
                # Try to find any format with a direct URL
                for f in reversed(formats):
                    if f.get("url"):
                        download_url = f["url"]
                        ext = f.get("ext", "mp4")
                        filesize = f.get("filesize") or f.get("filesize_approx")
                        quality_label = f.get("format_note") or (f.get("height") and f"{f['height']}p")
                        break

            if not download_url:
                raise HTTPException(status_code=422, detail="Could not extract a direct download URL for this video")

            # Thumbnail: prefer the highest resolution
            thumbnails = info.get("thumbnails", [])
            thumbnail = None
            if thumbnails:
                thumbnail = thumbnails[-1].get("url")
            if not thumbnail:
                thumbnail = info.get("thumbnail")

            return ResolveResponse(
                downloadUrl=download_url,
                title=info.get("title") or "video",
                ext=ext or "mp4",
                filesize=int(filesize) if filesize else None,
                thumbnail=thumbnail,
                duration=int(info["duration"]) if info.get("duration") else None,
                uploader=info.get("uploader") or info.get("channel"),
                qualityLabel=quality_label,
            )

    except HTTPException:
        raise
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        # Provide friendly error messages
        if "private" in msg.lower():
            detail = "This video is private and cannot be downloaded."
        elif "age" in msg.lower():
            detail = "Age-restricted video — cannot be downloaded without login."
        elif "not available" in msg.lower() or "unavailable" in msg.lower():
            detail = "This video is not available in your region."
        elif "removed" in msg.lower() or "deleted" in msg.lower():
            detail = "This video has been removed."
        elif "login" in msg.lower() or "sign in" in msg.lower():
            detail = "This video requires login to download."
        elif "copyright" in msg.lower():
            detail = "This video is blocked due to copyright."
        elif "no video formats" in msg.lower():
            detail = "No downloadable video formats found for this URL."
        else:
            detail = f"Could not download: {msg[:300]}"
        raise HTTPException(status_code=422, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:300]}")
