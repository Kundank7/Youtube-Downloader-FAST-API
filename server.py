from fastapi import FastAPI, APIRouter, HTTPException, Response, Request
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import yt_dlp
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
import secrets

# === Basic setup ===
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

STORAGE_PATH = Path("/tmp/ytdownloader_storage")
STORAGE_PATH.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))
api_router = APIRouter(prefix="/api")

executor = ThreadPoolExecutor(max_workers=4)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# === Models ===
class VideoMetadata(BaseModel):
    id: str
    title: str
    thumbnail: str
    duration: int
    channel: str
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    formats: List[dict]


class MetadataRequest(BaseModel):
    url: str


class DownloadRequest(BaseModel):
    url: str
    format_id: str
    quality: str


class DownloadHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str
    title: str
    quality: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str


# === Utility functions ===
def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    return filename[:200]


# === Updated Metadata extractor ===
def get_video_metadata(url: str) -> dict:
    """Extract video metadata using yt-dlp with cookies and anti-block settings"""
    try:
        cookie_path = ROOT_DIR / "cookies.txt"

        ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": False,
    "cookiefile": str(cookie_path) if cookie_path.exists() else None,
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.5993.70 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
}


        logger.info(f"Fetching metadata for: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        formats_list = []
        video_qualities = ["1080", "720", "480", "360"]
        for quality in video_qualities:
            matching_formats = [
                f for f in info.get("formats", [])
                if f.get("height") == int(quality) and f.get("vcodec") != "none"
            ]
            if matching_formats:
                best_format = max(matching_formats, key=lambda x: x.get("filesize", 0) or 0)
                formats_list.append({
                    "quality": f"{quality}p",
                    "format_id": best_format.get("format_id"),
                    "ext": best_format.get("ext", "mp4"),
                    "filesize": best_format.get("filesize", 0),
                    "type": "video"
                })

        audio_formats = [
            f for f in info.get("formats", [])
            if f.get("acodec") != "none" and f.get("vcodec") == "none"
        ]
        if audio_formats:
            best_audio = max(audio_formats, key=lambda x: x.get("abr", 0) or 0)
            formats_list.append({
                "quality": "MP3",
                "format_id": best_audio.get("format_id"),
                "ext": "mp3",
                "filesize": best_audio.get("filesize", 0),
                "type": "audio"
            })

        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "thumbnail": info.get("thumbnail"),
            "duration": info.get("duration", 0),
            "channel": info.get("uploader", info.get("channel", "Unknown")),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "formats": formats_list
        }

    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch video metadata: {str(e)}")


# === Routes ===
@api_router.post("/metadata")
async def fetch_metadata(request: MetadataRequest):
    try:
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(executor, get_video_metadata, request.url)
        return metadata
    except Exception as e:
        logger.error(f"Error fetching metadata: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch video metadata: {str(e)}")


@api_router.post("/download")
async def download_video(request: DownloadRequest, req: Request):
    try:
        session_id = req.session.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            req.session["session_id"] = session_id

        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(executor, get_video_metadata, request.url)
        safe_title = sanitize_filename(metadata["title"])

        cookie_path = ROOT_DIR / "cookies.txt"

        common_opts = {
            "cookiefile": str(cookie_path) if cookie_path.exists() else None,
            "headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0.5993.70 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            "quiet": True,
            "no_warnings": True,
        }
        
        if request.quality == "MP3":
            output_filename = f"{safe_title}.mp3"
            output_path = STORAGE_PATH / output_filename
            ydl_opts = {
                **common_opts,
                "format": "bestaudio/best",
                "outtmpl": str(STORAGE_PATH / f"{safe_title}.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
            }
else:
    output_filename = f"{safe_title}.mp4"
    output_path = STORAGE_PATH / output_filename
    quality_num = request.quality.replace("p", "")
    ydl_opts = {
        **common_opts,
        "format": f"bestvideo[height<={quality_num}]+bestaudio/best[height<={quality_num}]",
        "outtmpl": str(STORAGE_PATH / f"{safe_title}.%(ext)s"),
        "merge_output_format": "mp4",
    }


        def download_file():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([request.url])

        await loop.run_in_executor(executor, download_file)
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Download failed - file not created")

        history_item = DownloadHistory(
            video_id=metadata["id"],
            title=metadata["title"],
            quality=request.quality,
            session_id=session_id
        )
        doc = history_item.model_dump()
        doc["timestamp"] = doc["timestamp"].isoformat()
        await db.download_history.insert_one(doc)

        def file_iterator():
            try:
                with open(output_path, "rb") as f:
                    yield from f
            finally:
                if output_path.exists():
                    output_path.unlink()
                    logger.info(f"Deleted file: {output_path}")

        return StreamingResponse(
            file_iterator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
        )

    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@api_router.get("/history")
async def get_history(request: Request):
    session_id = request.session.get("session_id")
    if not session_id:
        return []

    history = await db.download_history.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", -1).limit(20).to_list(20)

    for item in history:
        if isinstance(item["timestamp"], str):
            item["timestamp"] = datetime.fromisoformat(item["timestamp"])
    return history


# === Cleanup task ===
async def cleanup_old_files():
    while True:
        try:
            current_time = datetime.now()
            for file_path in STORAGE_PATH.glob("*"):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age > timedelta(hours=1):
                        file_path.unlink()
                        logger.info(f"Cleaned up stale file: {file_path}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")

        await asyncio.sleep(1800)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_files())


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


# === Middleware and router ===
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
