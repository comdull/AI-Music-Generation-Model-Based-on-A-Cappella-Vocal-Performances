from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, uuid
from zipfile import ZipFile
from typing import Optional

from audio2midi import audio_to_midi
from bert_sentiment import analyze_lyrics_sentiment 
from generate_tracks import main_pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ZIP_DIR = "zips"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

@app.post("/generate_full_pipeline/")
async def api_full_pipeline(
    audio_file: UploadFile,
    lyrics_file: UploadFile,
    output_filename: Optional[str] = Form(None)
):
    audio_path = None

    try:
        # 1. 保存上传的音频文件
        audio_path = os.path.join(UPLOAD_DIR, audio_file.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # 读取歌词
        lyrics_bytes = await lyrics_file.read()
        lyrics = lyrics_bytes.decode("utf-8")

        # 音频转MIDI
        base_filename = os.path.splitext(audio_file.filename)[0]
        midi_path = os.path.join(OUTPUT_DIR, f"{base_filename}.mid")
        audio_to_midi(audio_path, midi_path)

        # 分析情感
        sentiment_label, confidence = analyze_lyrics_sentiment(lyrics)

        # 多轨生成
        main_pipeline(
            melody_midi_path=midi_path,
            sentiment_label=sentiment_label,
            output_dir=OUTPUT_DIR,
        )

        return {
            "message": "Full pipeline completed successfully!",
            "sentiment_analysis": {
                "label": sentiment_label,
                "confidence": confidence
            },
            "output_file": f"/download_zip/"  # ✅ 一定要返回 zip 路径！
        }

    except Exception as e:
        return {"error": str(e), "message": "Pipeline failed."}

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/download_zip/")
async def download_zip():
    files_to_zip = [
        "MELODY2BRIDGE_generated_from_input.mid",
        "MELODY2PIANO_generated_from_input.mid",
    ]

    files_exist = [f for f in files_to_zip if os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if not files_exist:
        return {"error": "No files found to zip."}

    zip_filename = f"generated_tracks_{uuid.uuid4()}.zip"
    zip_path = os.path.join(ZIP_DIR, zip_filename)

    with ZipFile(zip_path, "w") as zipf:
        for f in files_exist:
            abs_path = os.path.join(OUTPUT_DIR, f)
            zipf.write(abs_path, arcname=f)

    return FileResponse(
        path=zip_path,
        filename="Tracks.zip",
        media_type="application/zip"
    )
