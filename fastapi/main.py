from fastapi import FastAPI, Path, status
from database import init_db
from service import Service
from schemas import *
from fastapi.responses import FileResponse
import os

app = FastAPI()

service = Service()

@app.on_event("startup")
async def init():
    await init_db()


@app.post("/articles", status_code=status.HTTP_201_CREATED, response_model=ArticlePostResponse)
async def post_articles(articleRequest: ArticlePostRequest):
    """
    Accepts a list of articles urls and adds them to the working queue.
    Returns the podcast id
    """
    return await service.create_podcast(articleRequest)


@app.get("/info/{podcast_id}", response_model=PodcastGetResponse, response_model_exclude_unset=True)
async def get_podcast_info(podcast_id: PydanticObjectId = Path(title="Id of the podcast")):
    """
    Retrieve information about a podcast given its id
    """
    return await service.get_info(podcast_id)


@app.get("/files/{podcast_id}")
async def get_podcast_file(podcast_id: PydanticObjectId = Path(title="Id of the podcast")):
    """
    Download podcast with given id
    """
    return FileResponse(await service.get_file_path(podcast_id), media_type="audio/mp3")
