from pydantic import BaseModel, Field
from enums import *
from datetime import datetime
from beanie import Document, PydanticObjectId
from typing import Union


class Podcast(Document):
    status: str
    voice: str
    article_urls: list[str]
    created_at: Union[datetime, None] = None
    file_path: Union[str, None] = None

    class Settings:
        name = "podcast"


class PodcastGetResponse(BaseModel):
    id: PydanticObjectId
    status: str
    voice: str
    created_at: Union[datetime, None] = None


class ArticlePostResponse(BaseModel):
    podcast_id: PydanticObjectId


class ArticlePostRequest(BaseModel):
    article_urls: list[str]
    voice: Union[str, None] = Voice.Female1

    class Config:
        schema_extra = {
            "example": {
                "article_urls": ["https://www.ilgiornale.it/news/personaggi/boicottaggio-internazionale-minaccia-codacons-e-l-abbandono-2264291.html",
                                 "https://www.ilgiornale.it/news/personaggi/nessuna-incoronazione-frederik-danimarca-ecco-perch-2264306.html"],
                "voice": "Male1"
            }
        }
