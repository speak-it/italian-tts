from beanie.sync import Document
from datetime import datetime
from typing import Union, List


class Podcast(Document):
    status: str
    voice: str
    article_urls: List[str]
    created_at: Union[datetime, None] = None
    file_path: Union[str, None] = None

    class Settings:
        name = "podcast"
