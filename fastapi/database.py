from beanie import init_beanie
import traceback
import motor.motor_asyncio
from logger import get_logger
from schemas import Podcast

log = get_logger(__name__)


async def init_db():
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            "mongodb://mongo:27017"
        )
    except Exception as e:
        log.fatal(traceback.format_exc())
        exit(1)

    await init_beanie(database=client.podcast_db, document_models=[Podcast])
