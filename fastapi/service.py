from fastapi import HTTPException, status
from enums import Status
from schemas import *
import pika
import time


class Service:
    def __init__(self):
        time.sleep(10)  # wait for broker to start
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters("rabbitmq", heartbeat=0))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="tts_queue", durable=True)
        self.channel.confirm_delivery()

    async def create_podcast(self, articleRequest: ArticlePostRequest):
        podcast = Podcast(status=Status.NotStarted, voice=articleRequest.voice,
                          article_urls=articleRequest.article_urls)
        podcast = await podcast.insert()
        # add podcast id to worker's queue
        try:
            self.channel.basic_publish(exchange="",
                                       routing_key="tts_queue",
                                       body=str(podcast.id),
                                       properties=pika.BasicProperties(
                                           delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                                       ),
                                       mandatory=True)
        except pika.exceptions.UnroutableError:
            podcast.update({"$set": {Podcast.status: Status.Failed}})
        return ArticlePostResponse(podcast_id=podcast.id)

    async def get_info(self, podcast_id):
        podcast = await Podcast.get(podcast_id)
        if not podcast:
            raise (HTTPException(status_code=status.HTTP_404_NOT_FOUND))
        return PodcastGetResponse(id=podcast.id, status=podcast.status, voice=podcast.voice,
                                  created_at=podcast.created_at)

    async def get_file_path(self, podcast_id):
        podcast = await Podcast.get(podcast_id)
        if not podcast or not podcast.file_path:
            raise (HTTPException(status_code=status.HTTP_404_NOT_FOUND))
        return podcast.file_path
