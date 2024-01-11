import pika
from logger import get_logger
from multiprocessing import Process
from workers import FastPitchWorker, VitsWorker 
import os

log = get_logger(__name__)


def create_worker():
    """Creates a worker based on the model defined in the env variable: MODEL

    Returns
    -------
    Worker
        a concrete instance of a Worker
    """
    n_torch_threads = int(os.getenv("TORCH-THREADS", 1))
    model_name = str(os.getenv("MODEL", "FastPitch"))

    if model_name == "FastPitch":
        return FastPitchWorker(n_torch_threads)
    elif model_name == "Vits":
        return VitsWorker(n_torch_threads)
    else:
        log.fatal(f"No model found with name {model_name}.")
        exit(1)


def run():
    """Callable run by each process. Creates a consumer that receives podcast ids through the
    tts_queue.
    """
    worker = create_worker()
    connection = pika.BlockingConnection(
        pika.ConnectionParameters("rabbitmq", heartbeat=0))
    channel = connection.channel()
    channel.queue_declare(queue="tts_queue", durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="tts_queue",
                          on_message_callback=worker.run_inference)
    log.debug("TTS model loaded. Ready to consume.")
    channel.start_consuming()


if __name__ == "__main__":
    n_proc = int(os.getenv("PROCESSES", 1))

    # launch inference processes
    log.debug(f"SPAWNING {n_proc} PROCESSES.")
    processes = []
    for i in range(n_proc):
        process = Process(target=run)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
