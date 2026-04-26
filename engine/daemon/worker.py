import os
import redis
from rq import Worker, Queue
from engine.core.logger import daemon_logger

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    daemon_logger.info("Starting RQ Worker... Listening to 'default' queue.")

    queue = Queue('default', connection=conn)
    worker = Worker([queue], connection=conn)
    worker.work()
