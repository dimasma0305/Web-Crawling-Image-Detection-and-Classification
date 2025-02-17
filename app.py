from time import sleep
from flask import Flask, request, render_template, redirect, url_for, send_file, abort
import zipfile
import io
import imagescrapper
from threading import Thread
from pathlib import Path
from flask_socketio import SocketIO
from engineio.async_drivers import gevent as _
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from urllib.parse import urlparse
import logging
import asyncio

app = Flask(__name__)
socketio = SocketIO(app)
queue = []
threads: dict[str, tuple[Thread|str]] = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_thread_until_success(url):
    url_parsed = urlparse(url)
    logger.info(f"Crawling {url}")
    try:
        thread = Thread(target=asyncio.run, args=(crawl_and_watch(url),))
        thread.start()
        return threads.update({url_parsed.hostname: [thread, url]})
    except Exception as e:
        logger.error(f"Error crawling {url}: {e}")
        return run_thread_until_success(url)

class QueueThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            for _ in range(len(queue)):
                url = queue.pop(0)
                run_thread_until_success(url)
            sleep(1)

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, host):
        super().__init__()
        self.host = host

    def on_modified(self, event):
        if event.is_directory:
            return

        wait_time = 0.1
        max_wait_time = 5
        waited = 0
        while waited < max_wait_time:
            if os.access(event.src_path, os.R_OK):
                break
            sleep(wait_time)
            waited += wait_time

        if not os.access(event.src_path, os.R_OK):
            logger.warning(f"File not accessible: {event.src_path}")
            return

        relative_path = Path(event.src_path).relative_to(imagescrapper.SAVE_DIR / self.host)
        logger.info(f"File changed: {relative_path}")
        socketio.emit("file_changed", {"host": self.host, "file": str(relative_path)})

def start_observer(host):
    path = imagescrapper.SAVE_DIR / host
    path.mkdir(parents=True, exist_ok=True)
    event_handler = FileChangeHandler(host)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

async def crawl_and_watch(url):
    await imagescrapper.ImageScrapper(url).crawl()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/crawl")
def crawl():
    url = urlparse(request.args.get("url"))

    if not url:
        return "URL is required", 400
    path = imagescrapper.SAVE_DIR / url.hostname
    if path.exists():
        return redirect(url_for("show_images", host=url.hostname, img_type='original'))
    queue.append(url.geturl())
    start_observer(url.hostname)

    wait_time = 0.5
    max_wait_time = 30
    waited = 0
    while waited < max_wait_time:
        if path.exists():
            break
        sleep(wait_time)
        waited += wait_time

    if not path.exists():
        abort(404, description="Failed to create directory for images")

    return redirect(url_for("show_images", host=url.hostname, img_type='original'))

@app.route("/images/<host>")
def show_images(host):
    img_type = request.args.get('img_type', 'original')
    images = []
    if img_type == 'original':
        path = imagescrapper.SAVE_DIR / host
    elif img_type == 'grayscale':
        path = imagescrapper.GRAYSCALE_DIR / host
    elif img_type == 'edge':
        path = imagescrapper.EDGE_DIR / host
    elif img_type == 'bounding':
        path = imagescrapper.BOUNDING_DIR / host
    else:
        abort(404, description="Invalid image type")

    if not path.exists():
        abort(404, description="Host not found")
    for img in path.iterdir():
        images.append(img.name)
    return render_template("images.html", images=images, host=host, img_type=img_type)

@app.route("/images/<host>/refresh")
def refresh(host):
    thread_and_url = threads.get(host)

    if thread_and_url:
        _, url = thread_and_url
        try:
            queue.append(url)
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
        return redirect(url_for("show_images", host=host, img_type='original'))
    else:
        return redirect(url_for("show_images", host=host, img_type='original'))



@app.route("/images/<host>/<image>")
def show_image(host, image):
    img_type = request.args.get('img_type', 'original')
    if img_type == 'original':
        file_path = imagescrapper.SAVE_DIR / host / image
    elif img_type == 'grayscale':
        file_path = imagescrapper.GRAYSCALE_DIR / host / image
    elif img_type == 'edge':
        file_path = imagescrapper.EDGE_DIR / host / image
    elif img_type == 'bounding':
        file_path = imagescrapper.BOUNDING_DIR / host / image
    else:
        abort(404, description="Invalid image type")

    if not file_path.exists():
        abort(404, description="File not found")
    return send_file(file_path)

@app.route("/images/<host>/download")
def compress_image(host):
    file_path = imagescrapper.COMPRESS_DIR / host
    if not file_path.exists():
        abort(404, description="Host not found")
    zipBuffer = io.BytesIO()
    with zipfile.ZipFile(zipBuffer, "w", zipfile.ZIP_DEFLATED) as z:
        for img in file_path.iterdir():
            z.write(img, img.name)
    zipBuffer.seek(0)
    return send_file(
        zipBuffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{host}.zip",
    )

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html", error=error.description), 404

if __name__ == "__main__":
    try:
        queue_thread = QueueThread()
        queue_thread.start()
        socketio.run(app, debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
