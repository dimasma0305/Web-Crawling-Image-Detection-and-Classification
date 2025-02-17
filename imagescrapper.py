import base64
import hashlib
import pathlib
from urllib.parse import urljoin, urlparse
import httpx
from PIL import Image
import io
import cv2
import numpy as np
import asyncio
from playwright.async_api import async_playwright
from helper import process_image
import traceback
import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import detection
from torchvision import models, transforms

BASE_DIR = "files"

SAVE_DIR = pathlib.Path(__file__).parent / BASE_DIR / "images"
COMPRESS_DIR = pathlib.Path(__file__).parent / BASE_DIR / "compressed_images"
GRAYSCALE_DIR = pathlib.Path(__file__).parent / BASE_DIR / "grayscale_images"
EDGE_DIR = pathlib.Path(__file__).parent / BASE_DIR / "edge_images"
BOUNDING_DIR = pathlib.Path(__file__).parent / BASE_DIR / "bounding_images"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
COMPRESS_DIR.mkdir(parents=True, exist_ok=True)
GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
EDGE_DIR.mkdir(parents=True, exist_ok=True)
BOUNDING_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "ico", "svg", "webp"}

# Add CNN-specific configuration
CNN_MODEL_PATH = "./gender-detection-and-classification-image-dataset.pth"
CLASS_NAMES = ['men', 'women']
IMG_SIZE = 224

def convert_to_png(image):
    img = Image.open(io.BytesIO(image))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = buffer.getvalue()
    ext = "png"
    return ext, data

class ImageScrapper:
    def __init__(self, url, max_workers=5, max_depth=3):
        self.url = urlparse(url)
        self.raw_url = f"{self.url.scheme}://{self.url.netloc}"
        self.max_depth = max_depth
        self.already_visited = set()
        self.max_workers = max_workers
        self.gender_model = self._load_gender_model()
        self.detection_model = self._load_detection_model()
        self.transform = self._get_transforms()
        self.min_confidence = 0.50

    def _load_gender_model(self):
        """Load gender classification model"""
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
        model.eval()
        return model

    def _load_detection_model(self):
        """Load object detection model"""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        model.eval()
        return model.to('cpu')

    def _get_transforms(self):
        """Image transformations for gender model"""
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    async def __get_images_and_url(self, url="/"):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(urljoin(self.raw_url, url))
                old_content = await page.content()
                while True:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                    new_content = await page.content()
                    if old_content == new_content:
                        break
                    old_content = new_content

                images = await page.evaluate(
                    """
                    const imgElements = Array.from(document.images).map(img => img.src);
                    const inlineStyleImages = Array.from(document.querySelectorAll('[style*="url("]'))
                    .map(el => {
                        const style = el.style.backgroundImage || el.style.content;
                        const match = style.match(/url\\(["']?(.*?)["']?\\)/);
                        return match ? match[1] : null;
                    })
                    .filter(url => url); // Filter out any null values
                    const allImageUrls = [...imgElements, ...inlineStyleImages];
                    allImageUrls
                    """
                )
                urls = await page.evaluate(
                    """
                    const allLinks = Array.from(document.querySelectorAll('a')).map(a => a.href);
                    allLinks
                    """
                )
                images = set(images)
                return images, urls
            except Exception as e:
                print(f"Failed to get images and urls: {e}")
                await browser.close()
                browser = None
                await self.__get_images_and_url(url)
            finally:
                if browser:
                    await browser.close()

    async def __download_image(self, url):
        if url in self.already_visited:
            return
        try:
            if url.startswith("data:"):
                ext = url.split(";")[0].split("/")[1]
                data = url.split("base64,")[1]
                data = base64.b64decode(data)
                name = hashlib.md5(data).hexdigest()
            else:
                async with httpx.AsyncClient(base_url=self.raw_url) as client:
                    response = await client.get(url)
                    name = response.url.path.strip("/").split("/")[-1].split(".")[0]
                    ext = response.headers["content-type"].split("/")[1]
                    data = response.content

            if ext not in ALLOWED_EXTENSIONS:
                return
            if ext == "webp" or ext == "svg":
                ext, data = convert_to_png(data)

            file_path = SAVE_DIR / self.url.hostname / f"{name}.{ext}"

            if file_path.exists():
                return

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(data)

            await self.__compress_image(file_path, name, ext)
            await self.__process_image(file_path, name, ext)
            self.already_visited.add(url)
        except Exception as e:
            print(f"Failed to download image {url}: {e}")

    async def __compress_image(self, file_path, name, ext):
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=10)
                compressed_data = buffer.getvalue()
            compressed_path = COMPRESS_DIR / self.url.hostname / f"{name}.jpg"
            compressed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(compressed_path, "wb") as f:
                f.write(compressed_data)
        except Exception as e:
            print(f"Failed to compress image {file_path}: {e}")

    async def __process_image(self, file_path, name, ext):
        try:
            if ext.lower() == "png":
                with Image.open(file_path) as img:
                    jpg_path = file_path.with_suffix(".jpg")
                    img.convert("RGB").save(jpg_path, "JPEG")
                    os.remove(file_path)
                    file_path = jpg_path
                    ext = "jpg"

            with Image.open(file_path).convert("RGB") as img:
                # Object Detection
                detection_tensor = transforms.ToTensor()(img).unsqueeze(0).to('cpu')
                with torch.no_grad():
                    detections = self.detection_model(detection_tensor)[0]

                # Process detections
                img_array = np.array(img)
                bounding_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Filter person detections 
                person_boxes = detections['boxes'][detections['labels'] == 1]
                person_scores = detections['scores'][detections['labels'] == 1]
                
                for box, score in zip(person_boxes, person_scores):
                    if score < 0.8:  # Detection confidence threshold
                        continue
                        
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    
                    # Gender Classification
                    try:
                        crop = img.crop((x1, y1, x2, y2))
                        tensor = self.transform(crop).unsqueeze(0)
                        
                        with torch.no_grad():
                            outputs = self.gender_model(tensor)
                            probs = F.softmax(outputs, dim=1)
                            conf, pred = torch.max(probs, 1)

                        if conf.item() < self.min_confidence:
                            continue
                            
                        # Draw results
                        label = f"{CLASS_NAMES[pred.item()]} {conf.item():.2f}"
                        color = (0, 255, 0)  # BGR format
                        cv2.rectangle(bounding_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(bounding_image, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    except Exception as e:
                        print(f"Error processing detection: {e}")

                # Save processed images
                bounding_path = BOUNDING_DIR / self.url.hostname / f"{name}.{ext}"
                bounding_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(bounding_path), bounding_image)

                # Existing image processing
                img_array = np.array(img)
                _, gray_image, edges = process_image(img_array, method="canny")

                # Save processed images
                gray_path = GRAYSCALE_DIR / self.url.hostname / f"{name}.{ext}"
                gray_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gray_path), gray_image)

                edge_path = EDGE_DIR / self.url.hostname / f"{name}.{ext}"
                edge_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(edge_path), edges)

        except Exception as e:
            print(f"Failed to process image {file_path}: {e}")
            traceback.print_exc()

    async def crawl(self, url="/", depth=0):
        if depth > self.max_depth:
            return
        images, urls = await self.__get_images_and_url(url)

        tasks = []
        for image in images:
            tasks.append(asyncio.create_task(self.__download_image(image)))

        await asyncio.gather(*tasks)

        for link in urls:
            parsed = urlparse(link)
            if (self.url.netloc != parsed.netloc) and (not link.startswith("/")):
                continue
            if link in self.already_visited:
                continue
            self.already_visited.add(link)
            await self.crawl(link, depth + 1)


if __name__ == "__main__":
    import sys
    asyncio.run(ImageScrapper(sys.argv[1], max_depth=1).crawl())
