import base64
import io

from PIL import Image


def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
