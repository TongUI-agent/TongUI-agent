from openai import OpenAI
import base64
from pathlib import Path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1344 * 28 * 28

    image_path = "assets/safari_google.png"
    image_base64 = encode_image(image_path)

    user_content = [
        {"type": "text", "text": _SYSTEM},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        {"type": "text", "text": "search box"},
    ]
    response = client.chat.completions.create(
        model="tongui-3b",
        messages=[{"role": "user", "content": user_content}],
    )

    print(response)