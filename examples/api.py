import ast

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image

TONGUI_HUGGINGFACE_SOURCE = "Bofeee5675/TongUI"
TONGUI_HUGGINGFACE_MODEL = "Bofeee5675/TongUI-3B"
TONGUI_HUGGINGFACE_API = "/on_submit"


class TongUIProvider:
    """
    The TongUI provider is used to make calls to TongUI.
    """

    def __init__(self):
        self.client = Client(TONGUI_HUGGINGFACE_SOURCE)

    def extract_norm_point(self, response, image_url):
        if isinstance(image_url, str):
            image = Image.open(image_url)
        else:
            image = Image.fromarray(np.uint8(image_url))

        point = ast.literal_eval(response)
        if len(point) == 2:
            x, y = point[0] * image.width, point[1] * image.height
            return x, y
        else:
            return None

    def call(self, prompt, image_data):
        result = self.client.predict(
            image=handle_file(image_data),
            query=prompt,
            api_name=TONGUI_HUGGINGFACE_API,
        )
        print(result)
        img_url, pred = result[0], result[1]
        result = self.extract_norm_point(pred, img_url)
        return result


if __name__ == "__main__":
    tongui_provider = TongUIProvider()
    img_url = "assets/safari_google.png"
    query = "search box"
    result = tongui_provider.call(query, img_url)
    print(result)
