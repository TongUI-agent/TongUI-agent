import os
import json
import time
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO

# from IPython.display import display
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from utils.agent_function_call import ComputerUse

from PIL import Image, ImageDraw, ImageColor


def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)
        except ValueError:
            color = (255, 0, 0, 128)
    else:
        color = (255, 0, 0, 128)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)], fill=color
    )

    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [
            (x - center_radius, y - center_radius),
            (x + center_radius, y + center_radius),
        ],
        fill=(0, 255, 0, 255),
    )

    image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)

    return combined.convert("RGB")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def perform_gui_grounding_with_api(
    screenshot_path,
    user_query,
    model_id,
    client,
    min_pixels=336 * 336,
    max_pixels=1350 * 28 * 28,
):
    """
    Perform GUI grounding using Qwen model to interpret user query on a screenshot.

    Args:
        screenshot_path (str): Path to the screenshot image
        user_query (str): User's query/instruction
        model: Preloaded Qwen model
        min_pixels: Minimum pixels for the image
        max_pixels: Maximum pixels for the image

    Returns:
        tuple: (output_text, display_image) - Model's output text and annotated image
    """

    # Open and process image
    input_image = Image.open(screenshot_path).convert("RGB")
    # base64_image = encode_image(screenshot_path)
    resized_height, resized_width = smart_resize(
        input_image.height,
        input_image.width,
        factor=28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_image = input_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    print("Original image: ", input_image.size)
    print("Resized image: ", resized_image.size)
    buffered = BytesIO()
    resized_image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Initialize computer use function
    computer_use = ComputerUse(
        cfg={"display_width_px": resized_height, "display_height_px": resized_width}
    )

    # Build messages
    system_message = NousFnCallPrompt().preprocess_fncall_messages(
        messages=[
            Message(
                role="system",
                content=[ContentItem(text="You are a helpful assistant.")],
            ),
        ],
        functions=[computer_use.function],
        lang=None,
    )
    system_message = system_message[0].model_dump()
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": msg["text"]}
                for msg in system_message["content"]
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": user_query},
            ],
        },
    ]
    # print(json.dumps(messages, indent=4))
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    output_text = completion.choices[0].message.content
    try:
        # Parse action and visualize
        action = json.loads(
            output_text.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
        )
        # display_image = input_image #.resize((input_image.width, input_image.height))
        # display_image = draw_point(
        #     display_image, action["arguments"]["coordinate"], color="green"
        # )
        # with open("temp.png", "wb") as f:
        #     display_image.save(f)
        coor = action["arguments"]["coordinate"]
        return [
            int(coor[0] * input_image.width / resized_width),
            int(coor[1] * input_image.height / resized_height),
        ]
    except:
        return [-1, -1]


if __name__ == "__main__":
    data = {
        "img_filename": "android_studio_mac/screenshot_2024-11-28_15-16-55.png",
        "bbox": [1774, 1586, 2113, 1618],
        "instruction": "modify the highlights of the photo with in the virtual android machine in android studio",
        "instruction_cn": "在 Android Studio 的安卓虚拟机中修改照片高光。",
        "id": "android_studio_macos_0",
        "application": "android_studio",
        "platform": "macos",
        "img_size": [3840, 2160],
        "ui_type": "icon",
        "group": "Dev",
    }
    client = OpenAI(
        api_key="empty",
        base_url="http://127.0.0.1:50004/v1",
    )
    s_t = time.time()
    coor = perform_gui_grounding_with_api(
        os.path.join("dataset/ScreenSpot-Pro/images", data["img_filename"]),
        data["instruction"],
        "qwen2.5-7b",
        client,
    )
    print(type(coor))
    print("time:", time.time() - s_t)
