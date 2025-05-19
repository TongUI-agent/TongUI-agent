import pytest
import openai
import base64
import os
from tongui.data.template.screenspot import _SCREENSPOT_SYSTEM, _SYSTEM_point
def test_call_vllm():
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:50004/v1",
    )

    
    response = client.chat.completions.create(
        model="tongui-3b",
        messages=[
            {"role": "user", "content": "Hello, world!"},
        ],
    )
    print(response)
    
def test_call_vllm_with_image():
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:50004/v1",
    )
    
    # Get the absolute path to the image file
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "safari_google.png")
    
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="tongui-3b",
        messages=[
            {"role": "user", "content": 
                [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        },
                        "min_pixels": 224 * 224,
                        "max_pixels": 3000 * 28 * 28,
                    }
                ]
            },
        ],
    )
    print(response)
    


def test_call_vllm_with_grounding():
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:50004/v1",
    )
    
    # Get the absolute path to the image file
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "safari_google.png")
    
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point
    
    response = client.chat.completions.create(
        model="tongui-3b",
        messages=[
            {"role": "user", "content": 
                [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    },
                    {"type": "text", "text": "Search bar"}
                ]
            },
        ],
    )
    print(response)