import pdb
import random

# support multi-turn
# https://github.com/njucckevin/SeeClick/blob/main/pretrain/task_prompts.py#L40
# text2bbox
_SYSTEM_text2pos = [
    "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions.",
    "Based on the screenshot of the page, I give a text description and you give its corresponding location.",
    "In the image above, I will give a series of descriptions of the elements to be clicked. Please predict where you want to click.",
    "I will give textual descriptions of certain elements in the screenshot. Please predict the location of the corresponding element.",
    "Please identify the coordinates of the webpage elements I describe based on the provided screenshot.",
    "Given a screenshot, I will describe specific elements; your task is to predict their locations.",
    "Using the image of this webpage, can you determine the coordinates of the elements I describe?",
    "In this webpage capture, I will describe certain elements. Please locate them for me.",
    "I'll provide textual descriptions of elements in this webpage screenshot. Can you find their coordinates?",
    "From the given webpage screenshot, I need you to identify the locations of described elements.",
    "Based on this screenshot, I'll describe some elements. Please pinpoint their exact locations.",
    "For the elements I describe in this page capture, can you predict their positions?",
    "I will describe elements from a webpage screenshot; your role is to locate them.",
    "Using the attached screenshot of a webpage, please find the coordinates of described elements.",
    "From the image of this webpage, I will describe elements for you to locate.",
    "I'll give descriptions of certain webpage elements; please identify where they are in this screenshot.",
    "On this webpage screenshot, I will point out elements; please predict their exact coordinates.",
    "In this web page image, please locate the elements as I describe them.",
    "Given this screenshot of a webpage, I'll describe some elements; locate them for me.",
    "Please use the provided webpage screenshot to locate the elements I describe.",
    "In the provided web page image, I'll describe specific elements. Identify their locations, please.",
    "With this screenshot of a webpage, can you locate the elements I describe?",
    "I will describe features on this webpage screenshot; please predict their positions.",
    "Using the screenshot of this webpage, identify the coordinates of elements I describe.",
    "On this webpage capture, I'll point out specific elements for you to locate.",
    "Please determine the location of elements I describe in this webpage screenshot.",
    "I'll describe certain elements on this webpage image; your task is to find their locations.",
    "Using this webpage screenshot, I'll describe some elements. Please locate them.",
    "Based on my descriptions, find the locations of elements in this webpage screenshot.",
    "In this web page capture, please predict the positions of elements I describe.",
    "I'll give textual clues about elements in this webpage screenshot; identify their coordinates.",
    "Using the provided screenshot, I'll describe webpage elements for you to locate.",
    "From this webpage image, I will describe specific elements. Please predict their exact locations."
]

_SYSTEM_pos2text = [
    "Based on the screenshot of the web page, I give you the location to click on and you predict the text content of the corresponding element.",
    "In the image above, I give a series of coordinates and ask you to describe the corresponding elements.",
    "On this page, I will give you a series of coordinates and ask you to predict the text of the clickable element that corresponds to these coordinates.",
    "Given a webpage screenshot, I provide coordinates; predict the text content of the elements at these locations.",
    "In this screenshot, I'll give coordinates and ask you to describe the text of the elements there.",
    "Using the provided image of the webpage, I'll specify locations; you predict the text content of those elements.",
    "With this webpage capture, I provide a series of coordinates; please identify the text content of each element.",
    "In this page image, I'll point to specific locations; you need to predict the text of the corresponding elements.",
    "From this screenshot, I'll give coordinates; can you describe the text of the elements at these points?",
    "Based on this web page screenshot, I provide coordinates; please predict the textual content at these spots.",
    "Using the given image of the webpage, I'll specify certain coordinates; describe the text of the elements there.",
    "On this captured webpage, I will give a series of coordinates; your task is to predict the text at these locations.",
    "With this webpage image, I provide coordinates; can you tell me the text of the elements at these points?",
    "In the provided webpage screenshot, I'll point out locations; please describe the text of the elements there.",
    "From this web page capture, I give specific coordinates; predict the text content of the elements at these locations.",
    "Using this screenshot of a webpage, I'll indicate coordinates; can you predict the text of the elements?",
    "On this image of a web page, I provide coordinates; you need to describe the text of the corresponding elements.",
    "Given this webpage capture, I'll specify locations; please predict the text content of the elements there.",
    "In this screenshot, I give a series of coordinates; your task is to predict the text content of the elements.",
    "From the given webpage image, I'll provide coordinates; can you describe the text of the elements at these points?",
    "On this captured webpage, I provide specific coordinates; you need to predict the text of the elements there.",
    "Using this web page screenshot, I'll indicate locations; please describe the text content of the elements.",
    "With this image of a webpage, I specify coordinates; your task is to predict the text of the corresponding elements.",
    "In this webpage capture, I'll give coordinates; can you predict the text content of the elements at these locations?",
    "Based on this screenshot, I provide a series of coordinates; describe the text of the elements there.",
    "Using the image of this webpage, I'll specify locations; you need to predict the text of the elements.",
    "On this page screenshot, I give coordinates; please predict the text content of the corresponding elements.",
    "From this webpage image, I'll indicate specific coordinates; can you describe the text of the elements?",
    "In this web page image, I provide coordinates; your task is to predict the text of the elements at these locations.",
    "Given this screenshot of a webpage, I specify locations; please describe the text of the elements there.",
    "Using the provided page image, I'll point to locations; you predict the text content of the elements.",
    "On this webpage capture, I provide a series of coordinates; can you predict the text of the elements?",
    "With this image of the web page, I give specific coordinates; your task is to describe the text of the elements at these points."
]

_SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
_SYSTEM_bbox = "The coordinates represent a bounding box [x1, y1, x2, y2] for an element, which are relative coordinates on the screenshot, scaled from 0 to 1."
_SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1000."
_SYSTEM_bbox_int = "The coordinates represent a bounding box [x1, y1, x2, y2] for an element, which are relative coordinates on the screenshot, scaled from 0 to 1000."


_USER_a = '<|image_1|>{system}{element}'
_USER_b = '{system}<|image_1|>{element}'
_USER_c = '{system}{element}<|image_1|>'
_USER_list = [_USER_a, _USER_b, _USER_c]

def grounding_to_qwen(element_name, image, sample_io=0, user_prompt_random=True, xy_int=False, uniform_prompt=False):
    """
    sample_io: {0: text2point, 1: text2bbox, 2: point2text, 3: bbox2text}
    """
    transformed_data = []
    user_content = []
    if sample_io in [0, 1]:
        system_prompt = random.choice(_SYSTEM_text2pos)
    elif sample_io in [2, 3]:
        system_prompt = random.choice(_SYSTEM_pos2text)
    else:
        raise ValueError(f"Invalid input type: {sample_i}")
    
    # align w. screenspot
    if uniform_prompt:
        system_prompt = _SYSTEM_text2pos[1]

    if sample_io in [0, 2]:
        if xy_int:
            system_prompt += ' ' + _SYSTEM_point_int
        else:
            system_prompt += ' ' + _SYSTEM_point
    elif sample_io in [1, 3]:
        if xy_int:
            system_prompt += ' ' + _SYSTEM_bbox_int
        else:
            system_prompt += ' ' + _SYSTEM_bbox
    else:
        raise ValueError(f"Invalid coordinate type: {sample_out}")

    '{system}<|image_1|>{element}'
    if user_prompt_random:
        user_prompt = random.choice(_USER_list)
    else:
        user_prompt = _USER_b

    if user_prompt == _USER_a:
        user_content.append(image)
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text",  "text": element_name})
    elif user_prompt == _USER_b:
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append(image)
        user_content.append({"type": "text",  "text": element_name})
    elif user_prompt == _USER_c:
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text",  "text": element_name})
        user_content.append(image)

    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data