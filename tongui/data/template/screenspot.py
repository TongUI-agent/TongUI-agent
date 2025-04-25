_SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
_SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
_SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."

_SCREENSPOT_USER = '<|image_1|>{system}{element}'

def screenspot_to_qwen(element_name, image, xy_int=False):
    transformed_data = []
    user_content = []

    if xy_int:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point_int
    else:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point

    '{system}<|image_1|>{element}'
    user_content.append({"type": "text", "text": system_prompt})
    user_content.append(image)
    user_content.append({"type": "text",  "text": element_name})

    # question = _SCREENSPOT_USER.format(system=_SCREENSPOT_SYSTEM, element=element_name)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data