### Mind2Web
_MIND2WEB_SYSTEM = """You are an assistant trained to navigate the web. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is the element to click and the position [x,y] is required.
2. `TYPE`: Type a string into an element, value is the string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is the value to select and the position [x,y] is required.

Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_MIND2WEB_SYSTEM_v2 = """You are an assistant trained to navigate the web. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is the element to click and the position [x,y] is required.
2. `TYPE`: Type a string into an element, value is the string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is the value to select and the position [x,y] is required.

Format your response as
Thought: <your reasoning process>
Action: <the next action>

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_MIND2WEB_USER = """{system}
Task: {task}
Observation: <|image_1|>
Action History: {action_history}
What is the next action?
"""

def mind2web_to_qwen(task, action_history, answer_dict=None, version="v1"):
    transformed_data = []
    user_content = []

    if version == "v1":
        user_content.append({"type": "text", "text": _MIND2WEB_SYSTEM})
    elif version == "v2":
        user_content.append({"type": "text", "text": _MIND2WEB_SYSTEM_v2})
    else:
        raise ValueError(f"Invalid version: {version}")
    user_content.append({"type": "text", "text": f"Task: {task}"})
    user_content.extend(action_history)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data