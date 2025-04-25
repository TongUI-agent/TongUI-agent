import pdb
### AITW
_AITW_SYSTEM = """You are an assistant trained to navigate the mobile phone. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `TYPE`: Type a string into an element, value is a string to type and the position is not applicable.
3. `SELECT`: Select a value for an element, value is the value to select and the position is not applicable.
4. `SCROLL UP`: Scroll up for the screen.
5. `SCROLL DOWN`: Scroll down for the screen.
6. `SCROLL LEFT`: Scroll left for the screen.
7. `SCROLL RIGHT`: Scroll right for the screen.
8. `PRESS BACK`: Press for returning to the previous step, value and position are not applicable.
9. `PRESS HOME`: Press for returning to the home screen, value and position are not applicable.
10. `PRESS ENTER`: Press for submitting the input content, value and position are not applicable.
11. `STATUS TASK COMPLETE`: Indicate the task is completed, value and position are not applicable.
12. `STATUS TASK IMPOSSIBLE `: Indicate the task is impossible to complete, value and position are not applicable.
"""




_AITW_SYSTEM_json = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_AITW_SYSTEM_json_v2 = """
Format your response as
Thought: <your reasoning process>
Action: <the next action>

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_AITW_USER = """{system}
Task: {task}
Action History: {action_history}
Observation: <|image_1|>
"""

def aitw_to_qwen(task, action_history, answer_dict=None, skip_readme=False, version="v1"):
    transformed_data = []
    user_content = []

    if not skip_readme:
        if version == "v1":
            system_prompt = _AITW_SYSTEM + _AITW_SYSTEM_json
        elif version == "v2":
            system_prompt = _AITW_SYSTEM + _AITW_SYSTEM_json_v2
        else:
            raise ValueError(f"Invalid version: {version}")
    else:
        system_prompt = _AITW_SYSTEM_json

    user_content.append({"type": "text", "text": system_prompt})
    user_content.append({"type": "text", "text": f"Task: {task}"})
    user_content.extend(action_history)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data