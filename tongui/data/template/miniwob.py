import pdb
### MiniWob
_MiniWob_SYSTEM = """You are an assistant trained to navigate the web.
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `TYPE`: Type a string into an element, value is a string to type and the position is not applicable.

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

MiniWob_SYSTEM_v2 = """You are an assistant trained to navigate the web.
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `TYPE`: Type a string into an element, value is a string to type and the position is not applicable.

Format your response as
Thought: <your reasoning process>
Action: <the next action>

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_MiniWob_USER = """{system}
Task: {task}
Action History: {action_history}
Observation: <|image_1|>
What is the next action?
"""

def miniwob_to_qwen(task, action_history, answer_dict=None, version="v1"):
    transformed_data = []
    user_content = []

    user_content.append({"type": "text", "text": _MiniWob_SYSTEM if version == "v1" else MiniWob_SYSTEM_v2})
    user_content.append({"type": "text", "text": f"Task: {task}"})
    user_content.extend(action_history)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data