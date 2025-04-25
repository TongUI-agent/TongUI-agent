import pdb
### NAV
_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""
_NAV_FORMAT = """
Format your response as
Thought: <your reasoning process>
Action: <the next action>

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

You can output multiple actions at once, and use JSON array to represent multiple actions.
If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

app_map = {
'desktop': 'computer',
'mobile': 'smartphone'
}

action_map = {
'desktop': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is start position of the scroll operation.
4. `LEFT_CLICK_DOUBLE`: Left click on an element twice, value is not applicable and the position [x,y] is required.
5. `RIGHT_CLICK_SINGLE`: Right click on an element once, value is not applicable and the position [x,y] is required.
6. `DRAG`: Drag the cursor to the specified position with the left button pressed. Value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the drag operation.
7. `HOT_KEY`: Press a hot key, value is the hot key and the position is not applicable.
8. `WAIT`: Wait for 5 seconds, and take a screenshot to check for any changes. Value and position are not applicable.
9. `FINISH`: Finish the task. Value and position are not applicable.
""",

'mobile': """
1. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `LONG_PRESS`: Long press on an element, value is the press duration and the position [x,y] is required.
5. `PRESS_HOME`: Press the home button, value and position are not applicable.
6. `PRESS_BACK`: Press the back button, value and position are not applicable.
7. `FINISH`: Submit the task regardless of whether it succeeds or fails. Value is the result of the task and position is not applicable.
"""
}

_NAV_USER = """{system}
Task: {task}
Observation: <|image_1|>
Action History: {action_history}
What is the next action?
"""

def navigation_to_qwen(task, image, action_history, split, answer_dict=None, skip_readme=False):
    transformed_data = []
    user_content = []
    if not skip_readme:
        system_prompt = _NAV_SYSTEM.format(_APP=app_map[split], _ACTION_SPACE=action_map[split])
    else:
        system_prompt = ""
    system_prompt += _NAV_FORMAT

    user_content.append({"type": "text", "text": system_prompt})
    user_content.append({"type": "text", "text": f"Task: {task}\n"})
    user_content.extend(action_history)

    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data

def format_box(box):
    box = eval(box)
    box = [round(box[0] / 1000, 2), round(box[1] / 1000, 2)]
    return box

def function_parser_desktop(function_str, app_type):
    # print("Parsing function string: ", function_str, app_type)
    
    func_name_proj = {
        "click": "CLICK",
        "left_double": "LEFT_CLICK_DOUBLE",
        "right_single": "RIGHT_CLICK_SINGLE",
        "drag": "DRAG",
        "hotkey": "HOT_KEY",
        "type": "INPUT",
        "scroll": "SCROLL",
        "finished": "FINISH",
        "wait": "WAIT",
        "call_user": "CALL_USER",
    }
    def click(start_box):
        start_box = format_box(start_box)
        args = {"action": "CLICK", "value": None, "position": start_box}
        return "click", args
    
    def left_double(start_box):
        start_box = format_box(start_box)
        args = {"action": "LEFT_CLICK_DOUBLE", "value": None, "position": start_box}
        return "left_double", args
    
    def right_single(start_box):
        start_box = format_box(start_box)
        args = {"action": "RIGHT_CLICK_SINGLE", "value": None, "position": start_box}
        return "right_single", args
    
    def drag(start_box, end_box):
        start_box = format_box(start_box)
        end_box = format_box(end_box)
        args = {"action": "DRAG", "value": None, "position": [start_box, end_box]}
        return "drag", args
    
    def hotkey(key):
        args = {"action": "HOT_KEY", "value": key, "position": None}
        return "hotkey", args
    
    def type(content):
        args = {"action": "INPUT", "value": content, "position": None}
        return "type", args
    
    def scroll(start_box, direction):
        start_box = format_box(start_box)
        args = {"action": "SCROLL", "value": direction, "position": start_box}
        return "scroll", args
    
    def finished():
        args = {"action": "FINISH", "value": None, "position": None}
        return "finished", args
    
    def wait():
        args = {"action": "WAIT", "value": None, "position": None}
        return "wait", args
    
    def call_user():
        args = {"action": "CALL_USER", "value": None, "position": None}
        return "call_user", args
    function_str = function_str.split("\n")
    function_str = [i.strip() for i in function_str if i.strip()]
    function_str = [i for i in function_str if len(i) > 0]

    func_names_outputs = []
    func_args_outputs = []
    for func in function_str:
        func_name, func_args = eval(func)
        func_names_outputs.append(func_name)
        func_args_outputs.append(func_args)
    if len(func_names_outputs) == 1:
        return func_names_outputs, func_args_outputs[0]
    else:
        return func_names_outputs, func_args_outputs

def function_parser_mobile(function_str, app_type):
    def click(start_box):
            start_box = format_box(start_box)
            args = {"action": "TAP", "value": None, "position": start_box}
            return "click", args
        
    def finished(content=None):
        args = {"action": "FINISH", "value": content, "position": None}
        return "finished", args
    
    def long_press(start_box, time):
        start_box = format_box(start_box)
        args = {"action": "LONG_PRESS", "value": int(time), "position": start_box}
        return "long_press", args
    
    def press_home():
        args = {"action": "PRESS_HOME", "value": None, "position": None}
        return "press_home", args
    
    def press_back():
        args = {"action": "PRESS_BACK", "value": None, "position": None}
        return "press_back", args
    
    def type(content):
        args = {"action": "INPUT", "value": content, "position": None}
        return "type", args

    def scroll(start_box, end_box):
        start_box = format_box(start_box)
        end_box = format_box(end_box)
        args = {"action": "SWIPE", "value": None, "position": [start_box, end_box]}
        return "scroll", args
    function_str = function_str.split("\n")
    function_str = [i.strip() for i in function_str if i.strip()]
    function_str = [i for i in function_str if len(i) > 0]
    func_names_outputs = []
    func_args_outputs = []
    for func in function_str:
        func_name, func_args = eval(func)
        func_names_outputs.append(func_name)
        func_args_outputs.append(func_args)
    if len(func_names_outputs) == 1:
        return func_names_outputs, func_args_outputs[0]
    else:
        return func_names_outputs, func_args_outputs
    
def parse_function_str(function_str, app_type):
    # print("Parsing function string: ", function_str, app_type)
    if app_type == "desktop":
        return function_parser_desktop(function_str, app_type)
    elif app_type == "mobile":
        return function_parser_mobile(function_str, app_type)
    else:
        raise ValueError(f"Invalid app type: {app_type}")
    
