

def parse_actions(action_dict):
    action = action_dict["action"]
    value = action_dict["value"]
    position = action_dict["position"]
    
    if action == "CLICK":
        return {"action_type": 4, "click_point": position}
    elif action == "TYPE":
        return {"action_type": 3, "typed_text": value}
    else:
        raise ValueError(f"Invalid action: {action}")