import json

def convert_to_llama_factory(source, answer, image_list = None):
    
    
    output = dict()
    images = []
    messages = []
    for turn in source:
        role = turn['role']
        content = turn['content']
        if type(content) is str:
            messages.append({"role": role, "content": content})
        elif type(content) is list:
            values = ""
            image_idx = 0
            for item in content:
                # print(item)
                if item['type'] == 'image':
                    if image_list is not None:
                        image = image_list[image_idx]
                        image_idx += 1
                    else:
                        image = item['image']
                    images.append(image)
                    values += "<image>"
                elif item['type'] == 'text':
                    values += item['text']
            messages.append({"role": role, "content": values})
    # messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False) if type(answer) is dict else answer})
    
    output['messages'] = messages
    output['images'] = images
    return output