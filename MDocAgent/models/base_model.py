import torch
from models.utils import encode_image
import mimetypes

class BaseModel():
    def __init__(self, config):
        """
        Base model constructor to initialize common attributes.
        :param config: A dictionary containing model configuration parameters.
        """
        self.config = config
        
    def predict(self, question, texts = None, images = None, history = None):
        pass
    
    def clean_up(self):
        torch.cuda.empty_cache()
    
    def process_message(self, question, texts, images, history):
        def get_image_format(image_path):
            mime = mimetypes.guess_type(image_path)[0]
            if mime:
                return mime.split('/')[-1]
            return 'jpeg'  # 默认

        if history is not None:
            assert(self.is_valid_history(history))
            messages = history.copy()
        else:
            messages = []

        content = []
        if images is not None and len(images) > 0:
            for image_path in images:
                img_format = get_image_format(image_path)
                base64_img = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{base64_img}"
                    }
                })
        if texts is not None and len(texts) > 0:
            for text in texts:
                content.append({"type": "text", "text": text})
        if question is not None and question != "":
            content.append({"type": "text", "text": question})

        if len(content) > 0:
            messages.append({
                "role": "user",
                "content": content
            })
        return messages
    
    def is_valid_history(self, history):
        return True