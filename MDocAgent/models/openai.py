from models.base_model import BaseModel
from models.utils import encode_image
import requests
import base64
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)

class MyOpenAI(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.config.model
        self.client = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client.mount("http://", adapter)
        self.client.mount("https://", adapter)
        
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
    
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    def predict(self, question, texts = None, images = None, history = None):
        messages = self.process_message(question, texts, images, history)
        start_time = time.time()
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
        }
        
        # 使用重试机制和增加的超时时间
        try:
            response = self.client.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            resp_json = response.json()
            elapsed = time.time() - start_time
            result = resp_json["choices"][0]["message"]["content"]
            usage = resp_json.get("usage", None)
            print(f"[AgentLog] model={self.model}, question={question[:80]}...\nusage={usage}, elapsed={elapsed:.2f}s\nresult={result[:200]}...\n")
            messages.append(self.create_ans_message(result))
            return result, messages, usage, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Request failed after 3 retries: {str(e)}"
            print(f"[AgentLog] model={self.model}, question={question[:80]}...\nERROR: {error_msg}, elapsed={elapsed:.2f}s\n")
            # 返回错误信息而不是抛出异常，让程序继续运行
            error_result = f"Error: {error_msg}"
            messages.append(self.create_ans_message(error_result))
            return error_result, messages, None, elapsed
    
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
    