import copy
import json
import asyncio
import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import logging
from typing import Dict
import base64


class Config:
    dataset_file: str = (
        ""  # 数据集文件路径
    )
    # 输出结果保存的根目录
    outcome_dir: str = ""
    # model_name: str = "google/gemini-2.5-pro-preview-03-25"  # 模型名称
    # model_name: str = "openai/o4-mini-high"  # 模型名称
    # model_name: str = "openai/gpt-4o"  # 模型名称
    # model_name: str = "meta-llama/llama-4-maverick"  # 模型名称
    # model_name: str = "meta-llama/llama-3.3-70b-instruct"  # 模型名称
    # model_name: str = "google/gemma-3-27b-it"  # 模型名称
    # model_name: str = "mistralai/mistral-small-3.1-24b-instruct"  # 模型名称
    # model_name: str = "qvq-72b-preview"  # 模型名称
    model_name: str = "qwen/qwen2.5-vl-32b-instruct"  # 模型名称
    # model_name: str = "x-ai/grok-2-vision-1212"  # 模型名称
    # model_name: str = "moonshotai/kimi-vl-a3b-thinking:free"  # 模型名称
    # model_name: str = "claude-3-7-sonnet-20250219"  # 模型名称
    # model_name: str = "anthropic/claude-3.7-sonnet"  # 模型名称
    # model_name: str = "anthropic/claude-3.7-sonnet:thinking"  # 模型名称
    rpm: int = 100  # 每分钟请求数限制
    max_no_improve_round_count: int = 3  # 最大连续未成功请求的轮数
    process_count: int = -1  # 每轮处理的数据项数量（-1为全部）
    # openrouter
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://openrouter.ai/api/v1",
    )
    # # alibaba dashscope
    # client: AsyncOpenAI = AsyncOpenAI(
    #     api_key="",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # # xai grok
    # client: AsyncOpenAI = AsyncOpenAI(
    #     api_key="",
    #     base_url="https://api.x.ai/v1",
    # )
    # # xai grok
    # client: AsyncOpenAI = AsyncOpenAI(
    #     api_key="",
    #     base_url="https://api.anthropic.com/v1",
    # )

    max_input_images: int = 1000  # 最大输入图片数量

    @staticmethod
    def setup_logging(output_dir: str):
        """设置日志配置"""
        log_file = os.path.join(output_dir, "execution.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)


def create_output_dirs(config: Config) -> str:
    """创建以时间戳、处理数据量命名的输出目录"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 提取模型名称的最后一部分
    model_name_part = config.model_name.split("/")[-1]
    output_dir = os.path.join(
        config.outcome_dir,
        config.dataset_file.split("/")[-1].split(".")[0],
        model_name_part,
        f"{timestamp}_process_count_{config.process_count}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rounds_outcome"), exist_ok=True)
    return output_dir


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"图片编码错误 {image_path}: {str(e)}")
        return ""


async def process_one_item(
    data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    """负责处理单个数据项，包含速率限制。针对本次数据集任务改进"""
    async with limiter:
        try:
            # 准备消息内容
            messages = []

            # 添加系统消息
            if "system_input" in data and data["system_input"]:
                messages.append({"role": "system", "content": data["system_input"]})

            # 准备用户消息内容和图片
            user_content = []

            # 添加文本内容
            if "user_input" in data and data["user_input"]:
                user_content.append({"type": "text", "text": data["user_input"]})

            # 添加图片内容
            if "images" in data and data["images"]:
                # 截断图片数量
                images = data["images"][: config.max_input_images]
                for image_path in images:
                    base64_image = encode_image_to_base64(image_path)
                    if base64_image:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            }
                        )

            # 添加用户消息
            if user_content:
                messages.append({"role": "user", "content": user_content})

            # 调用OpenAI API执行任务
            # 声明一个response变量
            response = None
            response = await config.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
            )

            data["execution_outcome"] = {
                "is_success": True,
                "raw_response": response.model_dump_json(),
                "extracted_content": response.choices[0].message.content,
            }
        except Exception as e:
            if response is not None:
                error_message = str(e) + "\t" + response.model_dump_json()
            else:
                error_message = str(e)
            logging.error(f"处理数据时出错: {error_message}")
            data["execution_outcome"] = {
                "is_success": False,
                "error": error_message,
            }
        return data


# 负责一轮处理
async def process_round(
    config: Config,
    limiter: aiolimiter.AsyncLimiter,
    pending_items_map: Dict,
    final_results: Dict,
) -> Dict:
    # 创建任务列表
    tasks = []
    for index, item in pending_items_map.items():
        tasks.append(process_one_item(item, config, limiter))

    current_round_results = await tqdm_asyncio.gather(*tasks)

    for i, result in enumerate(current_round_results):
        index = list(pending_items_map.keys())[i]
        final_results[index] = result

    return final_results


# 异步调用的主函数
async def main_async():
    # 基础配置
    config = Config()
    output_dir = create_output_dirs(config)
    config.setup_logging(output_dir)

    round_outcome_dir = os.path.join(output_dir, "rounds_outcome")
    limiter = aiolimiter.AsyncLimiter(1, 60.0 / config.rpm)

    with open(config.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if config.process_count > 0:
        dataset = dataset[: config.process_count]

    final_results = copy.deepcopy(dataset)
    pending_items_map = {i: dataset[i] for i in range(len(dataset))}
    round_count = 0

    # 主循环，每次循环代表一轮，会处理所有没有处理（或处理失败）的数据
    while (
        len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(f"第{round_count}轮，剩余未处理数据: {len(pending_items_map)}")

        # 处理一轮
        final_results = await process_round(
            config, limiter, pending_items_map, final_results
        )
        with open(
            os.path.join(round_outcome_dir, f"round_{round_count}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if not final_results[i]["execution_outcome"]["is_success"]
        }
        if len(pending_items_map) == 0:
            break

    with open(
        os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main_async())
