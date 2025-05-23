import os
from openai import OpenAI
import yaml # 导入 PyYAML 库

# 配置文件路径
CONFIG_PATH = "config/key.yaml"

def load_qwen_config(config_path=CONFIG_PATH):
    """从 YAML 文件加载 Qwen API 配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config and 'api_key' in config and 'qwen' in config['api_key']:
            return config['api_key']['qwen']
        else:
            raise ValueError("配置文件格式不正确或缺少 Qwen 配置")
    except FileNotFoundError:
        print(f"错误：配置文件 {config_path} 未找到。")
        return None
    except Exception as e:
        print(f"错误：加载配置文件失败：{e}")
        return None

# 加载配置
qwen_config = load_qwen_config()

if qwen_config:
    client = OpenAI(
        api_key=qwen_config.get("api_key"),
        base_url=qwen_config.get("base_url"),
    )

    model_name = qwen_config.get("model", "qwen-plus")
    temperature = qwen_config.get("temperature", 0.6) # 从配置加载，若无则默认0.7
    max_tokens = qwen_config.get("max_tokens", 150)    # 从配置加载，若无则默认150
    enable_thinking = qwen_config.get("enable_thinking", False)
    system_prompt_from_config = qwen_config.get("system_prompt") # 从配置加载

    actual_user_query = "你是谁？今天天气怎么样？" # 实际的用户测试问题

    

    try:
        completion_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt_from_config},
                {"role": "user", "content": actual_user_query}, 
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 检查配置中是否启用了思考过程功能
        if "enable_thinking" in qwen_config: 
            # 如果启用了思考过程，将其添加到请求参数中
            completion_params["extra_body"] = {"enable_thinking": enable_thinking}

        completion = client.chat.completions.create(**completion_params)
        # print(f"模型回复 ({model_name}):")
        print(completion.choices[0].message.content)
        # print(completion.model_dump_json()) # 如果需要完整的JSON输出，可以取消注释

    except Exception as e:
        print(f"调用 Qwen API 出错: {e}")
else:
    print("Qwen API 配置加载失败，无法执行 API 调用。")