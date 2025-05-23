import os
from ctypes import cdll, byref, string_at, c_void_p, CFUNCTYPE, c_char_p, c_uint64, c_int64
import pyaudio
from loguru import logger
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import wave
import jieba
import jieba.analyse
import yaml
import difflib
import sys
import asyncio
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.device import ModbusDeviceIdentification
from concurrent.futures import ThreadPoolExecutor
from request import text_to_speech
from change_state import ExpressionServer
from openai import AsyncOpenAI # LLM Client
from pymodbus.client import ModbusTcpClient
# from SparkApi import SparkApi  # For Spark LLM (if still used or for reference)

# --- Configuration ---
CONFIG_PATH = "config/key.yaml"

# 科大讯飞配置 (ASR/IVW)
XF_APPID = "74fa666f"
XF_APIKey = "1117d16429339b497550a24efbacfc44"
XF_APISecret = "YjBkMzc3NWU3NjYwZmY2MWNjMWY2YTc0"

# 音频录制参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 6

# --- Helper Functions ---
def setup_logger(logs_dir):
    logger.remove()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    common_args = {"rotation": "500 MB", "encoding": "utf-8", "enqueue": True, "retention": "10 days"}
    logger.add(os.path.join(logs_dir, f"info_{timestamp}.log"), level="INFO", **common_args)
    logger.add(os.path.join(logs_dir, f"error_{timestamp}.log"), level="ERROR", **common_args)
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

def load_api_config(config_path=CONFIG_PATH, service_name='qwen'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config and 'api_key' in config and service_name in config['api_key']:
            return config['api_key'][service_name]
        logger.warning(f"配置文件中未找到 {service_name} 的配置或格式不正确。")
        return None
    except FileNotFoundError:
        logger.error(f"错误：API配置文件 {config_path} 未找到。")
        return None
    except Exception as e:
        logger.error(f"错误：加载API配置文件失败：{e}")
        return None

# --- Main Class ---
class WakeWordJieba:
    def __init__(self, main_event_loop):
        self.main_event_loop = main_event_loop
        self.is_awake = False
        self.text_buffer = ""
        # not use modbus
        # self.modbus_ip = "192.168.1.125"
        self.modbus_ip = "127.0.0.1"
        self.modbus_port = 599
        self.store = ModbusSlaveContext(di=ModbusSequentialDataBlock(0,[0]*100), co=ModbusSequentialDataBlock(0,[0]*100), hr=ModbusSequentialDataBlock(0,[0]*100), ir=ModbusSequentialDataBlock(0,[0]*100))
        self.context = ModbusServerContext(slaves=self.store, single=True)

        self.keywords_dir = "./keywords"
        self.logs_dir = "./logs"
        self.keywords_dict_dir = "./keywords_dicts"
        for directory in [self.keywords_dir, self.logs_dir, self.keywords_dict_dir]:
            if not os.path.exists(directory): os.makedirs(directory)

        setup_logger(self.logs_dir)
        self.keywords_dict = self.load_keywords_dict()

        # LLM (Qwen) Configuration
        self.qwen_llm_config = load_api_config(service_name='qwen')
        self.openai_client = None
        self.qwen_system_prompt = "You are a helpful assistant." # Default
        if self.qwen_llm_config:
            if self.qwen_llm_config.get("api_key") and self.qwen_llm_config.get("base_url"):
                try:
                    self.openai_client = AsyncOpenAI(api_key=self.qwen_llm_config.get("api_key"), base_url=self.qwen_llm_config.get("base_url"))
                    logger.info("Qwen LLM 客户端初始化成功。")
                except Exception as e: logger.error(f"Qwen LLM 客户端初始化失败: {e}")
            else: logger.warning("Qwen LLM 的 api_key 或 base_url 未在配置中提供，LLM功能将受限。")
            
            loaded_prompt = self.qwen_llm_config.get("system_prompt")
            if loaded_prompt: self.qwen_system_prompt = loaded_prompt; logger.info("已从配置加载 Qwen system_prompt。")
            else: logger.warning("未在配置中找到 Qwen system_prompt，将使用默认提示。")
        else: logger.warning("Qwen LLM 配置未加载，通用对话功能将使用默认提示且可能无法工作。")

        # Xunfei IVW/ASR Configuration
        try:
            self.msc_load_library = './bin/msc_x64.dll'
            self.xf_app_id = XF_APPID
            self.ivw_threshold = '0:1450'
            self.jet_path = os.path.join(os.getcwd(), './bin/msc/res/ivw/wakeupresource.jet')
            self.work_dir = 'fo|' + self.jet_path
            self.dll = cdll.LoadLibrary(self.msc_load_library)
            self.CALLBACKFUNC = CFUNCTYPE(None, c_char_p, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p)
            self.callback = self.CALLBACKFUNC(self.wake_callback)
            logger.info("讯飞唤醒引擎回调设置完毕。")
        except Exception as e: logger.error(f"讯飞唤醒引擎初始化失败: {e}"); raise
        
        self.exit_event = asyncio.Event()
        self.expressions_server = ExpressionServer()
        self._init_expression_server()

    def _init_expression_server(self):
        try: self.expressions_server.start(); logger.info("表情服务器启动成功")
        except Exception as e: logger.error(f"表情服务器启动失败：{e}")

    def load_keywords_dict(self):
        try:
            dict_path = os.path.join(self.keywords_dict_dir, 'keywords_dict.yaml')
            if not os.path.exists(dict_path): logger.error(f"关键词词典文件不存在: {dict_path}"); return None
            with open(dict_path, 'r', encoding='utf-8') as f: keywords_data = yaml.safe_load(f)
            logger.info(f"成功加载关键词词典: {dict_path}"); return keywords_data
        except Exception as e: logger.error(f"加载关键词词典失败: {e}"); return None

    def wake_callback(self, sessionID, msg, param1, param2, info, userDate):
        logger.info("=" * 25 + " 检测到唤醒词! " + "=" * 25)
        text_to_speech("你好，需要我为你做什么")
        self.expressions_server.send("img", "blink")
        self.expressions_server.send("text_robot", "你好，需要我为你做什么")
        self.expressions_server.send("text_start", "start")
        self.is_awake = True

    async def request_monitor(self):
        while not self.exit_event.is_set():
            await asyncio.sleep(1)
            try:
                hr_values = self.context[0].getValues(3, 0, 10)
                if 111 in hr_values:
                    logger.info("Modbus监控: 检测到寄存器值为 111，准备退出...")
                    self.store.setValues(3, 0, [0] * 10) 
                    self.exit_event.set()
                    break
            except Exception as e: logger.error(f"Modbus 监控错误: {e}"); await asyncio.sleep(5)

    async def run_modbus_server(self):
        logger.info(f"Modbus 从机正在 {self.modbus_ip}:{self.modbus_port} 上监听...")
        try:
            server_task = asyncio.create_task(StartAsyncTcpServer(self.context, identity=ModbusDeviceIdentification(vendor_name="Custom"), address=(self.modbus_ip, self.modbus_port)))
            monitor_task = asyncio.create_task(self.request_monitor())
            done, pending = await asyncio.wait([server_task, monitor_task], return_when=asyncio.FIRST_COMPLETED)
            for task in pending: task.cancel()
            await asyncio.gather(*pending, return_exceptions=True) # Wait for cancellations
        except asyncio.CancelledError: logger.info("Modbus服务器或监控任务被取消")
        except Exception as e: logger.error(f"Modbus服务器运行失败: {e}")

    def start_wake_detection(self):
        try:
            login_params = f"appid={self.xf_app_id},engine_start=ivw"
            login_ret = self.dll.MSPLogin(None, None, login_params.encode('utf8'))
            if login_ret != 0:
                logger.error(f"MSPLogin fail, error code: {login_ret}")
                return
            logger.info(f"MSPLogin success.")
            
            err_code = c_int64()
            begin_params = f"sst=wakeup,ivw_threshold={self.ivw_threshold},ivw_res_path={self.work_dir}"
            self.dll.QIVWSessionBegin.restype = c_char_p
            session_id_ptr = self.dll.QIVWSessionBegin(None, begin_params.encode('utf8'), byref(err_code))
            if err_code.value != 0:
                logger.error(f"QIVWSessionBegin fail, error code: {err_code.value}")
                self.dll.MSPLogout() # Logout if session begin fails
                return
            session_id = string_at(session_id_ptr) # Get the actual session ID string
            logger.info(f"QIVWSessionBegin success, session_id: {session_id.decode('utf-8') if session_id else 'N/A'}")
            
            self.dll.QIVWRegisterNotify.argtypes = [c_char_p, c_void_p, c_void_p]
            reg_ret = self.dll.QIVWRegisterNotify(session_id, self.callback, None)
            if reg_ret != 0:
                logger.error(f"QIVWRegisterNotify fail, error code: {reg_ret}")
                self.dll.QIVWSessionEnd(session_id, "notify register fail".encode('utf8'))
                self.dll.MSPLogout()
                return
            logger.info("QIVWRegisterNotify success.")

            pa = pyaudio.PyAudio()
            stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            logger.info("开始监听唤醒词 (PyAudio stream opened)...")
            audio_frames_written = 0
            while not self.is_awake and not self.exit_event.is_set():
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                write_ret = self.dll.QIVWAudioWrite(session_id, audio_data, len(audio_data), 2) # 2: MSP_AUDIO_SAMPLE_CONTINUE
                if write_ret != 0:
                    logger.error(f"QIVWAudioWrite fail, error code: {write_ret}. Stopping wake detection for this cycle.")
                    break 
                audio_frames_written += 1
                if audio_frames_written % 100 == 0: # Log every 100 frames (approx every 6.4 seconds if CHUNK=1024, RATE=16000)
                    logger.debug(f"Still listening for wake word, frames written: {audio_frames_written}")
                time.sleep(0.01) # Avoid overly tight loop, but ensure responsiveness
            
            stream.stop_stream(); stream.close(); pa.terminate()
            logger.info("PyAudio stream closed.")
            
            end_ret = self.dll.QIVWSessionEnd(session_id, "normal end".encode('utf8'))
            if end_ret != 0: logger.error(f"QIVWSessionEnd fail, error code: {end_ret}")
            else: logger.info("QIVWSessionEnd success.")

            if self.is_awake and not self.exit_event.is_set():
                self.start_speech_recognition()
        except Exception as e:
            logger.error(f"唤醒检测主逻辑错误: {e}", exc_info=True)
        finally:
            logout_ret = self.dll.MSPLogout()
            if logout_ret != 0: logger.error(f"MSPLogout fail during cleanup, error code: {logout_ret}")
            else: logger.info("MSPLogout success during cleanup.")

    def start_speech_recognition(self):
        try:
            logger.info("开始语音识别...")
            self.text_buffer = ""
            websocket.enableTrace(False)
            wsUrl = self.create_asr_url()
            ws = websocket.WebSocketApp(wsUrl, on_message=self.on_asr_message, on_error=self.on_asr_error, on_close=self.on_asr_close, on_open=self.on_asr_open)
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}) # This is blocking
        except Exception as e: logger.error(f"语音识别启动错误: {e}")

    def create_asr_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now_dt = datetime.now(); date_str = format_date_time(mktime(now_dt.timetuple()))
        sig_origin = f"host: ws-api.xfyun.cn\ndate: {date_str}\nGET /v2/iat HTTP/1.1"
        sig_sha = base64.b64encode(hmac.new(XF_APISecret.encode('utf-8'), sig_origin.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
        auth_origin = f'api_key="{XF_APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{sig_sha}"'
        auth = base64.b64encode(auth_origin.encode('utf-8')).decode('utf-8')
        return f"{url}?authorization={auth}&date={date_str.replace(' ', '%20')}&host=ws-api.xfyun.cn"

    def on_asr_message(self, ws, message):
        try:
            msg_json = json.loads(message)
            if msg_json["code"] != 0: logger.error(f"ASR错误: {msg_json}"); ws.close(); return
            result = "".join(w["w"] for i in msg_json["data"]["result"]["ws"] for w in i["cw"])
            self.text_buffer += result
            # logger.debug(f"ASR partial: {result}") # Log partial results if needed
        except Exception as e: logger.error(f"ASR消息处理错误: {e}")

    def on_asr_error(self, ws, error): logger.error(f"ASR WebSocket错误: {error}")

    def on_asr_close(self, ws, close_status_code, close_msg):
        logger.info("ASR WebSocket已关闭.")
        recognized_text = self.text_buffer.strip()
        self.text_buffer = "" # Clear buffer for next time
        
        if recognized_text:
            logger.info(f"最终识别文本: {recognized_text}")
            self.expressions_server.send("text_person", recognized_text)
            
            local_command_executed = False
            try:
                local_command_executed = self.process_local_command(recognized_text)
            except Exception as e:
                logger.error(f"执行本地指令时出错: {e}")
            
            if not local_command_executed and not self.exit_event.is_set():
                logger.info("未匹配本地指令，转交通用对话模型...")
                asyncio.run_coroutine_threadsafe(self.handle_general_conversation(recognized_text), self.main_event_loop)
            # If local_command_executed is True, LLM is skipped.
        
        self.is_awake = False # Reset for next wake-up
        logger.info("已重置唤醒状态，等待下一次唤醒...")

    def on_asr_open(self, ws):
        def run(*args):
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            logger.info("* 开始录音并发送至ASR...")
            try:
                for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
                    if self.exit_event.is_set(): break
                    frame = stream.read(CHUNK, exception_on_overflow=False)
                    status = 1 if 0 < i < int(RATE / CHUNK * RECORD_SECONDS) - 1 else (0 if i == 0 else 2)
                    payload = {"common":{"app_id":self.xf_app_id}, "business":{"language":"zh_cn","domain":"iat","accent":"mandarin"}, "data":{"status":status,"format":"audio/L16;rate=16000","audio":base64.b64encode(frame).decode('utf-8'),"encoding":"raw"}}
                    ws.send(json.dumps(payload))
                    time.sleep(0.04) # As recommended by Xunfei
            except Exception as e: logger.error(f"ASR录音发送线程错误: {e}")
            finally:
                stream.stop_stream(); stream.close(); audio.terminate()
                if not self.exit_event.is_set() and ws.sock and ws.sock.connected: ws.close() # Ensure ws is closed if not already
                logger.info("* ASR录音发送结束.")
        thread.start_new_thread(run, ())

    def process_local_command(self, text: str) -> bool:
        self.expressions_server.send("text_person", text) # Show what user said
        
        action_keywords = ['放', '拿', '抓', '到', '进', '入']
        stop_keyword = '停'
        
        found_action = next((kw for kw in action_keywords if kw in text), None)

        if not found_action:
            if stop_keyword in text:
                logger.info(f"检测到停止指令: '{text}'")
                self.store.setValues(3, 1, [405])
                self.expressions_server.send("img", "happy"); self.expressions_server.send("text_robot", "收到，停止操作。")
                text_to_speech("好的，停止执行操作。")
                return True # Successfully handled stop command
            logger.info(f"文本 '{text}' 未包含已知动作或停止关键词，非本地指令。")
            return False # Not a local command

        parts = text.split(found_action)
        if len(parts) < 2:
            logger.warning(f"指令 '{text}' 格式不完整 (基于动作 '{found_action}')")
            return False # Incomplete command

        # Check for "拿出盒子" type commands which are unsupported.
        if '盒子' in parts[0] and found_action != stop_keyword : # Assuming stop keyword cannot be part of "拿出"
             logger.info(f"检测到不支持的指令组合 (如 '从盒子里拿出...'): {text}")
             self.store.setValues(3, 1, [406]) # Unsupported function
             self.expressions_server.send("img", "sad"); self.expressions_server.send("text_robot", "抱歉，这个操作我还不会。")
             text_to_speech("抱歉，我暂时没有这个功能。请重新唤醒我。")
             return True # Explicitly handled as unsupported, so LLM won't be called.

        front_tags = jieba.analyse.extract_tags(parts[0], topK=5, withWeight=True)
        back_tags = jieba.analyse.extract_tags(parts[1], topK=5, withWeight=True)

        matched_front = self.match_keywords_robust(front_tags, ['item1', 'item2'])
        matched_back = self.match_keywords_robust(back_tags, ['item3', 'item4'])

        command_map = {
            ('item1', 'item3'): (401, "绿色方块", "方形盒子里"),
            ('item1', 'item4'): (402, "绿色方块", "圆形盒子里"),
            ('item2', 'item3'): (403, "红色圆形", "方形盒子里"),
            ('item2', 'item4'): (404, "红色圆形", "圆形盒子里"),
        }

        if (matched_front, matched_back) in command_map:
            code, item_desc, loc_desc = command_map[(matched_front, matched_back)]
            logger.info(f"匹配到本地指令: {item_desc} -> {loc_desc} (Code: {code})")
            self.store.setValues(3, 1, [code])
            tts_msg = f"好的，正在把{item_desc}放到{loc_desc}"
            self.expressions_server.send("img", "happy"); self.expressions_server.send("text_robot", tts_msg)
            text_to_speech(tts_msg)
            self.save_command_to_yaml(text, matched_front, matched_back, code) # Save successful command
            return True # Successfully handled
        
        logger.info(f"指令 '{text}' 未匹配到具体操作组合 (front: {matched_front}, back: {matched_back})，非本地指令。")
        return False # Not a fully matched local command, let LLM handle or give generic fallback

    def match_keywords_robust(self, extracted_keywords, target_item_names):
        if not self.keywords_dict or not extracted_keywords: return None
        
        best_match_item = None
        highest_score = 0.0

        for item_name_in_dict in target_item_names: # e.g., 'item1', 'item2'
            item_config = next((i for i in self.keywords_dict.get('items', []) if i.get('name') == item_name_in_dict), None)
            if not item_config: continue

            for keyword_in_text, weight_in_text in extracted_keywords:
                for term_in_dict in item_config.get('keywords', []):
                    similarity = difflib.SequenceMatcher(None, keyword_in_text, term_in_dict).ratio()
                    # Consider a weighted score, e.g., similarity * weight_in_text (if jieba weight is meaningful)
                    # For now, simple similarity > threshold
                    if similarity > 0.75: # Adjusted threshold slightly
                        # Simple approach: if a keyword matches strongly, consider it for that item
                        # More complex: sum scores for an item, or pick best keyword match for an item
                        current_score = similarity # Could be weight_in_text * similarity
                        if current_score > highest_score:
                            highest_score = current_score
                            best_match_item = item_name_in_dict
                            # logger.debug(f"Keyword match: '{keyword_in_text}' (text) with '{term_in_dict}' (dict) for item '{item_name_in_dict}', score: {current_score:.2f}")
        
        if best_match_item: logger.info(f"Best match for target items {target_item_names} is '{best_match_item}' with score {highest_score:.2f}")
        return best_match_item

    def save_command_to_yaml(self, original_text, item, location, modbus_code):
        # Similar to save_results_to_yaml, but for successfully executed commands
        try:
            data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_text': original_text,
                'matched_item_code': item,
                'matched_location_code': location,
                'modbus_signal': modbus_code,
                'status': 'executed_locally'
            }
            filename = f"executed_cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            filepath = os.path.join(self.keywords_dir, filename) # Save in keywords dir or a new one
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            logger.info(f"本地执行的指令已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存执行的指令时出错: {e}")

    async def handle_general_conversation(self, text: str):
        if not self.openai_client or not self.qwen_llm_config:
            logger.error("Qwen LLM 客户端未初始化或配置不完整，无法进行通用对话。")
            fallback_msg = "抱歉，我的大脑连接好像出了点问题。"
            text_to_speech(fallback_msg); self.expressions_server.send("text_robot", fallback_msg); self.expressions_server.send("img", "sad")
            return
        try:
            logger.info(f"向通用对话模型发送: {text}")
            self.expressions_server.send("img", "thinking")
            
            model_name = self.qwen_llm_config.get("model", "qwen-plus")
            temperature = self.qwen_llm_config.get("temperature", 0.7)
            max_tokens = self.qwen_llm_config.get("max_tokens", 150)
            enable_thinking = self.qwen_llm_config.get("enable_thinking", False)

            params = {"model":model_name, "messages":[{"role":"system","content":self.qwen_system_prompt}, {"role":"user","content":text}], "temperature":temperature, "max_tokens":max_tokens}
            if "enable_thinking" in self.qwen_llm_config: params["extra_body"] = {"enable_thinking": enable_thinking}

            chat_completion = await self.openai_client.chat.completions.create(**params)
            llm_response = chat_completion.choices[0].message.content.strip()
            
            logger.info(f"通用对话模型回复: {llm_response}")
            text_to_speech(llm_response) # TTS播报LLM的回复
            self.expressions_server.send("text_robot", llm_response)
            self.expressions_server.send("img", "talking")
        except Exception as e:
            logger.error(f"调用通用对话模型失败: {e}", exc_info=True)
            error_msg = "抱歉，我现在有点糊涂，稍后再试试吧。"
            text_to_speech(error_msg); self.expressions_server.send("text_robot", error_msg); self.expressions_server.send("img", "sad")

async def main():
    loop = asyncio.get_running_loop()
    detector = WakeWordJieba(main_event_loop=loop)
    executor = ThreadPoolExecutor(max_workers=3) # For blocking IVW/ASR calls
    modbus_task = None
    try:
        modbus_task = asyncio.create_task(detector.run_modbus_server())
        logger.info("主程序开始运行，等待唤醒...")
        while not detector.exit_event.is_set():
            # detector.start_wake_detection() contains blocking ASR call, run in executor
            await asyncio.get_event_loop().run_in_executor(executor, detector.start_wake_detection)
            if detector.exit_event.is_set(): logger.info("主循环检测到退出事件，准备停止..."); break
            # is_awake is reset in on_asr_close after processing
            await asyncio.sleep(0.1) # Yield control briefly
        logger.info("主循环已结束。")
    except KeyboardInterrupt: logger.info("程序被用户中断 (KeyboardInterrupt)")
    except Exception as e: logger.error(f"主程序发生未捕获异常: {e}", exc_info=True)
    finally:
        logger.info("开始清理资源...")
        detector.exit_event.set() # Signal all tasks to stop
        if modbus_task and not modbus_task.done():
            modbus_task.cancel()
            try: await modbus_task
            except asyncio.CancelledError: logger.info("Modbus任务已成功取消.")
        logger.info("正在关闭线程池...")
        executor.shutdown(wait=True)
        logger.info("线程池已关闭。程序退出。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e: # Catch-all for asyncio.run related issues
        logger.error(f"启动程序时发生严重错误: {e}", exc_info=True)