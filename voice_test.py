import os
import asyncio
import threading
import time
from ctypes import cdll, byref, string_at, c_void_p, CFUNCTYPE, c_char_p, c_uint64, c_int64
import pyaudio
from loguru import logger
import websocket
import datetime
import hashlib
import base64
import hmac
import json
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import yaml
import difflib
import sys
from concurrent.futures import ThreadPoolExecutor
from request import text_to_speech
from change_state import ExpressionServer
from openai import AsyncOpenAI

# --- 配置常量 ---
CONFIG_PATH = "config/key.yaml"

# 科大讯飞配置
XF_APPID = "74fa666f"
XF_APIKey = "1117d16429339b497550a24efbacfc44"
XF_APISecret = "YjBkMzc3NWU3NjYwZmY2MWNjMWY2YTc0"

# 音频录制参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 6  # 默认录音时长
SILENCE_THRESHOLD = 1.0  # 静音检测阈值（秒）

# --- 辅助函数 ---
def setup_logger(logs_dir):
    """配置日志记录系统"""
    logger.remove()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    common_args = {"rotation": "500 MB", "encoding": "utf-8", "enqueue": True, "retention": "10 days"}
    logger.add(os.path.join(logs_dir, f"info_{timestamp}.log"), level="INFO", **common_args)
    logger.add(os.path.join(logs_dir, f"error_{timestamp}.log"), level="ERROR", **common_args)
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

def load_api_config(config_path=CONFIG_PATH, service_name='qwen'):
    """加载API配置"""
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

# --- 语音助手类 ---
class VoiceAssistant:
    def __init__(self, loop):
        # 初始化基础设置
        self.loop = loop
        self.logs_dir = "./logs"
        self.wake_word = "小佰"  # 初始唤醒词
        self.interrupt_wake_word = "小佰小佰"  # 中断唤醒词
        
        # 创建日志目录
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # 配置日志系统
        setup_logger(self.logs_dir)
        logger.info("语音助手初始化中...")
        
        # 初始化状态变量
        self.is_awake = False  # 是否被唤醒
        self.is_listening = False  # 是否正在监听用户指令
        self.is_playing = False  # 是否正在播放回复
        self.should_interrupt = False  # 是否需要中断播放
        self.text_buffer = ""  # ASR文本缓冲区
        
        # 初始化线程控制
        self.exit_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.listening_event = threading.Event()
        
        # 初始化表情服务器
        self.expressions_server = ExpressionServer()
        try:
            self.expressions_server.start()
            logger.info("表情服务器启动成功")
        except Exception as e:
            logger.error(f"表情服务器启动失败：{e}")
            logger.info("尝试使用本地模式启动...")
            # 如果使用的是模拟服务器，它已经配置为使用127.0.0.1
        
        # 初始化LLM配置
        self.qwen_llm_config = load_api_config(service_name='qwen')
        self.openai_client = None
        self.qwen_system_prompt = "You are a helpful assistant."  # 默认提示词
        
        if self.qwen_llm_config:
            if self.qwen_llm_config.get("api_key") and self.qwen_llm_config.get("base_url"):
                try:
                    self.openai_client = AsyncOpenAI(
                        api_key=self.qwen_llm_config.get("api_key"),
                        base_url=self.qwen_llm_config.get("base_url")
                    )
                    logger.info("Qwen LLM 客户端初始化成功。")
                    
                    # 加载系统提示词
                    loaded_prompt = self.qwen_llm_config.get("system_prompt")
                    if loaded_prompt:
                        self.qwen_system_prompt = loaded_prompt
                        logger.info("已从配置加载 Qwen system_prompt。")
                    else:
                        logger.warning("未在配置中找到 Qwen system_prompt，将使用默认提示。")
                except Exception as e:
                    logger.error(f"Qwen LLM 客户端初始化失败: {e}")
            else:
                logger.warning("Qwen LLM 的 api_key 或 base_url 未在配置中提供，LLM功能将受限。")
        else:
            logger.warning("Qwen LLM 配置未加载，通用对话功能将使用默认提示且可能无法工作。")
        
        # 初始化讯飞唤醒引擎
        try:
            self.msc_load_library = './bin/msc_x64.dll'
            self.xf_app_id = XF_APPID
            self.ivw_threshold = '0:1450'
            self.jet_path = os.path.join(os.getcwd(), './bin/msc/res/ivw/wakeupresource.jet')
            self.work_dir = 'fo|' + self.jet_path
            self.dll = cdll.LoadLibrary(self.msc_load_library)
            
            # 定义回调函数原型
            self.CALLBACKFUNC = CFUNCTYPE(None, c_char_p, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p)
            self.callback = self.CALLBACKFUNC(self.wake_callback)
            logger.info("讯飞唤醒引擎回调设置完毕。")
        except Exception as e:
            logger.error(f"讯飞唤醒引擎初始化失败: {e}")
            raise

        logger.info("语音助手初始化完成")
    
    def wake_callback(self, sessionID, msg, param1, param2, info, userDate):
        """唤醒词检测回调，触发唤醒状态和反馈"""
        logger.info("="*25 + " 检测到唤醒词! " + "="*25)
        self.is_awake = True
        
        # 如果正在播放，中断播放
        if self.is_playing:
            self.should_interrupt = True
            self.interrupt_event.set()
            logger.info("检测到中断唤醒词，停止当前播放")
            return
        
        # 否则是首次唤醒，播放唤醒提示
        text_to_speech("您好，我是A I语音助手小佰，需要我为您做什么？")  # 通过语音合成播放语音提示
        self.expressions_server.send("img", "blink")  # 发送眨眼动画指令
        self.expressions_server.send("text_robot", "您好，我是A I语音助手小佰，需要我为您做什么？")  # 在机器人界面上显示文字提示
        self.expressions_server.send("text_start", "start")
        self.listening_event.set()  # 触发监听用户指令
    
    def create_asr_url(self):
        """生成科大讯飞ASR WebSocket URL，包含签名参数"""
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now_dt = datetime.now()
        date_str = format_date_time(mktime(now_dt.timetuple()))
        
        # 构造签名原文
        sig_origin = f"host: ws-api.xfyun.cn\ndate: {date_str}\nGET /v2/iat HTTP/1.1"
        
        # 计算SHA256签名并Base64
        sig_sha = base64.b64encode(
            hmac.new(XF_APISecret.encode('utf-8'), sig_origin.encode('utf-8'), hashlib.sha256).digest()
        ).decode('utf-8')
        
        # 构造Authorization头并Base64
        auth_origin = f'api_key="{XF_APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{sig_sha}"'
        auth = base64.b64encode(auth_origin.encode('utf-8')).decode('utf-8')
        
        return f"{url}?authorization={auth}&date={date_str.replace(' ', '%20')}&host=ws-api.xfyun.cn"
    
    def on_asr_message(self, ws, message):
        """处理ASR消息，拼接识别结果"""
        try:
            msg_json = json.loads(message)
            if msg_json["code"] != 0:
                logger.error(f"ASR错误: {msg_json}")
                ws.close()
                return
            
            # 提取结果数据
            result = "".join(w["w"] for i in msg_json["data"]["result"]["ws"] for w in i["cw"])
            self.text_buffer += result
            
            # 检查是否是最后一条消息
            is_last = msg_json["data"].get("status") == 2
            if is_last:
                logger.info(f"ASR最终结果: {self.text_buffer}")
        except Exception as e:
            logger.error(f"ASR消息处理错误: {e}")
    
    def on_asr_error(self, ws, error):
        """处理ASR WebSocket错误"""
        logger.error(f"ASR WebSocket错误: {error}")
    
    def on_asr_close(self, ws, close_status_code=None, close_msg=None):
        """处理ASR WebSocket关闭连接"""
        logger.info("ASR WebSocket已关闭.")
        
        self.is_listening = False
        recognized_text = self.text_buffer.strip()
        self.text_buffer = ""  # 清空缓冲区
        
        if recognized_text:
            logger.info(f"最终识别文本: {recognized_text}")
            self.expressions_server.send("text_person", recognized_text)
            asyncio.run_coroutine_threadsafe(self.process_text(recognized_text), self.loop)
    
    def on_asr_open(self, ws):
        """处理ASR WebSocket连接打开"""
        logger.info("* 开始录音并发送至ASR...")
        self.is_listening = True
        
        def run(*args):
            silence_start = None
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            
            try:
                for i in range(int(RATE / CHUNK * RECORD_SECONDS * 2)):  # 增加时间以允许静音检测
                    if self.exit_event.is_set() or not self.is_listening:
                        break
                    
                    frame = stream.read(CHUNK, exception_on_overflow=False)
                    
                    # 简单的静音检测 (可以完善为更复杂的VAD)
                    is_silence = max(abs(int.from_bytes(frame[i:i+2], 'little', signed=True)) for i in range(0, len(frame), 2)) < 500
                    
                    if is_silence:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_THRESHOLD:
                            logger.info("检测到静音，停止录音")
                            break
                    else:
                        silence_start = None
                    
                    # 发送音频数据
                    status = 1 if 0 < i < int(RATE / CHUNK * RECORD_SECONDS) - 1 else (0 if i == 0 else 2)
                    payload = {
                        "common": {"app_id": self.xf_app_id},
                        "business": {"language": "zh_cn", "domain": "iat", "accent": "mandarin", "vad_eos": 3000},
                        "data": {
                            "status": status,
                            "format": "audio/L16;rate=16000",
                            "audio": base64.b64encode(frame).decode('utf-8'),
                            "encoding": "raw"
                        }
                    }
                    ws.send(json.dumps(payload))
                    time.sleep(0.04)  # 按讯飞推荐频率
                
                # 发送最后一帧
                payload = {
                    "common": {"app_id": self.xf_app_id},
                    "business": {"language": "zh_cn", "domain": "iat", "accent": "mandarin"},
                    "data": {"status": 2, "format": "audio/L16;rate=16000", "audio": "", "encoding": "raw"}
                }
                ws.send(json.dumps(payload))
                
            except Exception as e:
                logger.error(f"ASR录音发送线程错误: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                if not self.exit_event.is_set() and ws.sock and ws.sock.connected:
                    ws.close()  # 确保WebSocket已关闭
                logger.info("* ASR录音发送结束.")
        
        thread.start_new_thread(run, ())
    
    def listen_for_command(self):
        """监听用户命令"""
        try:
            logger.info("开始语音识别...")
            websocket.enableTrace(False)
            wsUrl = self.create_asr_url()
            
            # 创建WebSocket并绑定回调
            ws = websocket.WebSocketApp(
                wsUrl,
                on_message=self.on_asr_message,
                on_error=self.on_asr_error,
                on_close=self.on_asr_close,
                on_open=self.on_asr_open
            )
            # ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 原来的阻塞调用

            # 在单独的线程中运行 ws.run_forever()
            ws_thread = threading.Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
            ws_thread.daemon = True  # 设置为守护线程，以便主程序退出时它也退出
            ws_thread.start()

            # 等待 ASR 完成 (is_listening 变为 False) 或退出事件被设置
            while self.is_listening and not self.exit_event.is_set():
                time.sleep(0.1)

            if self.exit_event.is_set():
                logger.info("检测到退出事件，正在关闭 ASR WebSocket...")
                if ws.sock and ws.sock.connected:
                    ws.close()
            
            # 等待WebSocket线程结束
            ws_thread.join(timeout=2.0) # 等待线程优雅退出

        except Exception as e:
            logger.error(f"语音识别启动错误: {e}")
            self.is_listening = False # 确保状态被重置
    
    async def process_text(self, text):
        """处理识别的文本，发送给LLM并处理回复"""
        if not text:
            logger.warning("识别文本为空，不进行处理")
            self.return_to_standby()
            return
        
        # 检查LLM是否可用
        if not self.openai_client or not self.qwen_llm_config:
            logger.error("LLM客户端未初始化，无法处理文本")
            text_to_speech("抱歉，我的大脑连接好像出了点问题。")
            self.expressions_server.send("text_robot", "抱歉，我的大脑连接好像出了点问题。")
            self.expressions_server.send("img", "sad")
            self.return_to_standby()
            return
        
        try:
            # 通知界面切换到思考状态
            logger.info(f"向LLM发送文本: {text}")
            self.expressions_server.send("img", "thinking")
            
            # 准备LLM参数
            model_name = self.qwen_llm_config.get("model", "qwen-plus")
            temperature = self.qwen_llm_config.get("temperature", 0.7)
            max_tokens = self.qwen_llm_config.get("max_tokens", 150)
            enable_thinking = self.qwen_llm_config.get("enable_thinking", False)
            
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": self.qwen_system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if "enable_thinking" in self.qwen_llm_config:
                params["extra_body"] = {"enable_thinking": enable_thinking}
            
            # 调用LLM
            chat_completion = await self.openai_client.chat.completions.create(**params)
            llm_response = chat_completion.choices[0].message.content.strip()
            
            logger.info(f"LLM回复: {llm_response}")
            
            # 播放回复前，启动监听中断唤醒词的线程
            self.start_interrupt_listener()
            
            # 使用TTS播放回复
            self.is_playing = True
            text_to_speech(llm_response)
            self.expressions_server.send("text_robot", llm_response)
            self.expressions_server.send("img", "talking")
            
            # 等待播放完成或中断
            if not self.interrupt_event.is_set():
                logger.info("TTS播放完成")
            else:
                logger.info("TTS播放被中断")
                self.interrupt_event.clear()
            
            self.is_playing = False
            
        except Exception as e:
            logger.error(f"处理文本失败: {e}", exc_info=True)
            text_to_speech("抱歉，我现在有点糊涂，稍后再试试吧。")
            self.expressions_server.send("text_robot", "抱歉，我现在有点糊涂，稍后再试试吧。")
            self.expressions_server.send("img", "sad")
        finally:
            # 停止中断监听
            self.should_interrupt = False
            self.return_to_standby()
    
    def start_interrupt_listener(self):
        """启动监听中断唤醒词的线程"""
        logger.info("启动中断监听线程")
        
        # 这里简化实现，实际应调用唤醒词检测功能
        # 由于唤醒引擎已经在主循环中启动，这里不需要额外实现
        # 唤醒回调中会检查is_playing状态，如果是播放状态就会触发中断
        pass
    
    def return_to_standby(self):
        """返回待机状态，重置状态变量"""
        self.is_awake = False
        self.is_listening = False
        self.is_playing = False
        self.should_interrupt = False
        self.interrupt_event.clear()
        self.listening_event.clear()
        logger.info("已重置状态，返回待机")
    
    def start_wake_detection(self):
        """启动唤醒词检测"""
        try:
            # 登录MSP
            login_params = f"appid={self.xf_app_id},engine_start=ivw"
            login_ret = self.dll.MSPLogin(None, None, login_params.encode('utf8'))
            if login_ret != 0:
                logger.error(f"MSPLogin fail, error code: {login_ret}")
                return
            logger.info("MSPLogin success.")
            
            # 开始唤醒会话
            err_code = c_int64()
            begin_params = f"sst=wakeup,ivw_threshold={self.ivw_threshold},ivw_res_path={self.work_dir}"
            self.dll.QIVWSessionBegin.restype = c_char_p
            session_id_ptr = self.dll.QIVWSessionBegin(None, begin_params.encode('utf8'), byref(err_code))
            if err_code.value != 0:
                logger.error(f"QIVWSessionBegin fail, error code: {err_code.value}")
                self.dll.MSPLogout()
                return
            session_id = string_at(session_id_ptr)
            logger.info(f"QIVWSessionBegin success, session_id: {session_id.decode('utf-8') if session_id else 'N/A'}")
            
            # 注册回调
            self.dll.QIVWRegisterNotify.argtypes = [c_char_p, c_void_p, c_void_p]
            reg_ret = self.dll.QIVWRegisterNotify(session_id, self.callback, None)
            if reg_ret != 0:
                logger.error(f"QIVWRegisterNotify fail, error code: {reg_ret}")
                self.dll.QIVWSessionEnd(session_id, "notify register fail".encode('utf8'))
                self.dll.MSPLogout()
                return
            logger.info("QIVWRegisterNotify success.")
            
            # 打开音频流
            pa = pyaudio.PyAudio()
            stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            logger.info("开始监听唤醒词 (PyAudio stream opened)...")
            audio_frames_written = 0
            
            # 读取并写入音频数据到唤醒引擎
            while not self.exit_event.is_set():
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                write_ret = self.dll.QIVWAudioWrite(session_id, audio_data, len(audio_data), 2)
                if write_ret != 0:
                    logger.error(f"QIVWAudioWrite fail, error code: {write_ret}")
                    break
                
                audio_frames_written += 1
                if audio_frames_written % 100 == 0:
                    logger.debug(f"Still listening for wake word (is_awake: {self.is_awake}, is_playing: {self.is_playing}, listening_event: {self.listening_event.is_set()}), frames written: {audio_frames_written}")
                
                # 检查是否被唤醒
                if self.is_awake and not self.is_playing:
                    # 被唤醒且不是播放中（中断）状态，开始命令监听
                    if self.listening_event.is_set():
                        logger.info("唤醒成功，开始监听命令")
                        self.listening_event.clear()
                        self.listen_for_command()
                        # 监听结束后会自动返回待机状态 (通过 on_asr_close -> process_text -> return_to_standby)
                
                time.sleep(0.01)  # 控制循环频率
            
            # 关闭音频流
            stream.stop_stream()
            stream.close()
            pa.terminate()
            logger.info("PyAudio stream closed.")
            
            # 结束唤醒会话
            end_ret = self.dll.QIVWSessionEnd(session_id, "normal end".encode('utf8'))
            if end_ret != 0:
                logger.error(f"QIVWSessionEnd fail, error code: {end_ret}")
            else:
                logger.info("QIVWSessionEnd success.")
                
        except Exception as e:
            logger.error(f"唤醒检测主逻辑错误: {e}", exc_info=True)
        finally:
            logout_ret = self.dll.MSPLogout()
            if logout_ret != 0:
                logger.error(f"MSPLogout fail during cleanup, error code: {logout_ret}")
            else:
                logger.info("MSPLogout success during cleanup.")
    
    def run(self):
        """主程序入口"""
        try:
            logger.info("语音助手启动...")
            
            # 启动唤醒词检测
            wake_thread = threading.Thread(target=self.start_wake_detection)
            wake_thread.daemon = True
            wake_thread.start()
            
            # 主线程等待退出事件
            try:
                while not self.exit_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("检测到键盘中断，程序退出")
                self.exit_event.set()
            
            # 等待线程结束
            wake_thread.join(timeout=1.0)
            logger.info("语音助手已退出")
        except Exception as e:
            logger.error(f"运行主程序时出错: {e}", exc_info=True)

async def main():
    """异步主函数"""
    loop = asyncio.get_running_loop()
    assistant = VoiceAssistant(loop=loop)
    
    # 运行助手
    await loop.run_in_executor(None, assistant.run)

if __name__ == "__main__":
    try:
        asyncio.run(main()) 
    except Exception as e:
        logger.error(f"启动程序时发生严重错误: {e}", exc_info=True)
