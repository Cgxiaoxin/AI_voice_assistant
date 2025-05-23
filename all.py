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

# 科大讯飞配置
APPID = "74fa666f"
APIKey = ""
APISecret = ""

# 音频录制参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10


def setup_logger(logs_dir):
    """配置日志系统"""
    # 移除默认的日志处理器
    logger.remove()

    # 获取当前时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 配置正常信息日志
    info_log_path = os.path.join(logs_dir, f"info_{timestamp}.log")
    logger.add(
        info_log_path,
        rotation="500 MB",
        encoding="utf-8",
        enqueue=True,
        retention="10 days",
        level="INFO"
    )

    # 配置错误信息日志
    error_log_path = os.path.join(logs_dir, f"error_{timestamp}.log")
    logger.add(
        error_log_path,
        rotation="500 MB",
        encoding="utf-8",
        enqueue=True,
        retention="10 days",
        level="ERROR"
    )

    # 添加控制台输出
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>"
    )


class WakeWordJieba:
    def __init__(self):
        self.is_awake = False
        self.text_buffer = ""

        # Modbus服务器配置
        self.modbus_ip = "192.168.1.138"
        self.modbus_port = 599

        # 创建数据存储
        self.store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * 100),
            co=ModbusSequentialDataBlock(0, [0] * 100),
            hr=ModbusSequentialDataBlock(0, [0] * 100),
            ir=ModbusSequentialDataBlock(0, [0] * 100)
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)

        # 创建必要的目录
        self.keywords_dir = "./keywords"
        self.logs_dir = "./logs"
        self.keywords_dict_dir = "./keywords_dicts"

        for directory in [self.keywords_dir, self.logs_dir, self.keywords_dict_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # 设置日志系统
        setup_logger(self.logs_dir)

        # 加载关键词词典
        self.keywords_dict = self.load_keywords_dict()

        # 初始化唤醒配置
        try:
            self.msc_load_library = './bin/msc_x64.dll'
            self.app_id = APPID
            self.ivw_threshold = '0:1450'
            self.jet_path = os.path.join(os.getcwd(), './bin/msc/res/ivw/wakeupresource.jet')
            self.work_dir = 'fo|' + self.jet_path

            # 加载DLL
            self.dll = cdll.LoadLibrary(self.msc_load_library)

            # 设置回调函数
            self.CALLBACKFUNC = CFUNCTYPE(None, c_char_p, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p)
            print("oncall")
            self.callback = self.CALLBACKFUNC(self.wake_callback)

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

        # 创建退出事件
        self.exit_event = asyncio.Event()

    def load_keywords_dict(self):
        """加载关键词词典"""
        try:
            dict_path = os.path.join(self.keywords_dict_dir, 'keywords_dict.yaml')
            if not os.path.exists(dict_path):
                logger.error(f"关键词词典文件不存在: {dict_path}")
                return None

            with open(dict_path, 'r', encoding='utf-8') as f:
                keywords_data = yaml.safe_load(f)
                logger.info(f"成功加载关键词词典: {dict_path}")
                return keywords_data
        except Exception as e:
            logger.error(f"加载关键词词典失败: {e}")
            return None

    def compare_keywords(self, extracted_keywords):
        """比较提取的关键词与词典中的关键词"""
        if not self.keywords_dict:
            return False

        matched_items = {}

        for keyword, weight in extracted_keywords:
            for item in self.keywords_dict['items']:
                for term in item['keywords']:
                    similarity = difflib.SequenceMatcher(None, keyword, term).ratio()
                    if similarity > 0.8:
                        logger.info(f"找到匹配关键词: {keyword} 匹配 {term}")
                        if item['name'] not in matched_items:
                            matched_items[item['name']] = 0
                        matched_items[item['name']] += weight

        if matched_items:
            selected_item = max(matched_items, key=matched_items.get)
            logger.info(f"选择的物品: {selected_item}，权重: {matched_items[selected_item]}")

            if selected_item == 'item1':
                self.store.setValues(3, 1, [401])  # 更新保持寄存器
            elif selected_item == 'item2':
                self.store.setValues(3, 1, [402])  # 更新保持寄存器
            return True

        return False

    def wake_callback(self, sessionID, msg, param1, param2, info, userDate):
        """唤醒回调函数"""
        logger.info("=" * 50)
        logger.info(">>> 检测到唤醒词！请说出您的指令 <<<")
        text_to_speech("你好，需要我为你做什么")
        logger.info("=" * 50)
        self.is_awake = True

    async def request_monitor(self):
        """监控请求并打印数据"""
        while not self.exit_event.is_set():
            await asyncio.sleep(1)
            hr_values = self.context[0].getValues(3, 0, 10)
            logger.info(f"当前保持寄存器数据: {hr_values}")
            if 111 in hr_values:
                logger.info(f"保持寄存器读到111")
                logger.info(f"当前保持寄存器数据: {hr_values}")
                self.store.setValues(3, 0, [0])
                logger.info(f"开始清零")
                logger.info(f"当前保持寄存器数据: {hr_values}")
                self.store.setValues(3, 0, [0])
                logger.info(f"当前保持寄存器数据: {hr_values}")
                logger.info("检测到寄存器值为 111，准备退出...")
                self.exit_event.set()  # 设置退出事件
                break

    async def run_modbus_server(self):
        """运行Modbus服务器"""
        logger.info(f"Modbus 从机正在 {self.modbus_ip}:{self.modbus_port} 上监听...")
        try:
            server_task = asyncio.create_task(StartAsyncTcpServer(self.context, identity=self.identity(), address=(self.modbus_ip, self.modbus_port)))
            monitor_task = asyncio.create_task(self.request_monitor())

            done, pending = await asyncio.wait(
                [server_task, monitor_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()  # 取消未完成的任务
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info("任务已取消")
        except asyncio.CancelledError:
            logger.info("Modbus服务器任务被取消")

    def identity(self):
        """设备标识"""
        identity = ModbusDeviceIdentification()
        identity.VendorName = "Custom"
        identity.ProductCode = "ModbusSlave"
        identity.VendorUrl = "http://example.com"
        identity.ProductName = "Modbus Server"
        identity.ModelName = "Modbus TCP"
        identity.MajorMinorRevision = "1.0"
        return identity

    def start_wake_detection(self):
        """启动唤醒检测"""
        try:
            # MSP登录
            login_params = f"appid={self.app_id},engine_start=ivw"
            ret = self.dll.MSPLogin(None, None, login_params.encode('utf8'))
            if ret != 0:
                logger.error(f"MSPLogin failed, error code: {ret}")
                return

            # 开始会话
            error_code = c_int64()
            begin_params = f"sst=wakeup,ivw_threshold={self.ivw_threshold},ivw_res_path={self.work_dir}"
            self.dll.QIVWSessionBegin.restype = c_char_p
            session_id = self.dll.QIVWSessionBegin(None, begin_params.encode('utf8'), byref(error_code))

            if error_code.value != 0:
                logger.error(f"QIVWSessionBegin failed, error code: {error_code.value}")
                return

            # 注册回调
            self.dll.QIVWRegisterNotify.argtypes = [c_char_p, c_void_p, c_void_p]
            ret = self.dll.QIVWRegisterNotify(session_id, self.callback, None)
            if ret != 0:
                logger.error(f"QIVWRegisterNotify failed, error code: {ret}")
                return

            # 开始录音
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

            logger.info("开始监听唤醒词...")

            while not self.is_awake:
                audio_data = stream.read(1024)
                audio_len = len(audio_data)
                ret = self.dll.QIVWAudioWrite(session_id, audio_data, audio_len, 2)
                if ret != 0:
                    break

            # 清理资源
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self.dll.QIVWSessionEnd(session_id, "normal end".encode('utf8'))

            # 如果被唤醒，开始语音识别
            if self.is_awake:
                self.start_speech_recognition()

        except Exception as e:
            logger.error(f"唤醒检测错误: {e}")
        finally:
            self.dll.MSPLogout()

    def start_speech_recognition(self):
        """启动语音识别"""
        try:
            logger.info("开始语音识别...")
            self.text_buffer = ""

            websocket.enableTrace(False)
            wsUrl = self.create_url()
            ws = websocket.WebSocketApp(
                wsUrl,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        except Exception as e:
            logger.error(f"语音识别错误: {e}")

    def create_url(self):
        """创建科大讯飞WebSocket连接URL"""
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"

        signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        return url + "?" + urlencode(params)

    def extract_keywords(self, text, topK=5):
        """使用jieba提取关键词"""
        try:
            keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=True)
            return keywords
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return None

    def save_results_to_yaml(self, text, keywords):
        """保存识别结果和关键词到YAML文件"""
        try:
            data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_text': text,
                'keywords': [{'word': word, 'weight': float(weight)} for word, weight in keywords]
            }

            # 修改保存路径到keywords目录
            filename = f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            filepath = os.path.join(self.keywords_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)

            logger.info(f"\n关键词已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存结果时出错: {e}")
            return False

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            message = json.loads(message)
            code = message["code"]
            if code != 0:
                logger.error(f"错误码：{code}，错误信息：{message}")
                ws.close()
            else:
                data = message["data"]["result"]["ws"]
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                logger.info(result)
                self.text_buffer += result

        except Exception as e:
            logger.error(f"接收消息错误：{e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket错误: {error}")

    def process_text_and_send_signal(self, text):
        """处理识别的文本并发送Modbus信号"""
        try:
            # 以"到"为节点拆分句子
            parts = text.split('放')
            if len(parts) != 2:
                logger.info("未检测到{放}指令，进行停止检索")
                all_text = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
                matched_text = self.match_keywords(all_text, ['item5'])
                if matched_text == 'item5':
                    logger.info("检测到停止指令，发送信号: 405")
                    self.store.setValues(3, 1, [405])
                    text_to_speech("好的开始执行停止指令")
                    self.is_awake = False
                    return
                else:
                    self.store.setValues(3, 1, [406])
                    logger.info("语音匹配检索错误，发送信号: 406")
                    text_to_speech("语音识别错误，请重新唤醒我")
                    self.is_awake = False
                    return



            # 前半部分分词并匹配
            print("正在拆分前面句子")
            front_keywords = jieba.analyse.extract_tags(parts[0], topK=5, withWeight=True)
            matched_front = self.match_keywords(front_keywords, ['item1', 'item2'])
            print("正在拆分后面句子")
            # 后半部分分词并匹配
            back_keywords = jieba.analyse.extract_tags(parts[1], topK=5, withWeight=True)
            matched_back = self.match_keywords(back_keywords, ['item3', 'item4'])

            # 根据匹配结果发送信号
            if matched_front == 'item1' and matched_back == 'item3':
                self.store.setValues(3, 1, [401])
                logger.info("发送信号: 401")
                text_to_speech("好的开始执行停止指令")
            elif matched_front == 'item1' and matched_back == 'item4':
                self.store.setValues(3, 1, [402])
                logger.info("发送信号: 402")
                text_to_speech("好的开始执行停止指令")
            elif matched_front == 'item2' and matched_back == 'item3':
                self.store.setValues(3, 1, [403])
                logger.info("发送信号: 403")
                text_to_speech("好的开始执行停止指令")
            elif matched_front == 'item2' and matched_back == 'item4':
                self.store.setValues(3, 1, [404])
                logger.info("发送信号: 404")
                text_to_speech("好的开始执行停止指令")
            else:
                logger.info("未找到匹配的组合")
                self.store.setValues(3, 1, [406])
                logger.info("发送信号: 406")
                text_to_speech("语音识别错误，请重新唤醒我")
                self.is_awake=False
                return
        except Exception as e:
            logger.error(f"处理文本时出错: {e}")
            self.is_awake=False

    def match_keywords(self, extracted_keywords, items):
        """匹配提取的关键词与指定的物品"""
        if not self.keywords_dict:
            return None

        for keyword, weight in extracted_keywords:
            for item in self.keywords_dict['items']:
                if item['name'] in items:
                    for term in item['keywords']:
                        similarity = difflib.SequenceMatcher(None, keyword, term).ratio()
                        if similarity > 0.8:
                            logger.info(f"找到匹配关键词: {keyword} 匹配 {term}")
                            return item['name']
        return None

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭事件"""
        logger.info("\n### 语音识别结束 ###")
        if self.text_buffer:
            logger.info("\n正在分析关键词...")
            # 调用新的文本处理和信号发送方法
            self.process_text_and_send_signal(self.text_buffer)

        self.text_buffer = ""
        self.is_awake = False  # 重置唤醒状态
        logger.info("等待唤醒...")

    def on_open(self, ws):
        """处理WebSocket连接打开事件"""

        def run(*args):
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            logger.info("* 开始录音...")

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                data = base64.b64encode(data).decode('utf-8')

                d = {
                    "common": {"app_id": APPID},
                    "business": {
                        "language": "zh_cn",
                        "domain": "iat",
                        "accent": "mandarin",
                    },
                    "data": {
                        "status": 1,
                        "format": "audio/L16;rate=16000",
                        "audio": data,
                        "encoding": "raw"
                    }
                }

                if i == 0:
                    d["data"]["status"] = 0
                elif i == int(RATE / CHUNK * RECORD_SECONDS) - 1:
                    d["data"]["status"] = 2

                ws.send(json.dumps(d))
                time.sleep(0.04)

            stream.stop_stream()
            stream.close()
            audio.terminate()
            ws.close()
            logger.info("* 录音结束")

        thread.start_new_thread(run, ())


async def main():
    detector = WakeWordJieba()
    executor = ThreadPoolExecutor(max_workers=2)  # 创建一个线程池

    try:
        await detector.run_modbus_server()
        while True:
            # 启动唤醒检测
            await asyncio.get_event_loop().run_in_executor(executor, detector.start_wake_detection)

            # 如果被唤醒，启动语音识别
            if detector.is_awake:
                await asyncio.get_event_loop().run_in_executor(executor, detector.start_speech_recognition)

            # 运行Modbus服务器
            # await detector.run_modbus_server()

            # 重置唤醒状态
            detector.is_awake = False
            logger.info("继续监听唤醒词...")
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("程序已停止")
        detector.exit_event.set()  # 手动设置退出事件
    finally:
        executor.shutdown(wait=True)  # 关闭线程池


if __name__ == "__main__":
    asyncio.run(main())