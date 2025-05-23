import websocket
import datetime
import hashlib
import base64
import hmac
import json
import ssl
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import time
import os
import threading
import pyaudio

'''
文本转语音模块
'''

# 科大讯飞配置
APPID = "cc42f548"
APIKey = ""
APISecret = "MTg3ZDc4OWViZDQwNmY3N2I3NjA4ZmMx"

def create_url():
    """创建科大讯飞WebSocket连接URL"""
    url = 'wss://tts-api.xfyun.cn/v2/tts'
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    signature_origin = "host: " + "tts-api.xfyun.cn" + "\n"
    signature_origin += "date: " + date + "\n"
    signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"

    signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                             hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

    authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

    # 调试：打印签名和授权信息
    print("Signature Origin:", signature_origin)
    print("Signature SHA:", signature_sha)
    print("Authorization Origin:", authorization_origin)
    print("Authorization:", authorization)

    params = {
        "authorization": authorization,
        "date": date,
        "host": "tts-api.xfyun.cn"
    }
    return url + "?" + urlencode(params)

def on_message(ws, message):
    """处理WebSocket消息"""
    try:
        message = json.loads(message)
        code = message["code"]
        if code != 0:
            print(f"错误码：{code}，错误信息：{message}")
            ws.close()
        else:
            audio = message["data"]["audio"]
            audio = base64.b64decode(audio)
            # 调试输出：检查接收到的音频数据长度
            print(f"接收到音频数据长度: {len(audio)}")
            # 累积音频数据
            ws.audio_data.append(audio)
    except Exception as e:
        print(f"接收消息错误：{e}")

def play_audio(audio_data):
    """播放音频数据"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True)
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

def on_error(ws, error):
    print(f"WebSocket错误: {error}")

def on_close(ws, close_status_code, close_msg):
    """WebSocket连接关闭"""
    print("WebSocket连接关闭")
    # 播放完整的音频数据
    complete_audio = b''.join(ws.audio_data)
    play_audio(complete_audio)

def on_open(ws, text):
    """处理WebSocket连接打开事件"""
    # xiaoyan: 女声
    # xiaoyu: 男声
    # xiaofeng: 男声
    # xiaomei: 女声
    # xiaolin: 女声
    # xiaorong: 女声
    def run(*args):
        d = {
            "common": {"app_id": APPID},
            "business": {
                "aue": "raw",
                "auf": "audio/L16;rate=16000",
                "vcn": "xiaoyan",
                "tte": "utf8"
            },
            "data": {
                "status": 2,
                "text": base64.b64encode(text.encode('utf-8')).decode('utf-8')
            }
        }
        ws.send(json.dumps(d))
        time.sleep(0.5)
        ws.close()

    threading.Thread(target=run).start()

def text_to_speech(text):
    """将文本转换为语音"""
    websocket.enableTrace(False)
    wsUrl = create_url()
    ws = websocket.WebSocketApp(
        wsUrl,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=lambda ws: on_open(ws, text)
    )
    ws.audio_data = []  # 初始化音频数据列表
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

if __name__ == "__main__":
    text_to_speech("好的开始执行指令")