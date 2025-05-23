#coding=utf-8

'''
需要 Python 3.6 或更高版本

需要安装的包:
pip install asyncio
pip install websockets
'''

import asyncio
import websockets
import uuid
import json
import gzip
import copy

# 定义消息类型常量
MESSAGE_TYPES = {11: "仅音频服务器响应", 12: "前端服务器响应", 15: "服务器错误消息"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "无序列号", 1: "序列号 > 0",
                               2: "服务器最后一条消息 (序列号 < 0)", 3: "序列号 < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "无序列化", 1: "JSON", 15: "自定义类型"}
MESSAGE_COMPRESSIONS = {0: "无压缩", 1: "gzip", 15: "自定义压缩方法"}

# 配置参数
appid = "3724909676"
token = "4whn-QnCvsyAxZqnwByIe0TzxUIaySo8"
cluster = "volcano_tts"
voice_type = "Skye"
host = "openspeech.bytedance.com"
api_url = f"wss://{host}/api/v1/tts/ws_binary"

# 默认头部信息说明:
# version: b0001 (4位)
# header size: b0001 (4位)
# message type: b0001 (完整客户端请求) (4位)
# message type specific flags: b0000 (无) (4位)
# message serialization method: b0001 (JSON) (4位)
# message compression: b0001 (gzip) (4位)
# reserved data: 0x00 (1字节)
default_header = bytearray(b'\x11\x10\x11\x00')

# 请求JSON模板
request_json = {
    "app": {
        "appid": appid,
        "token": "access_token",
        "cluster": cluster
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": "Skye",
        "encoding": "wav",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": "xxx",
        "text": "字节跳动语音合成。",
        "text_type": "plain",
        "operation": "xxx"
    }
}


async def test_submit():
    """测试提交请求的函数"""
    submit_request_json = copy.deepcopy(request_json)
    submit_request_json["audio"]["voice_type"] = voice_type
    submit_request_json["request"]["reqid"] = str(uuid.uuid4())
    submit_request_json["request"]["operation"] = "submit"
    payload_bytes = str.encode(json.dumps(submit_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # 如果不需要压缩，注释此行
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # 负载大小(4字节)
    full_client_request.extend(payload_bytes)  # 负载数据
    print("\n------------------------ 测试 'submit' -------------------------")
    print("请求JSON: ", submit_request_json)
    print("\n请求字节: ", full_client_request)
    file_to_save = open("test_submit.wav", "wb")
    header = {"Authorization": f"Bearer; {token}"}
    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        while True:
            res = await ws.recv()
            done = parse_response(res, file_to_save)
            if done:
                file_to_save.close()
                break
        print("\n正在关闭连接...")


async def test_query():
    """测试查询请求的函数"""
    query_request_json = copy.deepcopy(request_json)
    query_request_json["audio"]["voice_type"] = voice_type
    query_request_json["request"]["reqid"] = str(uuid.uuid4())
    query_request_json["request"]["operation"] = "query"
    payload_bytes = str.encode(json.dumps(query_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # 如果不需要压缩，注释此行
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # 负载大小(4字节)
    full_client_request.extend(payload_bytes)  # 负载数据
    print("\n------------------------ 测试 'query' -------------------------")
    print("请求JSON: ", query_request_json)
    print("\n请求字节: ", full_client_request)
    file_to_save = open("test_query.wav", "wb")
    header = {"Authorization": f"Bearer; {token}"}
    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        res = await ws.recv()
        parse_response(res, file_to_save)
        file_to_save.close()
        print("\n正在关闭连接...")


def parse_response(res, file):
    """解析服务器响应的函数"""
    print("--------------------------- 响应信息 ---------------------------")
    # print(f"原始响应字节: {res}")
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size*4]
    payload = res[header_size*4:]
    print(f"            协议版本: {protocol_version:#x} - 版本 {protocol_version}")
    print(f"             头部大小: {header_size:#x} - {header_size * 4} 字节")
    print(f"            消息类型: {message_type:#x} - {MESSAGE_TYPES[message_type]}")
    print(f"     消息类型特定标志: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS[message_type_specific_flags]}")
    print(f"   消息序列化方法: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS[serialization_method]}")
    print(f"        消息压缩方式: {message_compression:#x} - {MESSAGE_COMPRESSIONS[message_compression]}")
    print(f"                保留位: {reserved:#04x}")
    if header_size != 1:
        print(f"          头部扩展: {header_extensions}")
    if message_type == 0xb:  # 仅音频服务器响应
        if message_type_specific_flags == 0:  # 无序列号作为确认
            print("                负载大小: 0")
            return False
        else:
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload = payload[8:]
            print(f"             序列号: {sequence_number}")
            print(f"                负载大小: {payload_size} 字节")
        file.write(payload)
        if sequence_number < 0:
            return True
        else:
            return False
    elif message_type == 0xf:
        code = int.from_bytes(payload[:4], "big", signed=False)
        msg_size = int.from_bytes(payload[4:8], "big", signed=False)
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        error_msg = str(error_msg, "utf-8")
        print(f"          错误消息代码: {code}")
        print(f"          错误消息大小: {msg_size} 字节")
        print(f"               错误消息: {error_msg}")
        return True
    elif message_type == 0xc:
        msg_size = int.from_bytes(payload[:4], "big", signed=False)
        payload = payload[4:]
        if message_compression == 1:
            payload = gzip.decompress(payload)
        print(f"            前端消息: {payload}")
    else:
        print("未定义的消息类型!")
        return True


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_submit())
    loop.run_until_complete(test_query())
