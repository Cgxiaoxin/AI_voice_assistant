import asyncio
import websockets
import socket
import threading
from typing import Set
from datetime import datetime
import json  # 导入json模块

'''表情/状态显示 WebSocket 服务器'''

class ExpressionServer:
    # def __init__(self, host="192.168.60.53", port=8888):
    # def __init__(self, host="192.168.1.125", port=15000):
    def __init__(self, host="0.0.0.0", port=15000):
        """
        初始化表情服务器
        :param host: 绑定地址（默认所有接口）
        :param port: 监听端口
        """
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.loop = asyncio.new_event_loop()
        self._running = False

        self.message_data = {
            "type": "",
            "content": ""
        }

        self.timer_task = None
        self.timer_inierval = 0.3  # 单位：秒

    def _run_server(self):
        """内部方法：启动异步事件循环"""
        asyncio.set_event_loop(self.loop)
        try:
            # 启动WebSocket服务器
            self.loop.run_until_complete(self._start_server())
            print("[日志] WebSocket服务器已启动，开始创建定时任务")
            # 定时任务
            self.loop.run_until_complete(self._send_periodically())
            # self.timer_task = self.loop.create_task(self._send_periodically())
            print("[日志] 定时任务已创建")
            self.loop.run_forever()
        except Exception as e:
            print(f"[错误] 事件循环运行出错: {e}")

    async def _start_server(self):
        """启动WebSocket服务器"""
        self.server = await websockets.serve(
            self._client_handler,
            self.host,
            self.port
        )

        # 打印绑定信息
        print(f"表情服务器已启动 | 地址: {self.host}:{self.port}")
        print(f"实际绑定IP: {self._get_actual_ip()}")
        self._running = True
        await self.server.wait_closed()

    async def _client_handler(self, websocket, path):
        """处理客户端连接"""
        client_ip, client_port = websocket.remote_address
        # print(f"[连接] 客户端 {client_ip}:{client_port} 已连接")

        try:
            self.clients.add(websocket)

            # 保持连接活跃
            async for _ in websocket:
                pass

        except websockets.exceptions.ConnectionClosed:
            print(f"[断开] 客户端 {client_ip}:{client_port} 已断开")
        finally:
            self.clients.remove(websocket)

    def start(self):
        """启动服务器（同步方法）"""
        if self._running:
            raise RuntimeError("服务器已在运行中")

        # 在独立线程中运行事件循环
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()

        # 等待服务器确认启动
        while not self._running:
            pass

    def send(self, message_type: str, content: str) -> bool:
        """
        发送消息（同步方法）
        :param message_type: 消息类型（例如'expression'或'text'）
        :param content: 消息内容
        :return: 是否发送成功
        """
        if not self._running:
            print("错误：服务器未启动")
            return False


        # 构造消息字典
        self.message_data = {
            "type": message_type,
            "content": content
        }
        # 转换为JSON字符串
        json_message = json.dumps(self.message_data, ensure_ascii=False)
        # 异步发送
        # 异步发送
        future = asyncio.run_coroutine_threadsafe(
            self._broadcast(json_message),  # 发送JSON字符串
            self.loop
        )
        # future = asyncio.run_coroutine_threadsafe(
        #     self._broadcast(expression),  # 直接发送表情键
        #     self.loop
        # )

        try:
            # 等待发送完成（超时3秒）
            result = future.result(timeout=3)
            print(f"[日志] 已发送表情 '{json_message}' 给 {result} 个客户端")
            return True
        except asyncio.TimeoutError:
            print("[日志] 发送超时")
            return False
        except Exception as e:
            print(f"[日志] 发送失败：{str(e)}")
            return False

    async def _broadcast(self, message: str) -> int:
        """内部广播方法"""
        if not self.clients:
            print("[日志] 警告：当前无客户端连接")
            return 0

        # 过滤有效连接
        valid_clients = [client for client in self.clients if client.open]

        if not valid_clients:
            print("[日志] 警告：所有客户端连接已关闭")
            return 0

        # 打印实际发送的信息
        print(f"[日志] 正在发送消息: {message}")

        # 并发发送
        tasks = [client.send(message) for client in valid_clients]
        await asyncio.gather(*tasks)
        return len(tasks)

    def _get_actual_ip(self) -> str:
        """获取实际绑定的IP地址"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "未知IP"

    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.close()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._running = False
            print("服务器已停止")

    async def _send_periodically(self):
        """定时任务：每1秒发送一次消息"""
        while True:
            await asyncio.sleep(self.timer_inierval)
            if self.message_data["type"] == "img":
                success = self.send("img", self.message_data["content"])
                if success:
                    print("[定时任务]消息'{content}'发送成功")
                else:
                    print("[定时任务]消息'{content}'发送失败")

if __name__ == "__main__":
    # 使用示例
    server = ExpressionServer()
    server.start()  # 启动服务器

    try:
        while True:
            # 从控制台输入发送表情
            expression = input("输入要发送的表情（happy/sad/blink）：").strip().lower()

            if expression in ['exit', 'quit']:
                break
            server.send("img",expression)

    finally:
        server.stop()
