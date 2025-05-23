# Intelligent Voice Assistant Research Platform (智能语音助手研究平台)

## 项目简介 (Project Overview)

本项目旨在构建一个集成多模态感知与交互能力的智能语音助手原型系统。核心功能涵盖离线语音唤醒、在线自动语音识别 (ASR)、基于大型语言模型 (LLM) 的自然语言理解与生成，以及与外部设备进行状态同步与控制的能力。项目设计考虑了实时音频流处理、多线程/异步并发以及模块化组件集成，为进一步深入研究和开发智能语音交互应用提供了基础平台。

## 功能与技术亮点 (Features and Technical Highlights)

*   **混合式语音唤醒与指令识别 (Hybrid IVW & ASR)**:
    *   利用科大讯飞的离线唤醒引擎 (IVW) 在设备端低功耗监听特定唤醒词。
    *   唤醒后，通过 WebSocket 连接科大讯飞的在线 ASR 服务，实现高效、准确的用户语音指令转写。
    *   实现了从 IVW 状态到 ASR 状态的流畅切换机制。
*   **实时音频流处理 (Real-time Audio Stream Processing)**:
    *   使用 PyAudio 库进行跨平台的音频输入捕获。
    *   在独立的线程中处理音频数据，并将其发送给 IVW 引擎或 ASR WebSocket。
    *   包含基础的静音检测 (VAD) 逻辑，用于判断用户语音的结束。
*   **大型语言模型 (LLM) 集成 (LLM Integration)**:
    *   通过兼容 OpenAI API 的接口（配置为调用通义千问 Qwen），将用户识别到的文本指令发送给 LLM。
    *   利用 LLM 的强大能力进行复杂的自然语言理解、意图识别和生成富有逻辑和语义的回复。
    *   支持通过配置文件灵活切换 LLM 模型和调整参数 (如 temperature, max_tokens)。
*   **文本转语音 (TTS) 输出 (Text-to-Speech Output)**:
    *   将 LLM 生成的文本回复或系统提示文本通过 TTS 服务合成语音进行播放，实现自然的语音反馈。
*   **并发与状态管理 (Concurrency and State Management)**:
    *   采用了多线程和 asyncio 协程结合的并发模型：
        *   阻塞式的本地 SDK 调用 (如讯飞 IVW AudioWrite) 和 WebSocket 的 `run_forever` 运行在独立的线程中。
        *   异步网络请求 (如 LLM API 调用) 和可能的异步服务 (如 Modbus TCP Server) 使用 asyncio 事件循环管理。
        *   通过事件 (Event) 和线程安全的方式 (如 `asyncio.run_coroutine_threadsafe`) 在同步线程和异步事件循环之间进行通信和任务调度。
    *   设计了清晰的状态标志 (如 `is_awake`, `is_listening`, `is_playing`) 来管理系统的运行流程。
*   **外部状态与控制接口 (External State & Control Interface)**:
    *   **表情服务器 (Expression Server)**: 通过 WebSocket 与一个外部表情/状态服务器通信，实时更新助手的可视化状态或发送表情指令。
    *   **Modbus TCP 从机 (Modbus TCP Slave)**: 集成了 Modbus TCP 服务器功能 (基于 Pymodbus)，暴露寄存器接口，允许外部 Modbus 主机读取或写入数据，实现与工业设备或其他系统的简单交互。
*   **中断机制 (Interruption Mechanism)**:
    *   在 TTS 播放过程中，通过再次检测到唤醒词，可以中断当前的语音输出，允许用户随时打断并发出新的指令。

## 架构概览 (Architecture Overview)

系统采用了一种混合并发架构，以应对不同组件的特性：

1.  **主线程**: 运行 asyncio 事件循环，负责调度异步任务 (如 LLM 调用) 和管理线程池。
2.  **唤醒线程**: 运行在单独的守护线程中，持续监听音频流并调用讯飞 IVW SDK 进行唤醒词检测。检测到唤醒词后，通过设置事件通知主流程。
3.  **ASR 线程**: 运行在单独的守护线程中，建立 WebSocket 连接到讯飞 ASR 服务，并将捕获的音频数据发送进行实时识别。通过回调函数处理识别结果。
4.  **音频捕获**: PyAudio 在唤醒线程和 ASR 线程中分别负责音频流的输入。
5.  **LLM 异步任务**: 在主线程的 asyncio 事件循环中执行 LLM API 调用。
6.  **TTS**: 通过 `request.py` 中的同步函数进行调用（如果其内部实现是同步阻塞的，它可能也需要运行在独立的线程或进程中，尽管当前 VoiceAssistant 类直接调用了它）。
7.  **ExpressionServer**: 作为 WebSocket 服务器，运行在单独的线程中，处理来自 GUI 的连接和消息。VoiceAssistant 通过线程安全的方式向其发送消息。
8.  **ModbusServer**: 作为 Modbus TCP 从机，可以配置运行在 asyncio 事件循环中，与 Modbus 主机通信。

![Architecture Diagram Placeholder](link_to_your_architecture_diagram)
*(您可以在这里添加一个系统架构图，例如使用 Mermaid 或 PlantUML 语法，或者链接到外部图片。)*

## 依赖项 (Dependencies)

### Python 包 (Python Packages)

本项目依赖以下 Python 库，您可以通过 `pip` 进行安装：

```bash
pip install -r requirements.txt
```

### 外部 SDK/库 (External SDKs/Libraries)

*   **科大讯飞离线唤醒库 (iFlytek MSC Library)**:
    *   `msc_x64.dll` (或适用于您系统架构的版本，如 `msc_x86.dll`)
    *   唤醒词资源文件 `wakeupresource.jet`
    *   需要从科大讯飞开放平台下载对应的 SDK，并按照安装说明放置到指定目录。

## 安装与部署 (Setup and Installation)

### 1. 环境准备 (Environment Setup)

*   安装 Python 3.6+。
*   安装 PyAudio 所需的系统依赖 (例如 PortAudio)。具体方法取决于您的操作系统。
*   获取科大讯飞和 LLM 服务的 API 凭证。

### 2. 获取代码 (Get the Code)

克隆本项目仓库：

```bash
git clone <您的仓库地址>
cd <项目目录>
```

### 3. 创建并激活虚拟环境 (Create and Activate Virtual Environment)

推荐使用虚拟环境管理项目依赖：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 4. 安装 Python 依赖 (Install Python Dependencies)

```bash
pip install -r requirements.txt
```

### 5. 配置 API 凭证 (Configure API Credentials)

*   创建 `config` 文件夹（如果不存在）。
*   在 `config` 文件夹中创建 `key.yaml` 文件。
*   编辑 `key.yaml`，填入您的 API 密钥和其他配置信息。参考 [配置 (Configuration)](#配置-configuration) 部分的说明。

### 6. 放置讯飞 SDK 文件 (Place iFlytek SDK Files)

*   在项目根目录创建 `bin` 文件夹（如果不存在）。
*   如果您的 SDK 需要 `msc` 子目录存放资源，也请创建。
*   将 `msc_x64.dll` (或对应文件) 放置到 `./bin/` 目录下。
*   将 `wakeupresource.jet` 放置到 `./bin/msc/res/ivw/` 目录下（请根据实际 SDK 结构调整路径）。

### 7. (可选) 设置模拟服务 (Optional: Setup Mock Services)

如果您没有实际的外部 ExpressionServer 或不想连接实际的 TTS/Modbus，可以使用 `mock_services.py` 进行本地测试。确保 `mock_services.py` 文件存在，并且根据需要修改 `voice_test.py` 或使用独立的启动脚本 (如 `run_local.py`) 来确保在导入 `voice_test` 前替换了原始模块。

## 配置 (Configuration)

主要配置通过 `config/key.yaml` 文件进行：

```yaml
api_key:
  qwen:
    api_key: "YOUR_QWEN_API_KEY"  # 您的通义千问 API Key
    base_url: "YOUR_QWEN_BASE_URL" # 通义千问或兼容 OpenAI 的 Base URL
    model: "qwen-plus"            # 使用的 LLM 模型名称
    temperature: 0.7              # 控制 LLM 输出随机性，0-1 范围
    max_tokens: 150               # LLM 生成的最大 token 数
    enable_thinking: false        # 是否启用 LLM 思考过程 (取决于服务商支持)
    system_prompt: |              # 提供给 LLM 的系统提示词，定义助手角色和行为
      系统prompt：
      ##人设
      你是一个全能智能体...
      ##技能
      1. 当用户询问...
      ## token限制
      输出token不超过100

  # 如果您使用了其他兼容 OpenAI API 的服务，可以在此处添加配置
  # openai:
  #   api_key: "YOUR_OPENAI_API_KEY"
  #   base_url: "https://api.openai.com/v1"

# TTS 服务配置 (如果 request.py 中的 text_to_speech 函数需要 API Key)
# TTS_key:
#   api_key: "YOUR_TTS_API_KEY"
#   base_url: "YOUR_TTS_BASE_URL"
```

科大讯飞的 `XF_APPID`, `XF_APIKey`, `XF_APISecret` 硬编码在 `voice_test.py` 文件顶部。建议您根据需要将其移至 `key.yaml` 进行管理。

## 运行项目 (Running the Project)

在已激活虚拟环境并完成所有安装配置步骤后：

```bash
python voice_test.py
```

如果使用了模拟服务并创建了 `run_local.py` 启动脚本：

```bash
python run_local.py
```

程序启动后，日志将显示初始化过程。当看到 "开始监听唤醒词..." 的日志时，请说出您配置的唤醒词（默认为“小佰”）。成功唤醒后，程序将提示您说话，然后将您的语音发送给 ASR 和 LLM 进行处理。


