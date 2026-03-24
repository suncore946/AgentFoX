import os
import shutil
import json
from pathlib import Path
import time
import gradio as gr

# ================= 设置本地临时目录 =================
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_temp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# ================= 默认展示数据路径 =================
DISPLAY_DIR = "/data2/yuyangxin/Agent/resources/display"
DISPLAY_IMG_PATH = os.path.join(DISPLAY_DIR, "image.png")
DISPLAY_JSON_PATH = os.path.join(DISPLAY_DIR, "content.json")

# ================= 全局配置与初始化 =================
DEFAULT_CONFIG_PATH = "forensic_agent/configs/config_qwen3_32b_benchmark.yaml"
EXAMPLE_IMAGE_DIR = "resources/image"
IS_DEBUG = False
MAX_WORKERS = 4
afa_app = None


def init_system():
    global afa_app
    print(f"⏳ 正在启动系统...")
    try:
        from agent_pipeline import AFAApplication

        app = AFAApplication(config_path=DEFAULT_CONFIG_PATH, is_debug=IS_DEBUG, max_workers=MAX_WORKERS)
        app.initialize()
        print("✅ 系统初始化完成。")
        return app
    except ImportError:
        print("❌ 错误: 未找到 agent_pipeline.py (仅测试UI模式)")
        return None
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return None


afa_app = init_system()


# ================= 辅助函数 =================
def get_example_gallery_data():
    path = Path(EXAMPLE_IMAGE_DIR)
    if not path.exists():
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for ext in extensions:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    files.sort(key=lambda f: f.name.lower())  # 按名称排序
    # 返回列表: [(path_str, filename_str), ...]
    return [(str(f), f.name) for f in files]


def format_chat_history(messages):
    history = []
    for msg in messages:
        # 兼容字典对象或类实例
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("type", "unknown")
        else:
            content = getattr(msg, "content", "")
            role = getattr(msg, "type", "unknown")

        if not content:
            continue

        # 强制转为字符串，防止None或对象导致报错
        content_str = str(content)

        # 映射到 Gradio 接受的角色: "user" 或 "assistant"
        if role == "human":
            history.append({"role": "user", "content": content_str})
        elif role == "ai":
            history.append({"role": "assistant", "content": content_str})
        elif role == "tool":
            # 工具输出通常作为 Assistant 的一种特殊回复展示
            history.append({"role": "assistant", "content": f"🛠️ **Tool Output:**\n{content_str[:300]}..."})
        else:
            history.append({"role": "assistant", "content": f"**[{role}]** {content_str}"})

    return history


# =================【新增】节点链路渲染逻辑 =================
def render_process_graph(messages):
    """
    将消息列表转换为 HTML 节点链路图。
    支持显示 reasoning_content (Chain of Thought)。
    """
    if not messages:
        return "<div style='padding:20px; text-align:center; color:#888;'>暂无分析过程数据</div>"

    html_parts = ['<div class="process-timeline">']

    for i, msg in enumerate(messages):
        # 兼容字典对象或类实例
        m_type = getattr(msg, "type", None) or msg.get("type", "unknown")
        m_content = getattr(msg, "content", None) or msg.get("content", "")
        m_kwargs = getattr(msg, "additional_kwargs", {}) or msg.get("additional_kwargs", {})
        m_name = getattr(msg, "name", None) or msg.get("name", "")

        # 忽略单纯的状态更新消息
        if "update stage to:" in str(m_content):
            continue

        # --- 1. HUMAN NODE ---
        if m_type == "human":
            html_parts.append(
                f"""
            <div class="process-node node-human">
                <div class="node-icon">👤</div>
                <div class="node-content">
                    <div class="node-title">USER REQUEST</div>
                    <div class="node-text">{m_content[:300]}...</div>
                </div>
            </div>
            <div class="process-link"></div>
            """
            )

        # --- 2. TOOL NODE ---
        elif m_type == "tool":
            tool_name = m_name if m_name else "Tool Execution"
            # 尝试格式化 JSON 内容以便阅读
            display_content = m_content
            try:
                parsed = json.loads(m_content)
                display_content = f"<pre>{json.dumps(parsed, indent=2, ensure_ascii=False)}</pre>"
            except:
                display_content = m_content[:500] + ("..." if len(m_content) > 500 else "")

            html_parts.append(
                f"""
            <div class="process-node node-tool">
                <div class="node-icon">🛠️</div>
                <div class="node-content">
                    <div class="node-title">TOOL OUTPUT: {tool_name}</div>
                    <div class="node-body">{display_content}</div>
                </div>
            </div>
            <div class="process-link"></div>
            """
            )

        # --- 3. AI NODE (包含 Reasoning 和 Content) ---
        elif m_type == "ai":
            # 3.1 思考过程 (Reasoning Content)
            reasoning = m_kwargs.get("reasoning_content", "")
            if reasoning:
                html_parts.append(
                    f"""
                <div class="process-node node-reasoning">
                    <div class="node-icon">🧠</div>
                    <div class="node-content">
                        <div class="node-title">CHAIN OF THOUGHT (REASONING)</div>
                        <div class="node-text markdown-body">{reasoning}</div>
                    </div>
                </div>
                <div class="process-link dashed-link"></div>
                """
                )

            # 3.2 最终回答 (Content)
            # 简单的换行处理，Gradio HTML组件不完全支持所有Markdown，但也足够展示
            formatted_content = str(m_content).replace("\n", "<br>")

            html_parts.append(
                f"""
            <div class="process-node node-ai">
                <div class="node-icon">🤖</div>
                <div class="node-content">
                    <div class="node-title">AGENT RESPONSE</div>
                    <div class="node-text markdown-body">{formatted_content}</div>
                </div>
            </div>
            <div class="process-link"></div>
            """
            )

    html_parts.append('<div class="end-point">END OF PROCESS</div></div>')
    return "".join(html_parts)


# =================【新增】加载默认演示数据逻辑 =================
def load_demo_data():
    """
    加载指定的 /data2/... 下的演示文件
    """
    if not os.path.exists(DISPLAY_JSON_PATH) or not os.path.exists(DISPLAY_IMG_PATH):
        return (
            None,  # Image
            """<div class="verdict-box v-unknown"><div class="v-icon">❌</div><div class="v-text">ERROR</div><div class="v-sub">演示文件未找到</div></div>""",
            "### 错误: 无法找到指定路径的 image.png 或 content.json",
            "<div style='text-align:center'>无法加载图表</div>",  # Graph
            [],  # Chatbot
            {"error": "File not found"},  # JSON
        )

    try:
        # 1. 读取 JSON
        with open(DISPLAY_JSON_PATH, "r", encoding="utf-8") as f:
            messages = json.load(f)

        # 2. 生成图表 HTML
        graph_html = render_process_graph(messages)

        # 3. 提取结论 (简单的启发式规则，模拟 analyze_image 的逻辑)
        # 找到最后一条 AI 消息的内容作为报告
        last_ai_msg = next((m for m in reversed(messages) if m.get("type") == "ai" and "Final Decision" in m.get("content", "")), None)

        verdict_html = """<div class="verdict-box v-unknown"><div class="v-icon">⚪</div><div class="v-text">UNKNOWN</div><div class="v-sub">无法提取结论</div></div>"""
        report_md = "无法提取最终报告。"

        if last_ai_msg:
            content = last_ai_msg.get("content", "")
            report_md = content

            # 根据内容文本判断红绿灯 (逻辑需与 analyze_image 保持一致或适应内容)
            if "Final Decision**: 1" in content or "AI-Generated" in content:
                verdict_html = """<div class="verdict-box v-fake"><div class="v-icon">🔴</div><div class="v-text">FAKE</div><div class="v-sub">检测到 AI伪造痕迹 (Demo)</div></div>"""
            elif "Final Decision**: 0" in content:
                verdict_html = """<div class="verdict-box v-real"><div class="v-icon">🟢</div><div class="v-text">REAL</div><div class="v-sub">未发现篡改痕迹 (Demo)</div></div>"""

        return (
            DISPLAY_IMG_PATH,  # input_image
            verdict_html,  # verdict_output
            report_md,  # reasoning_output
            graph_html,  # 【新增】graph_output
            format_chat_history(messages),  # chatbot_output
            messages,  # json_output
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, "Error", str(e), str(e), [], {}


# ================= 核心分析逻辑 (修改后适配新的输出) =================
def analyze_image(
    image_path,
    vllm_base_url,
    vllm_api_key,
    vllm_model,
    vllm_provider,
    vllm_temp,
    agent_model,
    agent_provider,
    agent_base_url,
    agent_stream,
    agent_reasoning,
    agent_seed,
    agent_num_ctx,
    agent_temp,
):
    # 增加一个 graph_html 空占位符作为第3个返回值
    empty_graph = "<div>System not ready</div>"

    if afa_app is None:
        time.sleep(0.5)
        return (
            """<div class="verdict-box v-unknown"><div class="v-icon">⚪</div><div class="v-text">Error</div><div class="v-sub">系统未初始化</div></div>""",
            "### System Not Initialized",
            empty_graph,
            [],
            {},
        )

    if not image_path:
        return None, "请先选择示例或上传图片。", empty_graph, [], {}

    # 文件处理
    temp_dir = Path(os.environ["GRADIO_TEMP_DIR"]) / "uploads"
    temp_dir.mkdir(exist_ok=True, parents=True)
    target_path = temp_dir / f"analysis_{int(time.time()*1000)}{Path(image_path).suffix}"
    try:
        shutil.copy(image_path, target_path)
    except:
        target_path = Path(image_path)

    try:
        t0 = time.time()
        result = afa_app.analyze_single_image(target_path)
        cost = time.time() - t0

        final = result.get("final_response", {})
        messages = result.get("messages", [])
        pred = str(final.get("pred_result", -1))
        overall = final.get("overall_assessment", "暂无详细评估")

        if pred == "0":
            html = """<div class="verdict-box v-real"><div class="v-icon">🟢</div><div class="v-text">REAL</div><div class="v-sub">未发现篡改痕迹</div></div>"""
        elif pred == "1":
            html = """<div class="verdict-box v-fake"><div class="v-icon">🔴</div><div class="v-text">FAKE</div><div class="v-sub">检测到伪造特征</div></div>"""
        else:
            html = f"""<div class="verdict-box v-unknown"><div class="v-icon">⚪</div><div class="v-text">UNKNOWN</div><div class="v-sub">置信度不足 ({pred})</div></div>"""

        md = f"### ⏱️ Analysis Time: {cost:.2f}s\n\n{overall}"

        # 【新增】生成图表
        graph_html = render_process_graph(messages)

        return html, md, graph_html, format_chat_history(messages), result
    except Exception as e:
        import traceback

        traceback.print_exc()
        return "ERROR", f"Exception: {e}", f"<div>Error: {str(e)}</div>", [], {"error": str(e)}


# 【新增】画廊选择事件处理函数
def on_gallery_select(evt: gr.SelectData):
    if isinstance(evt.value, dict) and "image" in evt.value:
        return evt.value["image"]["path"]
    return evt.value


# ================= CSS 样式 (保留原有 + 新增节点样式) =================
custom_css = """
.gradio-container { font-family: 'Segoe UI', sans-serif !important; }

/* 头部 */
.header-container {
    background: linear-gradient(120deg, #2b5876 0%, #4e4376 100%);
    padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; color: white; text-align: center;
}
.header-container h1 { font-size: 2rem; margin: 0; font-weight: 700; color: white; }

/* 配置栏 */
.config-bar { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; margin-bottom: 15px; }

/* 结论卡片 */
.verdict-box {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 160px;
}
.v-icon { font-size: 2.5rem; }
.v-text { font-size: 2.2rem; font-weight: 800; letter-spacing: 1px; }
.v-sub { font-size: 0.9rem; opacity: 0.9; margin-top: 5px; }
.v-real { background: #10b981; } .v-fake { background: #ef4444; } .v-unknown { background: #6b7280; }

/* 按钮 */
.primary-btn { background-color: #2563eb !important; color: white !important; font-size: 1.1em !important; }

/* 报告 */
.report-area { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }

/* Gallery */
.caption-label { font-size: 0.85em !important; text-align: center; color: #475569; font-weight: 600; }

/* =================【新增】节点链路 CSS ================= */
.process-timeline {
    padding: 20px 10px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    max-height: 800px;
    overflow-y: auto;
}
.process-node {
    position: relative;
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 15px;
    margin: 0 auto;
    max-width: 95%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    display: flex;
    gap: 15px;
    z-index: 2;
}
.node-icon {
    font-size: 1.5rem;
    min-width: 40px;
    height: 40px;
    display: flex; align-items: center; justify-content: center;
    background: #f1f5f9;
    border-radius: 50%;
}
.node-content { flex: 1; overflow: hidden; }
.node-title {
    font-size: 0.85rem; font-weight: 700; color: #64748b; margin-bottom: 5px; text-transform: uppercase;
}
.node-text { font-size: 0.95rem; line-height: 1.5; color: #334155; white-space: pre-wrap; }
.node-body pre { background: #f1f5f9; padding: 10px; border-radius: 4px; font-size: 0.85rem; overflow-x: auto; }

/* 不同类型的节点颜色 */
.node-human { border-left: 4px solid #3b82f6; }
.node-human .node-icon { background: #dbeafe; }
.node-tool { border-left: 4px solid #f59e0b; background: #fffbeb; }
.node-tool .node-icon { background: #fef3c7; }
.node-reasoning { border-left: 4px solid #a855f7; background: #faf5ff; } 
.node-reasoning .node-text { font-style: italic; color: #581c87; }
.node-ai { border-left: 4px solid #10b981; }

/* 连接线 */
.process-link {
    height: 30px;
    width: 2px;
    background: #cbd5e1;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}
.dashed-link {
    background: transparent;
    border-left: 2px dashed #cbd5e1;
    width: 0;
}
.end-point {
    text-align: center; font-size: 0.8rem; color: #94a3b8; font-weight: bold; margin-top: 5px;
}
"""

theme = gr.themes.Soft(primary_hue="blue")

# ================= 界面构建 =================
with gr.Blocks(theme=theme, css=custom_css, title="AFA Forensic") as demo:

    # 1. 标题
    gr.HTML("<div class='header-container'><h1>🔍 AFA Forensic Agent</h1></div>")

    # 2. 全局配置区域
    with gr.Accordion("🛠️  环境配置 (Configuration)", open=True, elem_classes="config-bar"):
        with gr.Row(variant="compact"):
            gr.Markdown("### 📡 VLLM")
            vllm_base_url = gr.Textbox(label="Base URL", value="https://api.openai.com/v1", scale=2)
            vllm_api_key = gr.Textbox(label="API Key", type="password", scale=1)
            vllm_model = gr.Textbox(label="Model", value="gpt-4o", scale=1)
            vllm_provider = gr.Textbox(label="Provider", value="openai", scale=0)
            vllm_temp = gr.Slider(label="Temperature", value=0, maximum=1, step=0.1, scale=1)

        with gr.Row(variant="compact"):
            gr.Markdown("### 🤖 Agent")
            agent_base_url = gr.Textbox(label="Ollama URL", value="localhost:11434", scale=2)
            agent_model = gr.Textbox(label="Model", value="qwen3:32b", scale=1)
            agent_provider = gr.Textbox(label="Provider", value="ollama", scale=0)
            agent_temp = gr.Slider(label="Temperature", value=0, maximum=1, step=0.1, scale=1)

            with gr.Column(visible=False):
                agent_stream = gr.Checkbox(value=False)
                agent_reasoning = gr.Checkbox(value=False)
                agent_seed = gr.Number(value=42)
                agent_num_ctx = gr.Number(value=32000)

    # 3. 主体内容
    with gr.Row():

        # === 左侧：输入控制 ===
        with gr.Column(scale=4):
            gr.Markdown("### 1. 选择示例 (Quick Examples)")

            # columns=3 确保图片较大，allow_preview=False 点击即选中
            example_gallery = gr.Gallery(
                value=get_example_gallery_data(),
                label=None,
                show_label=False,
                columns=2,  # 设置3列，图片会比较大
                height="auto",  # 自适应高度
                object_fit="contain",
                allow_preview=False,  # 关闭预览模式，使其像按钮一样工作
                elem_id="example-gallery",
            )

            gr.Markdown("### 2. 或上传图片 (Or Upload)")

            # 定义图片组件，供 Gallery 填充
            input_image = gr.Image(type="filepath", label="Input Image", height=300)

            # 【新增】增加加载默认Demo的按钮
            with gr.Row():
                analyze_btn = gr.Button("🚀  开始分析 (Analyze)", variant="primary", scale=2, elem_classes="primary-btn")
                load_demo_btn = gr.Button("📂 加载默认案例 (Default Demo)", variant="secondary", scale=1)

        # === 右侧：分析结果 ===
        with gr.Column(scale=6):
            verdict_output = gr.HTML(
                label="Verdict",
                value="""<div class="verdict-box v-unknown"><div class="v-icon">⌛</div><div class="v-text">READY</div><div class="v-sub">请在左侧上传图片并点击分析</div></div>""",
            )

            # 【修改】使用Tabs分隔可视化图表和原始Chatbot
            with gr.Tabs():
                # 【新增】节点链路可视化 Tab
                with gr.TabItem("🔗 分析过程图谱"):
                    graph_output = gr.HTML(label="Visual Graph")

                with gr.TabItem(" 📝 分析报告"):
                    reasoning_output = gr.Markdown("...", elem_classes="report-area")

                with gr.TabItem("💬 对话详情"):
                    chatbot_output = gr.Chatbot(label="Agent Process", height=500)

                with gr.TabItem("📄 原始 JSON"):
                    json_output = gr.JSON(label="Raw JSON")

    # ================= 事件绑定 =================

    config_inputs = [
        vllm_base_url,
        vllm_api_key,
        vllm_model,
        vllm_provider,
        vllm_temp,
        agent_model,
        agent_provider,
        agent_base_url,
        agent_stream,
        agent_reasoning,
        agent_seed,
        agent_num_ctx,
        agent_temp,
    ]

    # 0. 【新增】启动时自动加载 Demo (可选，若不想自动加载可注释掉)
    demo.load(
        fn=load_demo_data, inputs=None, outputs=[input_image, verdict_output, reasoning_output, graph_output, chatbot_output, json_output]
    )

    # 1. 按钮点击分析 (注意 output 数量增加了 graph_output)
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image] + config_inputs,
        outputs=[verdict_output, reasoning_output, graph_output, chatbot_output, json_output],
    )

    # 2. 【新增】Demo 按钮点击
    load_demo_btn.click(
        fn=load_demo_data, inputs=None, outputs=[input_image, verdict_output, reasoning_output, graph_output, chatbot_output, json_output]
    )

    # 3. 画廊点击逻辑
    example_gallery.select(fn=on_gallery_select, inputs=None, outputs=input_image).then(
        fn=analyze_image,
        inputs=[input_image] + config_inputs,
        outputs=[verdict_output, reasoning_output, graph_output, chatbot_output, json_output],
    )

if __name__ == "__main__":
    print(f"🚀 启动服务...")
    # 【修改】allowed_paths 增加 /data2 路径以便读取演示图片
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=[os.getcwd(), "/data2"])
