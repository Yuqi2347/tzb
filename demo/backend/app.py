# =================================================================
# === 支持多轮对话的后端代码 (版本 2) ===
# =================================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import io, asyncio, torch, os, time, logging, json, re, random, numpy as np
from transformers import AutoModel, AutoTokenizer

# --- 1. 日志和应用状态配置 (无变化) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_state = {
    "initialized": False,
    "initialization_error": None,
    "last_image": None
}

model = None
tokenizer = None

# --- 2. 创建 FastAPI 应用实例 (无变化) ---
app = FastAPI()

# --- 3. 添加中间件 (CORS) (无变化) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def load_model():
    model_path = os.environ.get('MODEL_PATH', 'models/FM9G4B-V')
    logger.info(f"正在从路径加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
    # 尝试设置生成配置为确定性，减少采样导致的波动
    try:
        gen = getattr(model, 'generation_config', None)
        if gen is not None:
            gen.temperature = 0.0
            gen.top_p = 1.0
            gen.do_sample = False
    except Exception:
        pass
    return model, tokenizer


@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    start_time = time.time()
    logger.info("开始初始化应用...")
    # 设定全局随机种子与确定性选项，尽量减少推理不确定性
    try:
        seed = int(os.environ.get('TZB_SEED', '0'))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        logger.warning("设置随机种子失败，将继续初始化。")
    try:
        model, tokenizer = load_model()
        app_state["initialized"] = True
        elapsed = time.time() - start_time
        logger.info(f"应用初始化成功，耗时 {elapsed:.2f} 秒")
    except Exception as e:
        app_state["initialization_error"] = str(e)
        logger.error(f"应用初始化失败: {e}", exc_info=True)


# --- 5. 定义所有的 API 路由 (@app.get, @app.post) ---

@app.get("/health")
async def health_check(): # (无变化)
    return {
        "initialized": app_state["initialized"],
        "error": app_state["initialization_error"],
        "timestamp": time.time()
    }


async def stream_response(text_generator): # (无变化)
    async for chunk in text_generator:
        yield chunk.encode('utf-8')
        await asyncio.sleep(0.01)

# 坐标提取：支持 <box> x1 y1 x2 y2 </box>、[x1,y1,x2,y2]、x1=,y1=,x2=,y2=、以及常见描述格式与百分比/px
def extract_bbox(text: str):
    def _parse_val(s: str) -> float:
        s = s.strip().lower().replace('px', '')
        if s.endswith('%'):
            return float(s[:-1]) / 100.0
        return float(s)

    patterns = [
        # 明确的 <box> 标签
        r"<box>\s*(-?\d+(?:\.\d+)?%?(?:px)?)\s+(-?\d+(?:\.\d+)?%?(?:px)?)\s+(-?\d+(?:\.\d+)?%?(?:px)?)\s+(-?\d+(?:\.\d+)?%?(?:px)?)\s*</box>",
        # JSON/数组形式
        r"\[\s*(\d+(?:\.\d+)?%?)\s*,\s*(\d+(?:\.\d+)?%?)\s*,\s*(\d+(?:\.\d+)?%?)\s*,\s*(\d+(?:\.\d+)?%?)\s*\]",
        # 带键名的形式 x1=, y1=, x2=, y2=
        r"x1\s*[:=]\s*(-?\d+(?:\.\d+)?%?(?:px)?)\s*[,\s]+y1\s*[:=]\s*(-?\d+(?:\.\d+)?%?(?:px)?)\s*[,\s]+x2\s*[:=]\s*(-?\d+(?:\.\d+)?%?(?:px)?)\s*[,\s]+y2\s*[:=]\s*(-?\d+(?:\.\d+)?%?(?:px)?)",
        # 两点形式 (x1,y1), (x2,y2)
        r"\(\s*(-?\d+(?:\.\d+)?%?)\s*,\s*(-?\d+(?:\.\d+)?%?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?%?)\s*,\s*(-?\d+(?:\.\d+)?%?)\s*\)",
        # 关键词引导 + 空格/逗号分隔
        r"(?:bbox|box|coordinates?)[:\s]*(-?\d+(?:\.\d+)?%?)\s*[,\s]+(-?\d+(?:\.\d+)?%?)\s*[,\s]+(-?\d+(?:\.\d+)?%?)\s*[,\s]+(-?\d+(?:\.\d+)?%?)",
        # 纯空格分隔的四个数
        r"(-?\d+(?:\.\d+)?%?)\s+(-?\d+(?:\.\d+)?%?)\s+(-?\d+(?:\.\d+)?%?)\s+(-?\d+(?:\.\d+)?%?)"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            g = m.groups()[:4]
            nums = [_parse_val(v) for v in g]
            x1, y1, x2, y2 = nums
            # 规范化保证左上/右下（在归一化或像素坐标下都适用）
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
    # 关键词存在时的宽松兜底：提取前四个数字
    if re.search(r"\b(box|bbox|coordinate|coordinates)\b", text, re.IGNORECASE):
        tokens = re.findall(r"-?\d+(?:\.\d+)?%?", text)
        if len(tokens) >= 4:
            nums = [_parse_val(tokens[i]) for i in range(4)]
            x1, y1, x2, y2 = nums
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
    return None

def clamp_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    # 至少形成有效框
    if x2 == x1:
        x2 = min(img_w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2

# ==================== 主要修改点 ====================
@app.post("/api/chat")
async def chat(
    text: str = Form(...),
    history_json: str = Form("[]"), # <-- 接收历史记录的JSON字符串
    locate_mode: int = Form(0),
    image: UploadFile = File(None)
):
    if not app_state["initialized"]:
        return StreamingResponse(io.StringIO("错误：模型尚未初始化完成。"), media_type="text/plain", status_code=503)

    # 解析前端传来的历史记录
    try:
        history = json.loads(history_json)
    except json.JSONDecodeError:
        return StreamingResponse(io.StringIO("错误：历史记录格式不正确。"), media_type="text/plain", status_code=400)

    user_image = None
    if image and image.filename:
        try:
            img_bytes = await image.read()
            user_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            # 缓存最近一次上传的图片，便于后续只给坐标也能绘制
            app_state["last_image"] = user_image.copy()
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            return StreamingResponse(io.StringIO(f"错误：无法处理上传的图片: {e}"), media_type="text/plain",
                                     status_code=400)

    # 形成完整上下文并进行一次性推理
    strict_prompt = (
        "请只输出一个边界框坐标（归一化到0到1范围，保留四位小数），使用如下唯一格式，不要输出任何其他文本：\n"
        "<box> x1 y1 x2 y2 </box>\n"
        "其中 x1,y1,x2,y2 为 0 到 1 之间的小数，且 x1 < x2、y1 < y2；若无法定位，请只输出：<error>无法定位</error>"
    )
    effective_text = f"{text}\n\n{strict_prompt}" if locate_mode else text
    current_message = {"role": "user", "content": effective_text}
    msgs = [current_message] if locate_mode else (history + [current_message])

    loop = asyncio.get_event_loop()
    def invoke_chat():
        try:
            return model.chat(
                image=user_image,
                msgs=msgs,
                tokenizer=tokenizer,
                temperature=0.0,
                top_p=1.0,
                do_sample=False
            )
        except TypeError:
            return model.chat(
                image=user_image,
                msgs=msgs,
                tokenizer=tokenizer
            )
    try:
        result = await loop.run_in_executor(None, invoke_chat)
    except Exception as e:
        logger.error(f"模型推理错误: {e}", exc_info=True)
        return StreamingResponse(io.StringIO(f"推理错误: {str(e)}"), media_type="text/plain")

    if locate_mode:
        # 定位模式：仅返回图片或错误提示
        bbox = extract_bbox(result)
        source_img = user_image if user_image is not None else app_state.get("last_image")
        if source_img is not None and bbox is not None:
            w, h = source_img.size
            x1, y1, x2, y2 = bbox
            # 归一化坐标支持：若全部在 [0,1]，按比例映射到像素
            if 0.0 <= min(x1, y1, x2, y2) and max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), w, h)
            output_img = source_img.copy()
            draw = ImageDraw.Draw(output_img)
            stroke = max(2, min(8, int(min(w, h) / 200)))
            logger.info(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=stroke)
            buf = io.BytesIO()
            output_img.save(buf, format='PNG')
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type='image/png')
        else:
            msg = (
                "定位失败：当前会话没有可用图像，请先上传图片" if source_img is None else
                "无法定位：未检测到符合格式的坐标"
            )
            return Response(content=msg, media_type='text/plain')

    # 正常模式：始终按文本流式返回
    async def text_gen_from_result(res_text: str):
        chunk_size = 10
        for i in range(0, len(res_text), chunk_size):
            yield res_text[i:i + chunk_size]
            await asyncio.sleep(0)  # 让出事件循环

    return StreamingResponse(stream_response(text_gen_from_result(result)), media_type="text/plain")
# ======================================================

# --- 6. 挂载静态文件 (更新为跨平台路径) ---
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
