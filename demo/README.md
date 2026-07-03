# 多模态AI助手 - 使用指南

这是一个基于FastAPI和前端技术构建的多模态AI助手网页应用，支持文本和图片输入，并使用您自己的多模态大模型进行推理。

## 项目结构
```
demo/
├── backend/
│   └── app.py         # FastAPI后端服务
├── frontend/
│   ├── index.html     # 聊天界面HTML
│   └── script.js      # 前端交互逻辑
├── requirements.txt   # 项目依赖
└── README.md          # 使用说明
```

## 环境准备

### 1. 安装依赖
```bash
# 进入demo目录
cd demo

# 创建虚拟环境（可选但推荐）
python -m venv venv
# Windows激活虚拟环境
venv\Scripts\activate
# macOS/Linux激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置模型路径
设置环境变量指定您的多模态模型路径：

Windows (命令行):
```cmd
set MODEL_PATH=您的模型路径
```

Windows (PowerShell):
```powershell
$env:MODEL_PATH="您的模型路径"
```

macOS/Linux:
```bash
export MODEL_PATH=您的模型路径
```

## 启动应用

```bash
# 进入backend目录
cd backend

# 使用uvicorn启动服务
uvicorn app:app --host 0.0.0.0 --port 8000
```
python -m uvicorn backend.app:app --app-dir demo --host 127.0.0.1 --port 8000 --reload 

## 访问应用
打开浏览器，访问以下地址：
http://localhost:8000

## 功能说明
- 支持文本输入和图片上传
- 实时流式显示模型回复
- 类似ChatGPT的用户界面
- 多模态输入处理（文本+图片）

## 注意事项
- 确保您的模型路径正确设置
- 模型需要支持提供的推理接口（model.chat方法）
- 推荐使用GPU运行以获得更好的性能
- 首次运行时模型加载可能需要几分钟时间