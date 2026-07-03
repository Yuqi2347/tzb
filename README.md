# 基于九格多模态大模型的遥感领域优化

[English](README_EN.md)

本项目是第十九届“挑战杯”全国大学生课外学术科技作品竞赛“人工智能+”专项赛作品的开源展示版，面向遥感影像分析与语义理解任务，对九格多模态大模型进行领域适配、推理增强与工程化整理。

项目重点不是提供完整数据和权重的一键复现实验，而是展示一套面向遥感多模态理解的技术方案：通过高质量数据清洗、LoRA 参数高效微调、CoT 思维链增强、GRPO/RGRPO 强化优化和大尺度图像分层聚焦，使通用多模态模型更好地适应遥感场景中的高分辨率、小目标、复杂空间关系和跨语言问答需求。

## 项目成果

| 任务/指标 | Baseline | Ours | 提升 |
| --- | ---: | ---: | ---: |
| VRSBench VQA Overall | 49.22% | 66.11% | +16.89 pp |
| VRSBench Caption BLEU-1 | 0.24 | 0.44 | +0.20 |
| VRSBench Caption BLEU-4 | 0.03 | 0.11 | +0.08 |
| VRSBench Refer Acc@0.5 | 5.48% | 13.31% | +7.83 pp |
| VRSBench Refer Acc@0.7 | 0.74% | 3.73% | +2.99 pp |
| VRSBench Refer mean IoU | 12.14% | 18.66% | +6.52 pp |
| MME-RealWorld-RS EN | 14.93% | 38.90% | 约 +160% |
| MME-RealWorld-RS CN | 13.67% | 40.00% | 约 +192% |

消融实验显示，LoRA 和 CoT/RL 具有互补作用：VRSBench VQA Overall 从 Baseline 的 49.22 提升到 LoRA 的 58.65、CoT/RL 的 57.77，最终组合达到 65.41。

## 技术路线

### 1. 面向遥感场景的数据构建

遥感图文数据常见问题包括标注噪声高、图文不一致、描述粒度不足和空间关系表达不稳定。项目采用 RCRS（Rule-Guided Consistency Rejection Sampling）数据筛选机制：

- 规则过滤：剔除空文本、异常图像、格式错误和低质量标注。
- 教师模型评估：使用 Qwen2.5-VL 对图文样本进行一致性推理。
- 拒绝采样：保留语义一致或相似度达到阈值的样本，提升训练集信噪比。

训练数据覆盖 VQA、Caption 和 Refer 三类任务，结合 EarthVQA、RSVQA-HR、NWPU-Captions、RefDrone 等遥感多模态数据来源进行整理。

### 2. 三阶段模型优化

- LoRA 参数高效微调：在保持九格模型通用能力的基础上，仅更新少量低秩适配参数，向模型注入遥感领域知识。
- CoT 思维链增强：通过显式推理链引导模型处理复杂问答、空间关系和场景推理问题，提升回答的逻辑一致性与可解释性。
- GRPO/RGRPO 强化优化：围绕 VQA、Caption、Refer 设计奖励信号，综合考虑答案准确性、语义一致性和推理链完整性，提高复杂场景下的输出稳定性。

### 3. 大尺度遥感图像推理

针对遥感影像“大范围场景 + 小尺度目标”并存的特点，项目引入分层视觉聚焦策略：先通过多分辨率图像金字塔建立全局感知，再聚焦关键区域进行高分辨率细化推理。九格切片和局部重推理机制用于兼顾全局布局、局部目标和空间关系。

## 评测范围

项目围绕三类核心能力进行评测：

- VQA：遥感视觉问答，使用 Accuracy 衡量类别、数量、颜色、位置、方向、场景和推理等子任务。
- Caption：遥感图像描述，使用 BLEU、METEOR、ROUGE-L、CIDEr 等指标衡量生成质量。
- Refer：自然语言指代表达定位，使用 Acc@0.5、Acc@0.7、mean IoU 和 cum IoU 衡量定位能力。

实验环境为 8 x NVIDIA V100 32GB，使用 torchrun、DeepSpeed ZeRO-2、BF16 混合精度、gradient checkpointing 和 TensorBoard 进行分布式训练与过程监控。报告中的训练成本约为：LoRA 阶段 12 万张数据/4 小时，CoT 阶段 9 万张数据/4 小时，强化学习阶段 2 万张图像/约 10 小时。

## 工程内容

本仓库已按开源项目标准重新整理，保留核心源码、脚本入口、配置模板、文档和最小样例；不提交模型权重、完整数据集、生成结果、私有路径或 API 密钥。

```text
src/remote_sensing_mllm/      核心数据集、Trainer、Reward、对话模板和模型代码
scripts/train/                SFT、LoRA、GRPO、PPO 训练入口
scripts/eval/                 VQA、Caption、Refer 与效率评测入口
scripts/tools/                数据处理、LoRA 合并、标注与维护工具
benchmarks/                   VRSBench、MME-RealWorld 等评测适配代码
configs/                      DeepSpeed 与环境配置模板
demo/                         FastAPI 后端和浏览器前端 Demo
docs/                         数据、训练、评测和 Demo 文档
examples/data/                最小训练/评测 JSON 样例
```

## 快速查看

安装依赖：

```bash
conda env create -f configs/environment.yml
conda activate tzb-remote-sensing-mllm
```

或使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

配置本地模型、数据和图片路径：

```bash
export MODEL_PATH=/path/to/FM9G4B-V
export TRAIN_DATA=/path/to/train.json
export IMAGE_ROOT=/path/to/images
```

训练、评测和 Demo 入口：

```bash
bash scripts/train/finetune_ds.sh
QUESTION_FILE=/path/to/eval.json IMAGE_FOLDER=/path/to/images bash scripts/eval/eval.sh
MODEL_PATH=/path/to/FM9G4B-V python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

## 数据与权重

完整数据集、生成答案、模型 checkpoint、tokenizer 导出文件和评测结果不纳入 Git 跟踪。数据目录规范和下载说明见 [docs/data.md](docs/data.md)。

[examples/data](examples/data) 中的 JSON 仅用于说明数据格式，不能用于真实训练。

## 文档

- [数据准备](docs/data.md)
- [训练说明](docs/training.md)
- [评测说明](docs/evaluation.md)
- [Demo 说明](docs/demo.md)
- [安全说明](SECURITY.md)

## 开源说明

本仓库是成果展示与工程复盘版本，仅保留可公开的源码、配置模板、文档和最小样例，不包含明文密钥、本机绝对路径、大体积数据、模型检查点和完整评测输出。

代码遵循 [LICENSE](LICENSE) 中的开源协议。第三方数据集和预训练模型遵循其原始许可证。
