# GraphNet Agent

自动样本抽取 Agent，实现从 HuggingFace ModelID 到 GraphNet Sample 的自动化转换。

## 安装

### 基础依赖
```bash
# 已包含在 GraphNet 主依赖中
pip install torch torchvision
```

### Agent 可选依赖
```bash
# 安装 Agent 相关依赖（包括 huggingface_hub）
pip install -e ".[agent]"

# 或单独安装
pip install huggingface_hub>=0.20.0
```

## 环境配置

### 设置工作空间
```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/path/to/your/workspace
```

或在代码中指定：
```python
from graph_net.agent import GraphNetAgent
agent = GraphNetAgent(workspace="/path/to/workspace")
```

## 使用示例

```python
from graph_net.agent import GraphNetAgent

# 初始化 Agent
agent = GraphNetAgent(
    workspace="./agent_workspace",
    hf_token=None  # 可选，用于访问私有模型
)

# 运行提取
success = agent.extract_sample("bert-base-uncased")

if success:
    print("✅ Sample extracted successfully")
else:
    print("❌ Extraction failed")
```

## 工作流程

1. **Fetch**: 从 HuggingFace 下载模型
2. **Analyze**: 解析 config.json 提取元数据
3. **CodeGen**: 生成 run_model.py 脚本
4. **Extract**: 执行脚本提取计算图
5. **Deduplicate**: 检查是否与已有样本重复
6. **Verify**: 验证样本完整性
7. **Archive**: 保存 run_model.py 到样本目录

## 测试

```bash
# 运行所有测试
pytest graph_net/agent/tests/ -v

# 运行实际模型测试（需要设置环境变量）
TEST_REAL_RUN=1 pytest graph_net/agent/tests/test_real_run.py -v
```

## 辅助脚本

`graph_net/agent/scripts/` 目录下提供批量任务相关的辅助脚本：

### check_extraction_progress.sh

一键查看当前抽取任务的运行状态。

```bash
# 自动查找最新日志
bash graph_net/agent/scripts/check_extraction_progress.sh

# 或指定日志文件
bash graph_net/agent/scripts/check_extraction_progress.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
```

输出包括：进程状态（PID、CPU/内存、Worker 数）、日志最新进度、成功/失败统计、处理速度估算、预计剩余时间、磁盘空间、样本目录文件数。

### analyze_extraction_log.sh

分析已完成批次的抽取日志，输出失败分布和根因统计。

```bash
bash graph_net/agent/scripts/analyze_extraction_log.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
```

输出包括：总体统计（成功率）、失败原因一级分布、模型过大分布、异常类型分布（ValueError/Dynamo/IndexError 等）、HTTP 状态码分布、辅助文件（生成已处理和成功模型列表到 `/tmp/`）。

**环境变量**：两个脚本默认使用 `$HOME/workspace/` 下的目录，可通过环境变量覆盖：

```bash
export GRAPHNET_LOG_DIR=/your/path/logs_and_lists
export GRAPHNET_SUCCESS_DIR=/your/path/success
export GRAPHNET_SAMPLES_DIR=/your/path/samples
```
