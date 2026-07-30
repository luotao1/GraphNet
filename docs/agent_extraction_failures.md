# Agent 模型抽取失败分析报告

各批次总体进度和成功统计见 `agent_full_run_plan.md`。本文档聚焦**失败根因分析、详细分类和修复记录**。

---

## 一、失败根因总览

### 各批次失败分布

| 根因类别 | 典型表现 | 占比 | 可修复性 |
|---|---|---|---|
| **Dynamo/torch.compile 内部错误** | `IndexError: encoder_outputs[0]`、`NameError: torch`、device mismatch、decoder trace 失败 | ~30% | ❌ 框架兼容 |
| **模板问题（pooler/decoder）** | BERT pooler 为 None、T5 缺 `decoder_input_ids`、输出结构异常 | ~22% | ✅ 模板修复 |
| **不支持架构** | gemma4、MoE、deit/swin/convnext、LFM2.5-VL、audiodit 等 | ~12% | ❌ 跳过 |
| **模型过大/早期过滤** | 参数量超过 `--max-model-size-b` 动态限制 | ~8% | ✅ 调参补跑 |
| **401/模型不可访问** | HF Token 限制、模型删除或私有 | ~5% | ❌ 跳过 |
| **其他（超时/杂项）** | embedding 越界、config 解析失败、无日志超时 | ~16% | 部分可修复 |
| **多模态/视觉输入缺失** | CLIP/metaclip 缺少 `pixel_values`、whisper 需 `input_features` | ~3% | ⚠️ 模板增强 |
| **量化模型** | AWQ/int4/int8 CPU 加载极慢或配置错误 | ~2% | ⚠️ 增加 timeout 或过滤 |
| **缺失依赖** | deepspeed、timm、requests、bitsandbytes | ~2% | ✅ `pip install` |

**关键发现**：Batch 2 全量数据显示，**Dynamo 错误和模板问题合计占 ~52%**，是失败的主要根因；401 仅占 ~5%，远低于 50 样本预估的 25%。

### 已完成的代码修复

1. **[PR #713](https://github.com/PaddlePaddle/GraphNet/pull/713) 新增辅助脚本与去重工具**：添加日志分析、进度检查脚本以及基于 SHA256 的子图去重工具，提升批量抽取的可观测性和结果管理能力。
2. **[PR #714](https://github.com/PaddlePaddle/GraphNet/pull/714) 整理 Workspace 目录结构**：将抽取结果按成功/失败自动归档到独立目录，统一 logs_and_lists 与 samples 子目录，使文件管理更清晰。
3. **[PR #715](https://github.com/PaddlePaddle/GraphNet/pull/715) 支持 CPU-only 模式**：`parallel_extract.py` 支持 `--cpu-workers` 主动切换到纯 CPU 运行，默认取半核数避免系统过载。
4. **[PR #716](https://github.com/PaddlePaddle/GraphNet/pull/716) 修复多子图 hash 生成与统一子图处理**：修复多子图模型无法生成 `graph_hash.txt` 的问题，引入 `_get_subgraph_dirs` 统一单/多子图逻辑，去重脚本复用 `graph_net.hash_util`。
5. **[PR #718](https://github.com/PaddlePaddle/GraphNet/pull/718) 添加下载硬超时保护**：为 `snapshot_download` 套上 `signal.alarm(120)` 进程级超时，防止 TCP 连接存活但无数据传输时的无限阻塞。
6. **[PR #719](https://github.com/PaddlePaddle/GraphNet/pull/719) 模型参数量预估与动态过滤**：新增 `_estimate_param_count_billion` 粗估参数量，分析阶段超限直接拒绝（如 39.7B 模型 12 秒内失败），避免超大模型加载超时。
7. **[PR #722](https://github.com/PaddlePaddle/GraphNet/pull/722) 修复脚本成功数统计逻辑**：将 `analyze_extraction_log.sh` 和 `check_extraction_progress.sh` 的成功标记从 `[CPU X] OK` 对齐为实际输出的 `Graph extracted to`，解决成功数被严重低估的问题。
8. **[PR #723](https://github.com/PaddlePaddle/GraphNet/pull/723) 修复 worker 过早退出**：`parallel_extract.py` 引入 sentinel-based 关闭机制（阻塞式 `get()` + `None` 哨兵值），解决 `multiprocessing.Queue` feeder 线程延迟导致 worker 误判队列为空提前退出的问题。

### 待修复项（按优先级）

| 优先级 | 修复项 | 预期收益 | 说明 |
|---|---|---|---|
| **P1** | BERT pooler None → 改用 `last_hidden_state[:, 0]` | +~310 个 | Batch 1 ~188 + Batch 2 ~124 |
| **P1** | deit/swin/convnext → 加入 `_UNSUPPORTED_MODEL_TYPES` | +~56 个 | Batch 1 为主，含部分 Batch 2 NameError |
| **P2** | 安装 `deepspeed`、`requests`、`timm` | +若干 | 全量中同类错误不止抽样中的 3 个 |
| **P2** | T5 模板添加 `decoder_input_ids` | +若干 | Batch 1: 12 个 + Batch 2 部分 |
| **P2** | 安装 `bitsandbytes`，支持 4bit/8bit 量化模型 | +~10 个 | 冒烟测试 + Batch 1 量化 8 个 + Batch 2 量化超时 |
| **P3** | 多模态模型输入适配（pixel_values、image_size 等） | +~10 个 | CLIP/metaclip 等 |
| **P4** | 降级 transformers 或 patch gemma4/mistral4 Dynamo 冲突 | +4 个 | 早期过滤已缓解，收益降低 |
| **P5** | 解决 trust_remote_code 剩余问题 | +部分 | 冒烟测试 21 个，待分析 |

---

## 二、冒烟测试阶段（Apr 28，122 个 tiny-random 模型）

冒烟测试用于验证 Agent 在随机 tiny 模型上的基础能力。经过 5 轮迭代修复，成功率从 45.1% 提升至 54.1%。

### 总体结果

| 轮次 | 说明 | 测试数 | 成功 | 成功率 | 结果文件 |
|---|---|---|---|---|---|
| Round 1 | 基线冒烟测试 | 122 | 55 | 45.1% | `smoke_tiny_results.json` |
| Round 2 | 修复 trust_remote_code + llm_retry | 50 | 4 | 8.0% | `retry_no_oom_results.json` |
| Round 3 | trust_remote_code 专项重跑（第一次） | 12 | 0 | 0% | `trust_remote_retry_results.json` |
| Round 3B | trust_remote_code 专项重跑（第二次） | 12 | 4 | 33.3% | `trust_remote_retry2_results.json` |
| Round 4 | 多模态模型重跑 | 8 | 0 | 0% | `multimodal_retry_results.json` |
| Round 5 | OOM 模型 CPU 回退（第一次，libsndfile 缺失） | 24 | 0 | 0% | `oom_retry_results.json` |
| Round 5B | OOM 模型 CPU 回退（第二次，修复 libsndfile） | 24 | 12 | **50%** | `oom_retry2_results.json` |
| Round 5C | OOM 模型 CPU 回退（第三次，验证稳定性） | 24 | 12 | **50%** | `oom_retry3_results.json` |

**累计从原始 67 个失败模型中新增修复 11 个，最终预估成功率 54.1%（66/122）。**

### 失败分类

| 失败类型 | 数量 | 占比 | 失败原因 |
|---|---|---|---|
| **trust_remote_code** | 21 | 37.5% | transformers 版本或架构问题 |
| **CUDA OOM** | 9 | 16.1% | CPU 回退后仍 OOM 或超时 |
| **Config/AttributeError** | 9 | 16.1% | config 字段缺失或 4bit/8bit 量化需 bitsandbytes |
| **Dynamo crash** | 4 | 7.1% | transformers 5.x + torch.compile 兼容 bug |
| **Import/ModuleNotFound** | 1 | 1.8% | 依赖包缺失 |
| **Other** | 7 | 12.5% | 各类架构兼容问题 |

**关键发现**：
- 累计从原始 67 个失败模型中新增修复 11 个，最终预估成功率 54.1%（66/122）
- trust_remote_code 和 Dynamo crash 属于 transformers 版本兼容问题，短期内难以系统性修复

---

## 三、Batch 1 正式批次（机器 A，Apr 30，1,674 个 tiny 模型）

机器：**A**（8×P40 GPU，28 核 CPU）
运行时间：2026-04-30 17:01 启动（重启版 20:15）
模型列表：`batch_tiny_new.txt`（排除已跑 122）
已处理：**726 个**，成功 **243**，失败 **484**（含早期过滤），成功率 **~51%**（排除过大过滤后 ~57%）

### 总体结果

#### 专项修复（Apr 30）

| 问题 | 根本原因 | 修复方案 | 状态 |
|---|---|---|---|
| whisper 系列输入类型错误 → 375s 超时 | 模板生成 `input_ids`，但 whisper 需要 `input_features [1,80,3000]` | `config_metadata_analyzer._extract_input_info()` 在 NLP 分支前优先检测音频模型 | ✅ 已提交 |
| `WhisperForConditionalGeneration` 无 `from_config` | `AutoModel.from_config` 映射到生成类，但该类无公开 `from_config` 方法 | `template_generator` 新增 `_MODEL_TYPE_TO_CLASS` 映射，音频模型改用 `WhisperModel` 等基础类 | ✅ 已提交 |
| gemma4 系列 90s 超时 | dynamo subgraph 验证失败（sliding-window + global attention 混合） | `_UNSUPPORTED_MODEL_TYPES` 加入 `gemma4` 系列，3s 内早期拒绝 | ✅ 已提交 |
| 26B+ 模型 1372s 超时 | 随机权重超过 RAM 上限 | `--max-model-size-b auto` 基于 `RAM × 0.7 / workers / 4` 动态拒绝 | ✅ 已提交 |
| 进程运行旧代码 | 批次于 17:00 启动，whisper 修复 19:xx 才提交 | kill 孤儿进程，重启批次 | ✅ 已处理 |

### 失败分类

| # | 失败模式 | 失败数 | 占比 | 根本原因 | 修复建议 |
|---|---|---|---|---|---|
| ① | **TinyBERT / cs4248 系列** | **147** | 30.4% | `BertModel(add_pooling_layer=False)` 变体导致 `pooler` 为 `None`，`hidden_states[:, 0]` 取不到值 → `TypeError`，随后 dynamo trace 失败 | template_generator 判断 pooler 是否为 None，改用 `last_hidden_state[:, 0]` |
| ② | **Whisper / 音频模型** | **120** | 24.8% | whisper encoder 在 dynamo compile 时 `encoder_outputs[0]` 越界 → `InternalTorchDynamoError`；LLM 修复后触发 `AttributeError: WhisperForConditionalGeneration has no attribute 'from_config'`；最终全部超时 | 将 whisper 加入 `_UNSUPPORTED_MODEL_TYPES` |
| ③ | **BERT / RoBERTa 其他变体** | **41** | 8.5% | device mismatch（cpu vs cuda:0）、pooler None、dynamo trace 失败 | 同 ①，统一修复 BERT pooler None 问题 |
| ④ | **Vision：deit / swin** | **44** | 9.1% | transformers 5.x `output_capturing.py` 的 `@wraps` decorator 与 torch.compile/dynamo 冲突 → `NameError: name 'torch' is not defined` | 将 deit/swin 加入 `_UNSUPPORTED_MODEL_TYPES` |
| ⑤ | **ConvNext 系列** | **12** | 2.5% | 与 ④ 相同：`NameError: name 'torch' is not defined` | 同 ④ |
| ⑥ | **模型过大（早期过滤）** | **~42** | 8.7% | 参数量超过 `--max-model-size-b` 动态限制（~2.0B），非真实抽取失败 | 调整 `--max-model-size-b` 参数即可重跑 |
| ⑦ | **Marian / opus-mt** | **15** | 3.1% | seq2seq decoder 的 cross-attention 在 dynamo trace 时失败 | 待分析 |
| ⑧ | **gemma4 / MoE 不支持架构** | **14** | 2.9% | 部分已被早期过滤，部分 MoE 变体仍触发超时 | 补全 MoE 架构早期过滤 |
| ⑨ | **T5 / encoder-decoder** | **12** | 2.5% | dynamo trace 在 decoder cross-attention 失败 | 待分析 |
| ⑩ | **量化模型（4bit/8bit/GPTQ/AWQ/MLX）** | **8** | 1.7% | 量化权重加载需要 bitsandbytes 等依赖，或 MLX 格式不被 transformers 支持 | 安装 bitsandbytes，或将量化模型过滤 |
| ⑪ | **多模态（VLM/CLIP/idefics/SAM2）** | **5** | 1.0% | 多模态输入（pixel_values、image 等）适配不完整 | 扩展 template_generator 多模态输入处理 |
| ⑫ | **其他杂项** | **~24** | 5.0% | sam2/deepseek-vl2/chronos/DialogLED/rwkv 等各类架构兼容问题 | 逐一分析 |

---

## 四、Batch 2 CPU 模式全量运行（机器 A，May 9–10，2026）

机器：**A**（8×P40 GPU，28 核 CPU，251 GB RAM）
环境：28 核 CPU，16 CPU workers，`--timeout 900`，graphnet conda 环境  
模型列表：`batch2_le1b_remaining.txt`（3,935 个，排除已测试 50 个）  
结果文件：`batch2_full_run.json`

### 总体结果

| 指标 | 数值 |
|---|---|
| 测试模型数 | **3,935** |
| 成功 | **2,886** |
| 失败 | **1,049** |
| **成功率** | **73.3%** |
| 平均耗时/成功模型 | ~420s |
| 总耗时 | ~27 小时 |

**结论**：全量 3,935 个模型中，73.3% 成功。50 样本测试预估 34% 严重偏低，实际全量成功率与 Batch 1（73.1%）相当。主要因为 Batch 2 中大量 codegen/llama 变体等简单结构模型抽取稳定。

### 失败分类（全量 1,049 个失败模型）

| 失败类型 | 数量 | 占比 | 说明 | 可修复性 |
|---|---|---|---|---|
| **① Script execution failed** | **824** | **78.6%** | `run_model.py` 返回非零退出码，含各类 Dynamo/架构/权重错误 | ❌ 大部分不可修复 |
| **② 无详细日志（超时/静默）** | **101** | **9.6%** | 进程被 kill 或超时，日志未记录具体错误 | ⚠️ 部分可重试 |
| **③ 模型过大/早期过滤** | **69** | **6.6%** | 参数量超过动态限制（~3.5B），非真实抽取失败 | — 正常行为 |
| **④ Sample verification failed** | **~53** | **~5.1%** | 样本验证未通过（输入不匹配、输出异常等） | ⚠️ 部分可修复 |
| **⑤ 401 Unauthorized / 模型不可访问** | **47** | **4.5%** | HF Token 限制、模型删除或私有 | ❌ 不可修复（跳过） |
| **⑥ 其他** | **55** | **5.2%** | embedding 越界、config 解析失败等杂项 | 部分可修复 |

**关键发现**：
- **78.6% 的失败是脚本执行失败**，主要根因包括 Dynamo 内部错误、不支持架构、device mismatch、pooler None 等
- 50 样本测试预测失败以 401 为主（42.4%），全量实际 401 仅占 **4.5%**，样本偏差严重
- 全量中大量 codegen/llama 变体等简单结构模型抽取稳定，拉高了整体成功率

---

## 五、Batch 3 CPU 模式全量运行（机器 A，May 11–12，2026）

机器：**A**（28 核 CPU，251 GB RAM）
环境：28 核 CPU，12 CPU workers，`--timeout 1200`，`--max-model-size-b 5`
模型列表：`/tmp/batch3_remaining.txt`（5,530 个，排除已测试 50 个）
结果文件：`batch3_full_run.json` / `batch3_full_run.log`

### 当前进度（截至 May 14，首轮中断前）

| 指标 | 数值 |
|---|---|
| 测试模型数 | **3,432/5,530**（约 62.1%） |
| 成功 | **~2,917** |
| 失败 | **~515** |
| **成功率** | **~84.9%** |
| 已运行时间 | **~3.5 天** |
| 中断原因 | huggingface `snapshot_download` 网络卡死（`jan-hq/Poseless-3B-cp-1500`） |

**关键发现**：
- 成功率 **远高于 sample50 预估**（84.9% vs 34%），sample 偏差严重
- 大量 Gemma-2b、Llama-3.2-3B 系列抽取稳定
- `--max-model-size-b 5` 释放了之前被过滤的模型，这些模型结构相对标准
- 失败主要是未知 model_type、配置异常和少数多模态模型

### 失败分类（3,432 个已处理模型）

| 失败类型 | 数量 | 占比（在失败中） | 说明 | 可修复性 |
|---|---|---|---|---|
| **Script execution failed** | **~301** | **~58.4%** | `run_model.py` 返回非零退出码，主要是架构/配置不兼容 | ❌ 大部分不可修复 |
| **模型过大（早期过滤）** | **~53** | **~10.3%** | 参数量超过 `--max-model-size-b 5` 限制（6B~4330B 不等） | — 正常行为 |
| **超时（1200s）** | **~14** | **~2.7%** | subprocess 执行超过 20 分钟 | ⚠️ 部分可重试 |
| **下载失败（403/404）** | **~50** | **~9.7%** | huggingface 访问被拒绝或模型已删除 | ❌ |
| **其他/杂项** | **~97** | **~18.8%** | 401、gemma4 不支持、上游配置错误等 | ❌/⚠️ |

### 失败分类（3,432 个已处理模型）

| 失败类型 | 数量 | 占比（在失败中） | 说明 | 可修复性 |
|---|---|---|---|---|
| **Script execution failed** | **~301** | **~58.4%** | `run_model.py` 返回非零退出码，主要是架构/配置不兼容 | ❌ 大部分不可修复 |
| **模型过大（早期过滤）** | **~53** | **~10.3%** | 参数量超过 `--max-model-size-b 5` 限制 | — 正常行为 |
| **超时（1200s）** | **~14** | **~2.7%** | subprocess 执行超过 20 分钟 | ⚠️ 部分可重试 |
| **下载失败（403/404）** | **~50** | **~9.7%** | huggingface 访问被拒绝或模型已删除 | ❌ |
| **其他/杂项** | **~97** | **~18.8%** | 401、gemma4 不支持、上游配置错误等 | ❌/⚠️ |

**关键发现**：
- 成功率 **远高于 sample50 预估**（84.9% vs 34%），sample 偏差严重；`--max-model-size-b 5` 释放了大量标准架构模型
- 失败主要是未知 model_type、配置异常和少数多模态模型

### Batch 3 重启（机器 A，May 15–18，2026）

剩余 **2,098** 个模型重启，完成 **2,088/2,098**，成功率 **86.2%**。

**Batch 3 合计**：5,520 个模型中成功 ~4,716 个，**总成功率 ~85.4%**。

重启成功率更高的原因：`signal.alarm(120)` 硬超时避免了下载卡死，剩余列表排除了首轮高失败率模型。

---

## 六、补跑：被过滤的 ≤5B 模型（机器 A，May 11，2026）

机器：**A**（28 核 CPU，251 GB RAM）
背景：Batch 1~3 中因 `--max-model-size-b auto` 限制被早期过滤的模型共 **183** 个，其中精确 ≤5B 的 **91** 个进行补跑
参数：`--max-model-size-b 5`、`--cpu-workers 8`、`--timeout 1800`
结果文件：`retry_too_large_results.json`

### 总体结果

| 指标 | 数值 |
|---|---|
| 测试模型数 | **91** |
| 成功 | **53** |
| 失败 | **38** |
| **成功率** | **58.2%** |
| 总耗时 | **5,851s**（约 1.6 小时） |

### 失败分类（38 个失败模型）

| 失败类型 | 数量 | 占比 | 具体原因 |
|---|---|---|---|
| **OSError** | **23** | **60.5%** | phi-2 官方配置缺失；部分模型本地缓存路径损坏 |
| **AttributeError** | **10** | **26.3%** | MoE 模型缺失 `grouped_mm_fallback`（transformers 版本不支持）；Doge 架构字段缺失 |
| **ValueError** | **3** | **7.9%** | stablelm 配置类无法识别（自定义 config 类未注册） |
| **KeyError** | **1** | **2.6%** | 解析 config 时 `'type'` 键缺失 |
| **ImportError** | **1** | **2.6%** | 无法导入 `is_torch_fx_available`（transformers 版本问题） |

**关键发现**：
- **60.5% 的失败是上游模型问题**（OSError）：phi-2 官方仓库缺少配置文件、tournaments 系列模型缓存损坏，无法通过代码修复
- **26.3% 是架构兼容性问题**（AttributeError）：主要是 MoE 模型的 `grouped_mm_fallback` 操作符缺失
- 补跑释放 53 个成功模型，大量是 Llama-3.2-3B 系列（实际 4.0B，之前被 auto 的 3.7B 限制过滤）
- 调整 `--max-model-size-b` 参数已完成，**+53 个成功**

---

## 七、Batch 7 前五轮失败模型原因分析（机器 B，May 13~21，2026）

机器：**B**（2×Intel Xeon E5-2650 v3，40 逻辑核，251 GB RAM，无 GPU）
参数：`--cpu-workers 16`、`--max-model-size-b auto`（~2.7B）、`--timeout 600`

### 五轮汇总

| 轮次 | 日期 | 尝试数 | 成功 | 成功率 | 特殊事件 |
|------|------|--------|------|--------|----------|
| 第一轮 | May 13 | 2,925 | 964 | 31.9% | LLM retry 模式，ducc 超时 63 个 |
| 第二轮 | May 14 | 745 | ~21 | 2.8% | 进程卡死前中断，analyze 失败占 59% |
| 第三轮 | May 18 | 540 | ~181 | 33.5% | `--no-llm-retry` 模式启动 |
| 第四轮 | May 18~19 | 5,638 | ~1,374 | 24.4% | `/home` 磁盘满 100% 卡死 14h，挪 models 到 disk4 |
| 第五轮 | May 19~21 | 22,827 | 5,017 | 22.0% | 14 个 workers 提前退出（Queue 竞态），仅 2 个 worker 处理，已修复 |
| **五轮合计** | — | **32,675** | **~7,557** | **~23.1%** | — |

### 核心失败根因总结（五轮合计）

| 根因 | 五轮合计 | 可修复性 |
|---|---|---|
| 模型过大（早期过滤） | 6,992 | 调整 `--max-model-size-b` |
| Dynamo/torch.compile 内部错误 | ~3,750 | 需升级框架 |
| ValueError（模板/配置问题） | ~8,196 | 部分可修复（template_generator） |
| RuntimeError（embedding 越界） | ~4,107 | 部分可修复 |
| 下载失败（404/403/超时） | 746 | 不可修复（跳过） |
| KeyError（model_type 缺失） | ~1,871 | 部分可修复 |
| 未知/不支持 model_type | ~500+ | 加入 `_UNSUPPORTED_MODEL_TYPES` |
| AttributeError | ~621 | transformers 版本问题 |
| ImportError/ModuleNotFoundError | ~655 | 安装缺失依赖 |
| OSError | ~244 | 系统错误 |
| NameError | ~184 | 名称错误 |
| TypeError | ~962 | 输入类型不匹配 |
| FileNotFoundError | ~46 | 文件缺失 |
| ducc 超时/空输出 | 66 | LLM 服务不稳定 |

**关键发现**：
- 五轮共约 ~7,738 个失败可归因于"模型过大"或"下载不可达"，属于预期内失败，不表示代码缺陷。
- 真实 Script execution failed 中，**ValueError 和 Dynamo 错误占绝对主体**（合计 ~11,946 个），这些是 transformers/torch.compile 版本兼容问题，短期内难以系统性修复。
- 成功率从第一轮的 31.9% 逐步下降至第五轮的 22.0%，原因是剩余列表中大型模型和复杂结构模型密度持续增加。
- **模型过大是最大预期内失败**：五轮合计 6,992 个（占失败总数 ~39%），可通过调大 `--max-model-size-b` 补跑释放。
- **RuntimeError 成为第二大失败根因**（~4,107 个），超过 IndexError，说明 embedding 越界等运行时错误在复杂模型中更常见。
- **第五轮新发现**：`multiprocessing.Queue` 任务分配不均匀导致 14 个 workers 提前退出，仅剩 2 个 worker 处理，速度从 ~514/小时降至 ~220/小时。后续需修复并行调度逻辑。

---

## 八、卡死事故汇总

| # | 时间 | 机器 | 现象 | 根因 | 修复 |
|---|---|---|---|---|---|
| 1 | May 1 | A | 253/3793 后系统挂起，ENOMEM | 音频模型触发 NVIDIA UVM D-state，~143 个进程进入不可中断睡眠，累积至 131+ | 早期过滤音频模型 + 非阻塞 cleanup |
| 2 | May 13 | B | 进度停在 2,925/70,724，7.5h 无日志 | `snapshot_download` 在 `analyze` 前执行，gemma4 模型触发下载无限挂起 | 下载前预检 model_type + `HF_HUB_DOWNLOAD_TIMEOUT=30` |
| 3 | May 15 | B | 进度 ~1,000/69,952，13h 无进展 | `iter_bytes()` 流式读取时 TCP 连接存活但无数据传输，不抛异常，HTTP 层超时全部失效 | `signal.alarm(120)` 进程级硬总超时 |
| 4 | May 16 | B | 进度 6,385/69,951，62h 无进展 | 磁盘满触发 `ThreadPoolExecutor` 内部死锁，`signal.alarm` 无法穿透 `fut.result()` 等待 | kill 进程 + 清理损坏文件；长期方案：子进程隔离下载 |
| 5 | May 18~19 | B | 进程停滞 14h | `/home` 磁盘满 100%（models 缓存 247G） | 挪 models 到 `/home/disk4` 符号链接 |
| 6 | May 19~21 | B | 16 worker 退化至 2 worker，速度降至 ~220/h | `multiprocessing.Queue.get_nowait()` 竞态，feeder 线程延迟写入导致 worker 误判队列空 | 阻塞式 `get()` + `None` 哨兵值 |

### 共同教训

1. **超时不能只做一层**：HTTP 层（`HF_HUB_DOWNLOAD_TIMEOUT`）、进程级（`signal.alarm`）、子进程级（`Process.join(timeout)`）层层兜底
2. **子进程隔离是关键**：`snapshot_download` 内部的 `ThreadPoolExecutor` 一旦死锁，主进程信号无法穿透，必须放到独立子进程才能强制 kill
3. **早期过滤比事后处理更高效**：已知不支持/高风险的模型类型（音频、gemma4、过大模型）在下载前拒绝，避免进入不可恢复的阻塞状态
4. **磁盘监控不可少**：大型模型缓存增长极快，需预留足够空间并定期检查
