# 全量模型抽取计划

模型列表：`/home/luotao02/workspace/logs_and_lists/hf_models.txt`（106,658 个）

**机器分工**：
- **机器 A**（8×P40 GPU，28 核 CPU）：Batch 1~3 已完成
- **机器 B**（2×Intel Xeon E5-2650 v3，40 逻辑核，251 GB RAM）：Batch 7 正在运行，CPU-only 模式

---

## 当前状态（May 21）

### 批次进度

| 批次 | 模型数 | 成功数 | 失败数 | 已尝试 | 成功率 | 完成率 | 机器 | 状态 |
|---|---|---|---|---|---|---|---|---|
| **总计（已启动）** | **88,664** | **~19,683** | **~27,096** | **~50,588** | **~38.9%** | **~59.5%** | — | Batch 1~3 + 补跑 + Batch 7 部分 |
| 冒烟测试 | 122 | 66 | 56 | 122 | 54.1% | 100% | A（GPU） | ✅ 完成 |
| **Batch 1：tiny** | 1,674 | **1,064** | **392** | **1,456** | **73.1%** | **87.0%** | A（GPU） | ✅ 完成（已提交 1,064 个） |
| **Batch 2：≤1B** | 3,985 | **2,886** | **1,099** | **3,935** | **73.3%** | **98.7%** | A（CPU） | ✅ 完成 |
| **补跑：≤5B 过滤模型** | 183 | **53** | **38** | 91 | **58.2%** | — | A（CPU） | ✅ 完成 |
| **Batch 3：2~3B** | 5,580 | **~4,716** | **~804** | **5,520** | **85.4%** | **~99.1%** | A（CPU） | ✅ 完成 |
| Batch 4：4~7B | 15,103 | — | — | — | 预期 ~45% | — | — | 待执行（CPU 4并发） |
| Batch 5：8~13B | 8,093 | — | — | — | 预期 ~40% | — | — | 待执行（CPU 4并发） |
| Batch 6：14~29B | 456 | — | — | — | 预期 ~35% | — | — | 待执行（CPU 2并发） |
| **Batch 7：未知/无标注** | 70,724 | **~9,801** | **—** | **~38,771** | **~25.3%** | **~54.8%** | B（CPU） | 五轮均因下载/磁盘满/worker不均匀终止，已清理并生成剩余 35,105 个列表（见下方事故记录） |

**总完成模型：~18,620 个**（Batch 1: 1,064 + Batch 2: 2,886 + 补跑: 53 + Batch 3: ~4,716 + Batch 7: ~9,901）

---

## 执行命令参考

### 启动示例

```bash
nohup bash /home/luotao02/workspace/scripts/start_batch3_sample.sh > /dev/null 2>&1 &
```

各批次脚本位于 `/home/luotao02/workspace/scripts/`：`start_batch3_sample.sh`、`start_batch3.sh`、`start_retry_too_large.sh`、`start_batch7_safe.sh`、`start_batch7_retry.sh`。

### 查看进度

```bash
# 确认进程在跑
ps aux | grep "parallel_extract" | grep -v grep

# 确认 worker 数量正常（16 workers）
ps aux | grep "run_model" | grep -v grep | wc -l

# 查看实时进度
tail -f /home/luotao02/workspace/logs_and_lists/batch7_safe_run_no_llm.log | grep PROGRESS

# 查看成功/失败统计
echo "success: $(ls /home/luotao02/workspace/success/ | wc -l)"
echo "failed: $(ls /home/luotao02/workspace/failed/ | wc -l)"
echo "samples: $(ls /home/luotao02/workspace/samples/ | wc -l)"
```

---

### 各批次进度汇总

#### 已完成的批次

| 批次 | 完成时间 | 已尝试 | 成功 | 失败 | 成功率 | 机器 | 耗时 | 关键参数 |
|---|---|---|---|---|---|---|---|---|
| 冒烟测试 | Apr 28 | 122 | 66 | 56 | 54.1% | A（GPU） | — | — |
| **Batch 1：tiny** | Apr 30 | 1,456 | **1,064** | 392 | **73.1%** | A（GPU） | ~8h | — |
| **Batch 2：≤1B** | May 9~10 | 3,935 | **2,886** | 1,049 | **73.3%** | A（CPU） | ~27h | 16 workers, timeout 900s |
| **补跑：≤5B 过滤模型** | May 11 | 91 | **53** | 38 | **58.2%** | A（CPU） | ~1.6h | 8 workers, timeout 1800s |
| **Batch 3：2~3B** | May 11~18 | 5,520 | **4,716** | 804 | **85.4%** | A（CPU） | ~7 天 | 12 workers, timeout 1200s, max-model-size-b 5 |

**Batch 2 关键发现**：大量 codegen/llama 变体等简单结构模型抽取稳定，实际成功率（73.3%）远超 50 样本预估的 34%。

**Batch 3 关键发现**：
- 分两轮运行：首轮 3,432 个（成功 ~2,917，失败 ~515，成功率 **84.9%**），重启 2,088 个（成功 1,799，失败 289，成功率 **86.2%**）
- **合计 5,520 个，成功 ~4,716，失败 ~804，总成功率 85.4%**
- **最后剩余 10 个模型因进程僵死尚未处理**，待后续手动或单进程补跑
- `--max-model-size-b 5` 释放了被 auto（~3.7B）过滤的模型，是成功率高提升的关键因素
- **事故（May 14）**：进程 16619 于 5 月 14 日 22:18 因 `jan-hq/Poseless-3B-cp-1500` 下载超时卡死 ~12 小时。已修复（`4af8d76` 新增 `signal.alarm(120)` 硬超时保护）并重启完成

#### 进行中的批次

**Batch 7：未知/无标注**
- 机器 B（CPU），70,724 个模型，分三轮运行：

| 轮次 | 运行时间 | 已尝试 | 成功 | 卡死原因 | 产出 |
|---|---|---|---|---|---|
| 第一轮 | May 13~14 | ~2,925 | 964 | `snapshot_download` 在 `analyze` 之前，`madhured/irs-gemma4-e2b-merged` 下载卡死 | 964 success |
| 第二轮 | May 14~15 | ~1,000 | 821 | `Frank290350/ku-typhoon-v1-merged` 下载 `merges.txt` 超时，huggingface_hub 无限重试 | 821 success |
| 第三轮 | May 15~16 | 6,396 | ~1,631 | `ILKT/2024-06-24_22-31-28_epoch_32` 下载触发 `No space left on device`，`ThreadPoolExecutor` 死锁 | ~1,631 success |
| 第四轮 | May 18~19 | 5,638 | ~1,374 | `/home` 磁盘满（100%，仅剩 36M），models 缓存 247G 占满。已停进程、挪 models 到 disk4、重启第五轮 | ~1,374 success |
| 第五轮（已终止） | May 19~21 | 22,812 | 5,011 | `multiprocessing.Queue` 任务分配极度不均匀，14 个 workers 提前退出，仅剩 2 个 worker 处理，速度降至 ~220/小时。已停进程，准备修复后重启 | 5,011 success |
| **合计** | — | **~38,771** | **~9,801** | — | **~9,801 success** |
| **剩余待处理** | — | **35,105** | — | — | — |

---

## 核心策略

> 详细策略说明请参考 [agent_project_summary.md](./agent_project_summary.md#五核心策略详解)

### 策略一：CPU-only + 动态尺寸过滤

**所有批次均改为 CPU 执行**，避免 NVIDIA UVM 驱动导致的 D-state 死锁。

`--max-model-size-b auto` 公式：`RAM × 0.7 / workers / 4`

- 示例：251GB × 0.7 / 12 / 4 ≈ 3.6B 参数上限
- 超大模型在 `analyze()` 阶段即被拒绝，2 秒内快速跳过

### 策略二：三阶段执行（先跳过 llm-retry）

| 阶段 | 策略 | 目标 |
|------|------|------|
| 第一阶段 | 快速扫描（无 llm-retry） | 快速过滤，拿到"确定能成功"的模型集合 |
| 第二阶段 | 调参补跑（仍无 llm-retry） | 对 "Model too large" 的模型，调大 `--max-model-size-b` 统一补跑 |
| 第三阶段 | llm-retry 修复残留失败 | 只对 "Script execution failed" 类型用 `--llm-retry` 逐个修复 |

**收益**：速度提升 2~3 倍，节省 ~90% LLM Token。

---

### ducc LLM 配置

`ducc`（`~/.comate/baidu-cc/bin/ducc`）是 Baidu 内部封装的 Claude Code 命令行工具，GraphNet 的 `--llm-retry` 功能底层通过 `ducc` 执行。

**当前配置**：

```bash
# 查看 ducc 环境变量
ducc env | grep ANTHROPIC

# 输出示例
ANTHROPIC_MODEL=auto
ANTHROPIC_BASE_URL=https://oneapi-comate.baidu.com/api/llm/anthropic
```

| 配置项 | 值 | 说明 |
|---|---|---|
| `ANTHROPIC_MODEL` | `auto` | 默认自动选择模型 |
| `ANTHROPIC_BASE_URL` | `https://oneapi-comate.baidu.com/api/llm/anthropic` | Baidu 内部 oneapi-comate 代理，非 Anthropic 官方直连 |
| 实际调用模型 | `claude-3-5-sonnet-20241022` 或 `claude-3-7-sonnet-20250219` | `auto` 模式下根据负载选择，可通过 `ducc list-models` 查看可用列表 |

**指定模型版本**：

```bash
export ANTHROPIC_MODEL=claude-3-7-sonnet-20250219
ducc -p "修复这段代码..."
```

**Token 消耗估算**：

| 项目 | 数值 |
|---|---|
| 单次 LLM 调用 prompt | ~3,000 tokens |
| 单次 LLM 调用 completion | ~800 tokens |
| 单次 llm-retry 总消耗 | ~3,800 tokens |
| 每个失败模型最多 retry 2 次 | ~7,600 tokens |
| Batch 7 失败模型 ~42,000 个 | **若全开 llm-retry：~3.2 亿 tokens** |
| 先跳过 llm-retry，最后只对 ~3,000 个 script 失败开 | **~2,280 万 tokens，节省 ~90%** |

**注意事项**：
- `oneapi-comate` 代理可能存在 QPS 限制，并发 16 workers 同时触发 llm-retry 时可能排队等待
- `ducc` 已内置认证，无需手动设置 `ANTHROPIC_API_KEY`

---

## 模型规模分布与预估耗时

| 规模/批次 | 模型数 | 运行策略 | 并发 | 预期成功率 | timeout | 耗时/模型 | 墙钟时间 |
|---|---|---|---|---|---|---|---|
| **tiny / ~0** | ~853 | CPU | 8 | ~77%（实测） | 180s | — | — |
| **Mb 级（350m）** | ~3,222 | CPU | 8 | ~65% | 180s | — | — |
| **Batch 2：≤1B** | ~3,793 | CPU | 8 | ~65% | 180s | 45s | **~6 小时** |
| **Batch 3：2~3B** | 5,580 | CPU | 12 | ~55% | 1200s | ~89s（实测） | **~137 小时**（~5.7 天） |
| **Batch 4：4~7B** | ~15,103 | CPU | 4 | ~45% | 500s | 180s | **~189 小时** |
| **Batch 5：8~13B** | ~8,093 | CPU | 4 | ~40% | 800s | 240s | **~135 小时** |
| **Batch 6：14~29B** | ~456 | CPU | 2 | ~35% | 1200s | 480s | **~30 小时** |
| **≥30B** | ~885 | 跳过 | — | — | — | — | — |
| **Batch 7：未知/无标注** | ~70,724 | CPU | 16 | ~40% | 600s | 30s（无 llm-retry） | **~55 小时**（~2.3 天） |
| **合计（Batch 2~7）** | **~103,749** | — | — | — | — | — | **~35 天** |

**注意**：CPU 速度约为 GPU 的 1/3~1/5，总耗时从 ~7 天延长至 ~35 天。如时间不允许，可考虑：
1. 仅抽取 Batch 2~3（≤3B），覆盖 ~80% 的常用模型
2. 或分批提交，抽完一个批次就打包提交一次

---

## 各批次参数与结果

### Batch 3 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--gpus ""` | 空字符串 | **CPU-only 模式**，完全避开 GPU D-state 风险 |
| `--cpu-workers 12` | 12 个并发 | 基于 251GB RAM 计算：并发太多会导致 OOM，太少则耗时过长。12 并发是平衡点 |
| `--timeout 1200` | 1200 秒（20 分钟） | 2~3B 模型加载 + forward + subgraph trace 比 1B 慢很多，需更长的超时 |
| `--max-model-size-b 5` | 5B | 释放之前被 auto（~3.7B）过滤的模型（如 Llama-3.2-3B 实际 4.0B），成功率从 ~34% 提升至 ~83% |
| `--output` | `batch3_sample50.json` / `batch3_full_run.json` | 结果文件保存在 `logs_and_lists/` 下 |

各批次对应 timeout：

| 批次 | timeout |
|---|---|
| Batch 2：≤1B | 180s |
| Batch 3：2~3B | 1200s |
| Batch 4：4~7B | 500s |
| Batch 5：8~13B | 800s |
| Batch 6：14~29B | 1200s |
| Batch 7：未知 | 600s |

### 补跑：被过滤的 ≤5B 模型（May 11）

针对 Batch 1~3 中因 `--max-model-size-b auto` 限制而被早期过滤的 **183** 个 ≤5B 模型，提取其中精确 ≤5B 的 **91** 个进行补跑：

| 参数 | 值 | 说明 |
|---|---|---|
| `--cpu-workers 8` | 8 并发 | 模型较大，降低并发避免 OOM |
| `--timeout 1800` | 1800 秒（30 分钟） | 部分模型结构复杂，需要更长超时 |
| `--max-model-size-b 5` | 5B | 释放被 auto（~2.7B~3.7B）过滤的模型 |
| 模型列表 | `/tmp/retry_lte5b.txt` | 91 个 ≤5B 模型 |

**结果**：

| 指标 | 数值 |
|---|---|
| 测试模型数 | **91** |
| 成功 | **53** |
| 失败 | **38** |
| **成功率** | **58.2%** |
| 耗时 | **5,851s**（约 1.6 小时） |
| 结果文件 | `retry_too_large_results.json` |

**关键发现**：
- 约 **60% 失败是上游模型文件缺失**（OSError：`microsoft/phi-2` 配置缺失、`gradients-io-tournaments` 缓存损坏），不可修复
- 约 **26% 失败是 MoE/自定义架构兼容性问题**（AttributeError：`grouped_mm_fallback`、`attention_bias` 缺失），升级 transformers 可能修复
- 其余 ~14% 是 ValueError/KeyError/ImportError，属特定模型兼容性问题

**`max_model_size_b=auto` 自动计算原理**：
- 公式：`RAM × 0.7 / workers / 4`
- `251GB × 0.7 = 175GB`（系统保留 30% 余量）
- `175GB / 12 workers = 14.5GB` 每个 worker 可用内存
- `14.5GB / 4 bytes/param = 3.6B` 参数上限（按 FP32 保守估算）

### Batch 3 sample50 验证结果（May 11）

| 指标 | 数值 |
|---|---|
| 测试模型数 | 50（完成 48，取消 2） |
| 成功 | **17** |
| 失败 | **31** |
| 成功率 | **~34%** |

**关键发现**：
- **65% 的失败是"模型过大"早期过滤**，非真实抽取失败
- Llama-3.2-3B 系列实际 4.0B，超过 auto 计算的 3.7B 限制
- **未设 HF_TOKEN 导致下载极慢**：3B 模型下载占 90% 时间（30-60 分钟/个）
- `timeout=1200` 只控制 subprocess，不控制主进程下载

**失败分类**：

| 类型 | 数量 | 占比 | 可修复性 |
|---|---|---|---|
| 模型过大/早期过滤 | ~20 | ~65% | 调整 `--max-model-size-b` 到 5B |
| Script execution failed | ~11 | ~35% | 部分可修复 |
| 401/下载失败 | 2 | ~6% | / |

### Batch 7 参数说明（未知/无标注，70,724 个）

Batch 7 的模型规模**完全未知**，名字中约 **14% 含规模提示**（如 7b、3b、8m 等），其余 86% 无任何规模信息。实际抽样显示规模分布极广：

| 规模特征 | 数量 | 占比 |
|---|---|---|
| 名字含规模提示 | ~9,800 | ~14% |
| 其中 0b/tiny/small | ~1,800 | ~2.5% |
| 其中 3b~5b | ~900 | ~1.3% |
| 其中 7b~8b | ~2,200 | ~3.1% |
| 其中 13b~15b | ~280 | ~0.4% |
| **无规模信息** | **~60,900** | **~86%** |

**核心风险**：
- 大模型（7b+）比例不低，如果全部加载会导致 **OOM**
- 小模型（tiny/Mb 级）也很多，不能一棍子打死
- 预估 60-70% 的模型实际 ≤3.7B（可安全运行），但无法预先知道哪些是大模型

**推荐策略：分两轮跑**

**第一轮：安全扫描（`--max-model-size-b auto`）**

| 参数 | 值 | 说明 |
|---|---|---|
| `--model-list` | `batch7_unknown.txt` | 70,724 个模型 |
| `--cpu-workers` | **16** | 40 逻辑核，利用更高 |
| `--timeout` | **600** | 适中超时 |
| `--max-model-size-b` | **auto** (~2.7B) | 只跑 ≤2.7B 的模型，大模型快速过滤 |

**第一轮预期**：
- 成功：**~40%**（~28,000 个）
- 失败中 **~50% 是"模型过大"早期过滤**（~35,000 个）
- 其余 ~10% 是 Script failed / 401 等真实失败

**第二轮：补跑被过滤的 2.7B~5B 模型**

第一轮完成后，从失败结果中提取所有 "Model too large" 的模型，进一步筛选出 ≤5B 的进行补跑：

```python
import json, re
with open('batch7_safe_results.json') as f:
    d = json.load(f)
filtered = [r['model_id'] for r in d['details'] 
            if 'too large' in str(r.get('error','')).lower()]
with open('/tmp/batch7_retry_lte5b.txt', 'w') as f:
    for m in filtered:
        f.write(m + '\n')
```

| 参数 | 值 | 说明 |
|---|---|---|
| `--model-list` | `/tmp/batch7_retry_lte5b.txt` | 第一轮被过滤且 ≤5B 的模型 |
| `--cpu-workers` | **8** | 模型较大，降低并发 |
| `--timeout` | **1200** | 更长超时 |
| `--max-model-size-b` | **5** | 释放被 auto 过滤的 2.7B~5B 模型 |

**第二轮预期**：参考补跑结果，成功率 **~55-60%**

**Batch 7 时间估算**

| 轮次 | 模型数 | 并发 | 预估耗时 |
|---|---|---|---|
| 第一轮（安全扫描，无 llm-retry） | 70,724 | 16 | **~55 小时（~2.3 天）** |
| 第二轮（补跑 ≤5B） | ~5,000-10,000 | 8 | **~20-40 小时** |
| **合计** | — | — | **~3-4 天** |

**注意**：Batch 7 是最大的批次（占总数 66%），但大部分模型是小参数的 fine-tune/变体。如果资源有限，**建议只跑第一轮安全扫描**（覆盖 ~60-70% 的模型），放弃补跑。

**Batch 7 运行注意事项**

1. **排除 gemma4 模型**：列表中约 70 个 gemma4 变体，修复后的代码已自动跳过，但如需手动过滤：
   ```bash
   grep -iv "gemma4" batch7_unknown.txt > batch7_unknown_nogemma4.txt
   ```

2. **监控下载超时**：若日志中某个模型 `Fetching` 进度停滞超过 5 分钟，检查是否为 huggingface_hub 下载卡死：
   ```bash
   ps aux | grep "run_model" | grep -v grep | wc -l  # 应为 16 个 worker
   tail -f logs_and_lists/batch7_safe_run.log | grep PROGRESS  # 1 小时内应有进展
   ```

3. **缓存预热**：Batch 7 中大量模型未缓存，前几个小时速度较慢。缓存命中后速度会提升。

---

## 去重优化建议

fine-tune 变体的计算图结构与 base model 完全相同，可通过 config 指纹去重：

1. 提取 `{"model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"}` 作为指纹
2. 相同指纹的模型只抽取一次，其余直接复用 graph_hash

**预期去重效果**：106K → 估算 10K~20K 个不同图结构（主流架构不超过 50 种）。

---

## 附录：PR 改进列表

本文档聚焦于实验执行过程记录，PR 改进的详细说明请参考 [agent_project_summary.md#四本周详细进展](./agent_project_summary.md#四本周详细进展)。

| # | PR | 简要说明 |
|---|-----|----------|
| 1 | [#713](https://github.com/PaddlePaddle/GraphNet/pull/713) | 新增辅助脚本与去重工具 |
| 2 | [#714](https://github.com/PaddlePaddle/GraphNet/pull/714) | 整理 Workspace 目录结构 |
| 3 | [#715](https://github.com/PaddlePaddle/GraphNet/pull/715) | 支持 CPU-only 模式 |
| 4 | [#716](https://github.com/PaddlePaddle/GraphNet/pull/716) | 修复多子图 hash 生成与统一子图处理 |
| 5 | [#718](https://github.com/PaddlePaddle/GraphNet/pull/718) | 添加下载硬超时保护 |
| 6 | [#719](https://github.com/PaddlePaddle/GraphNet/pull/719) | 模型参数量预估与动态过滤 |
| 7 | [#722](https://github.com/PaddlePaddle/GraphNet/pull/722) | 修复成功数统计逻辑 |
| 8 | [#723](https://github.com/PaddlePaddle/GraphNet/pull/723) | 修复 worker 过早退出问题 |

**相关链接**：
- [完整 PR 列表](https://github.com/PaddlePaddle/GraphNet/pulls?q=is%3Apr+author%3Aluotao1+is%3Aclosed)
- [代码更新记录](https://github.com/PaddlePaddle/GraphNet/compare/develop...luotao1:GraphNet:luotao-fangfangssj-agent?expand=1)
