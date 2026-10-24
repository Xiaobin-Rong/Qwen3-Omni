# Easy-Turn 四状态控制：代码定位与建议

本文是针对复制的 0.30.0 版本的源码分析，不是已经实现的接口。当前阶段仅复制源码和编写设计，未修改生成行为。

## 控制语义

| 信号 | 用户定义 | 控制层动作 |
|---|---|---|
| complete | 用户结束，模型开始回答 | 取出累计的不完整片段与本段音频，形成完整输入；提交新回答请求。若有旧回答仍在运行，采用明确的替换策略：先阻断旧输出、取消旧请求，再启动新请求 |
| incomplete | 用户还没结束，模型继续听 | 保存本段音频，不发起新的模型请求；后续 complete 时拼接。默认不擅自取消已经运行的旧回答；若希望用户开始说话就停旧回答，需要额外的 VAD/打断策略 |
| backchannel | 用户音频不传给模型 | 丢弃本段，不修改 pending_audio，不停止当前回答，也不写入对话历史 |
| wait | 用户要求立刻停止或中止对话 | 先关闭当前回答输出闸门并清空未播放音频，再取消引擎请求、清除待回答输入。默认停止当前回答；“关闭整个会话”应使用额外 close_session 参数明确表达 |

标签在客户端/控制层解释，不作为普通文本附加到模型 prompt；这样不需要模型重新训练。缓存音频建议保存明确的 sample rate 和单调递增 segment_id，避免重复片段和乱序事件。

## 为什么 generate(control_signal=...) 不够

给 generate 增加信号参数能处理调用时的状态，但正在生成期间到达的 wait 必须走独立的事件入口。等 generate 返回后才读取 wait，无法及时打断。同步 generator 还可能阻塞在获取下一个结果，不能靠下一次 yield 的轮询来保证及时控制。

建议两个接口分工：

```python
# 示意接口，尚未实现
await controller.on_audio(session_id, segment_id, signal, audio, sample_rate)
async for event in controller.outputs(session_id):
    ...
```

controller 处理信号，后台 task 用 AsyncOmni.generate 生成；控制信号 task 与输出 task 并发执行。异步只是本地调度方式，不需要 HTTP 或服务器。

## 最小修改范围

### 1. 首先新增控制层，不改模型 forward

建议新增 `vllm_omni/control/turn_controller.py` 和 `types.py`，维护每个会话的：

- `pending_audio`、片段 ID、sample rate；
- `active_request_id`、生成 task、状态锁；
- `response_epoch`：新回答或 wait 时递增，识别迟到输出；
- 待播放音频队列和输出闸门；
- 可选的对话历史。模型不会因为 request_id 相同就自动拥有多轮历史；历史如何组装、被打断的回答保留多少应单独设计。

入口使用 `AsyncOmni`，保留已验证的部署配置和 NCCL 设置。输出消费者只接受当前 request_id/epoch 的数据；已经被取消的 Talker/Code2Wav 迟到数据必须丢弃。

### 2. 首选复用异步入口

文件：`entrypoints/async_omni.py`

- `generate()`，约第 113 行：调用者可提供 request_id，适合会话控制器管理。
- `abort()`，约第 605 行：将调用者的外部 ID 映射为内部请求 ID，再执行取消。

可以先完全不改此文件，用控制层调用已有 API。只有需要统一事件封装或发现退出/清理缺陷时再改。不要把外部 ID 直接传给底层引擎而跳过映射。

### 3. 如果一定保留同步 generate，需要处理请求 ID 和消费循环

文件：`entrypoints/omni.py`

- `generate()` / `_run_generation()`：当前约第 132 行在内部创建 UUID，调用者在首个输出前不便取得请求 ID。
- `abort()`，约第 222 行：同步取消入口。

可能需要允许外部提供 request_id 或返回 request handle；还需让 abort 后的 active_reqs 循环可靠结束。不要在控制线程中调用正在被另一线程执行的 generator.close()。优先使用 AsyncOmni 避免自行重做这些并发边界。

### 4. 底层已具备取消消息传递

| 文件/函数 | 当前实现 | 是否需要立即修改 |
|---|---|---|
| `engine/messages.py::AbortRequestMessage` | 传递 request_ids，可携带 rpc_id | 否 |
| `engine/omni_engine_base.py::abort()`，约 1007 行 | 只入队，不等待完成确认 | 否；不将返回视作所有阶段已停止 |
| `engine/omni_engine_base.py::abort_async()`，约 1024 行 | 等待 Orchestrator 取消确认 | 优先复用 |
| `engine/orchestrator.py::_handle_abort()`，约 592 行 | 执行取消及返回确认 | 先验证现有实现 |
| `engine/orchestrator.py::_abort_request_ids()`，约 632 行 | 遍历各 stage pool 传播取消 | 先验证所有三阶段都收到取消 |
| `engine/orchestrator.py::_cleanup_request_ids()`，约 1488 行 | 负责请求清理链路 | 出现残留或取消后仍调度才修改 |
| `engine/stage_pool.py::abort_requests()`，约 1245 行 | 按请求到 replica 的绑定路由取消，处理输出状态 | 出现路由/资源清理问题才修改 |

先验证这些代码，而不是另写一条平行停止通路。取消是协作式的，正在执行的 GPU kernel 不会被瞬间抢占。

### 5. 音频输出端必须支持 wait

当前 `inference.py` 在生成完成后写整份 WAV，没有实时播放队列。要演示 wait 立即静音，新增输出队列/播放层：

1. 收到 wait 时立即递增 epoch、关输出闸门、清待播队列；
2. 再 await abort，并等待生成 task 退出；
3. abort 返回期间以及之后的旧请求输出一律丢弃；
4. 已播放的音频无法撤销；保留已经播放的时间线，不向被取消结果追加新片段。

引擎取消有时会返回部分已生成 token/结果用于状态清理，不能把这些当作 wait 后允许播放的新内容。

`py_generator=True` 仅表示 Python 逐个产出结果，不保证音频逐小块生成。当前同步三阶段配置可能要等 Talker 完成才生成完整音频；后续若需要低延迟连续播放，再参考 `end2end_async_chunk.py` 的 AsyncOmni + async_chunk 配置。不要把使用 AsyncOmni 与开启 async_chunk 混为一谈。

## 已有 duplex 接口需要单独核实适配性

当前运行包还包含 `entrypoints/duplex_omni.py`，其会话接口有 `append_audio`、`commit`、`cancel_response`、`clear_output_audio`、`signal_turn`。这是可复用方向，但它使用独立的 duplex pipeline/模型适配机制。不能仅因方法存在，就认定当前已跑通的 Qwen3-Omni 三阶段部署可直接替换为 duplex 会话。

## 建议实现与验证顺序

1. 验证副本导入和原版推理结果；不改变权重与采样配置。
2. 控制层实现 incomplete 缓存、backchannel 丢弃、complete 提交。
3. 实现 wait：关闭输出 + abort + 请求/epoch 过滤；测试三个不同阶段收到 wait。
4. 测试重复 wait、complete 与 wait 并发、新回答启动后旧音频迟到、abort 后再次 complete、取消任务资源释放。
5. 有稳定的请求控制后，再接入 Easy-Turn 在线检测与真正流式音频播放。

这套最小方案属于外部信号控制：主要改会话/请求入口与输出端，不需要修改 Qwen3-Omni 的模型 forward、MoE、attention 或 CUDA kernel。
