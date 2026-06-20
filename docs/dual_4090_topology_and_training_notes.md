# 双 RTX 4090 拓扑与训练影响说明

日期：2026-06-08

## 硬件观察

这台服务器有两张 NVIDIA GeForce RTX 4090：

- GPU0：bus `00000000:01:00.0`，CPU affinity `48-63,112-127`
- GPU1：bus `00000000:81:00.0`，CPU affinity `16-31,80-95`，NUMA affinity `1`

`nvidia-smi topo -m` 显示 GPU0 和 GPU1 之间的连接类型是 `SYS`。

`SYS` 表示两张卡之间的路径会跨 PCIe 和 CPU/NUMA 域之间的系统互联。这明显弱于 `PIX`、`PXB`、`PHB`，更远弱于 `NV#` 表示的 NVLink 路径。

## P2P 和 NVLink

实测 P2P 状态：

- `nvidia-smi topo -p2p r`: GPU0 <-> GPU1 = `CNS`
- `nvidia-smi topo -p2p w`: GPU0 <-> GPU1 = `CNS`
- `nvidia-smi topo -p2p a`: GPU0 <-> GPU1 = `NS`
- `nvidia-smi topo -p2p n`: GPU0 <-> GPU1 = `NS`
- `nvidia-smi nvlink -s`：没有报告任何 NVLink 链路

解释：

- 两张 GPU 之间不可用 CUDA peer read/write。
- GPU atomics/native P2P 不可用。
- 没有 NVLink。
- 跨 GPU 数据传输必须走 host/PCIe/NUMA 路径，而不是直接 GPU-GPU 路径。

## 对 SkyRL 的影响

之前 SkyRL 的非 colocated 路径在跨 GPU 权重同步阶段失败。这个现象符合以下判断：该路径假设存在可用的直接 CUDA/NCCL GPU-GPU 传输、CUDA IPC，或者至少假设拓扑行为接近 direct peer access。

设置 `COLOCATE_ALL=true` 之后，训练可以稳定进入 rollout generation、weight sync、logprob forward 和 policy training。但实测 GPU 使用几乎都集中在 GPU0。这符合该 workaround 的性质：它通过 colocate、sleep、wake 推理和训练角色来绕开有问题的跨 GPU weight sync 路径，而不是把 GPU1 作为独立推理设备使用。

## 对多卡训练的影响

这个硬件条件并不意味着完全不能做多卡训练。只要配置保守，FSDP、DDP、DeepSpeed ZeRO，以及基于 socket/PCIe 路径的 NCCL 仍然可以运行。

预期代价：

- 多卡训练可以工作，但 all-reduce、all-gather、reduce-scatter 和参数同步会比 NVLink 或同 root complex GPU 慢。
- 在这台机器上，设置 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1` 是合理的，可以避免走不支持的 P2P/IB 路径。
- 大模型 full fine-tuning 会有较重通信开销；4B 模型在仔细配置 FSDP/ZeRO/offload 的情况下是有希望的，而 17B full RL training 只用这两张卡会比较紧张且速度较慢。

## 一卡推理加一卡训练

这是这台服务器上最薄弱的使用模式。它需要频繁把训练 GPU 上更新后的 actor 权重同步到推理 GPU。没有 P2P 或 NVLink 时，除非框架有健壮的 fallback，否则这些传输必须经过 CPU/host staging。

后果：

- 假设 CUDA IPC 或直接 NCCL peer path 可用的框架，可能失败或卡死。
- 如果框架明确支持 CPU-staged weight update、colocated rollout，或者健壮的 disaggregated rollout，则可能可以运行，但权重同步会成为主要瓶颈。
- 对这台服务器来说，colocated rollout/training 比“GPU0 训练、GPU1 推理”这种 disaggregated 模式更安全，除非已经验证框架能正确处理 non-P2P GPU。

## 对框架选择的影响

换掉 SkyRL 可能改善调度问题或缺失的 fallback 路径，但无法改变这台机器没有 GPU P2P 或 NVLink 的事实。

后续值得测试的候选框架：

- veRL：具备 actor/rollout/reference hybrid worker 设计，并支持 vLLM rollout。值得测试 colocated 模式，以及它显式支持的 disaggregated 模式。
- ms-swift：支持 RLHF/GRPO 类训练，支持 vLLM/SGLang/LMDeploy rollout，也支持 DeepSpeed/FSDP 类分布式训练。建议先测试 2-GPU FSDP/ZeRO。

建议验证顺序：

1. 先跑一个最小 2-GPU FSDP/DeepSpeed supervised fine-tuning smoke test，不带 vLLM rollout。
2. 再用某个框架跑 colocated RL smoke test。
3. 最后才测试 disaggregated inference/training，并且前提是框架明确支持 CPU-staged 或文档说明 non-P2P-safe 的 weight sync。



结论：**“一张卡推理、一张卡训练”在这台服务器上不适合作为主路线**。不是数学意义上绝对走不通，而是这两张 4090 之间没有 CUDA P2P/NVLink，GPU0-GPU1 是 `SYS` 跨 NUMA/CPU 路径，nvidia-smi topo -p2p r/w 都是 `CNS`。这意味着训练卡向推理卡频繁同步权重时，不能依赖直接 GPU-GPU CUDA IPC/P2P，必须走 host/CPU staged fallback，否则框架很容易挂或性能很差。

这不是纯 SkyRL 问题。SkyRL 的非 colocate 路径确实暴露了问题：它的 weight sync 路径对这种非 P2P 双卡不够稳，之前出现 Duplicate GPU/NCCL 相关失败。但根因的一半是硬件拓扑：这台机器本身不适合高频跨卡 actor weight sync。COLOCATE_ALL=true 能跑通，是因为绕开了“一卡推理一卡训练”的坏路径，让 vLLM 和 FSDP 在同卡上 sleep/wake/切换。

换 veRL 或 ms-swift 可能改善框架调度和 fallback，但不能改变没有 P2P/NVLink 这件事。veRL 官方文档里有 actor/rollout/ref hybrid worker 和 vLLM rollout 设计，ms-swift 官方文档也支持 DDP/FSDP/DeepSpeed 以及 vLLM/SGLang/LMDeploy rollout；所以它们值得测，但目标应该是：
1. 先跑通 2 卡 FSDP/DeepSpeed 纯训练 smoke test，不带 vLLM rollout。

2. 再测 colocated RL。

3. 最后才测 disaggregated rollout/training，并确认框架有 CPU-staged 或 non-P2P-safe weight sync。

对后续 4B/17B：多卡训练这条路仍然要跑通，但不要把“GPU0 训练、GPU1 推理”当成前提。4B 用 2 卡 FSDP/ZeRO 应该有希望；17B 在这两张 4090 上做完整 RL full finetune 会很紧，更多要依赖 FSDP/ZeRO3/offload/LoRA/QLoRA 或更多 GPU。

我已经把本机双卡情况写到：docs/dual_4090_topology_and_training_notes.md。

参考的框架文档：
veRL engine workers: https://verl.readthedocs.io/en/latest/workers/engine_workers.html
veRL installation/backend notes: https://verl-doc.readthedocs.io/en/latest/preparation/install.html
ms-swift quick start: https://swift.readthedocs.io/en/v3.11/GetStarted/Quick-start.html
Qwen ms-swift RL guide: https://qwen.readthedocs.io/en/v2.5/training/RL/ms_swift.html