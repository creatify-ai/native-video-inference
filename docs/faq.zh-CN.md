# 面向 PyTorch、SGLang 和 vLLM 工程师的 FAQ

[English](faq.md)

为从 PyTorch、Diffusers、SGLang 或 vLLM-Omni 转过来的工程师准备的简短问答。[设计决策](#设计决策)解释框架为什么这样构建，[速度](#速度)把它和 SGLang、vLLM-Omni 做对比。每个回答都链接到写有细节的文档。除非另有说明，数字都是在 Modal 上的 NVIDIA B200 测得的。

- [它是什么](#它是什么)
- [设计决策](#设计决策)
- [速度](#速度)
- [License](#license)
- [术语对照](#术语对照)

## 它是什么

### native-video-inference 是什么？

它把三条视频生成 pipeline（MiniMax H3、Wan 2.1 I2V-14B-720P 和 LTX 2.5 distilled）移植到了 Rust 和 CUDA，另有一个与模型无关的核心 [`core_infra/`](../core_infra/README.md)，负责 checkpoint 加载、启动调度和服务。Wan 和 LTX 在 B200 上与官方 PyTorch pipeline 逐字节一致；H3 的数值跟随 SGLang 在固定 commit 上的 H3 实现。核心围绕 cold start 设计：从容器启动到第一个 DiT step，P50 约 6–9 s，期间读取 47–96 GB 权重。

它不是通用引擎：没有能加载任意 Hugging Face 仓库的模型注册表，没有请求调度或 batching，也没有 HTTP API。加一个新模型就是做一次新的移植，见[移植一个模型](porting-a-model.md)。

### 它与 SGLang 和 vLLM-Omni 是什么关系？

它们是通用的服务引擎：支持很多模型和 GPU，有 batching、HTTP API、adapter、量化和更大的并行度。这个仓库做了不同的取舍：模型少，但逐一精确移植，启动过程围绕存储带宽来设计（[设计决策](#设计决策)）。

它建立在它们的工作之上。H3 的数值跟随 SGLang 固定版本的 H3 实现，稀疏 attention 的 router kernel 直接用 SGLang 的、未作修改，attention kernel 用的是 FlashInfer 的（[NOTICE](../licenses/NOTICE.md)）。README 和[速度](#速度)一节拿 SGLang 和 vLLM-Omni 来对比 cold start 和 warm request，是因为它们是服务这些模型的参照。cold start 的很多工作也可以放进它们里面：我们已经在 SGLang Diffusion 的一个分支上试过，并向上游提了 proposal（[SGLang 能启动得这么快吗？](#sglang-或-vllm-omni-能启动得这么快吗)）。

需要这里没有移植的模型、GPU、功能或并行度时，用它们。你的模型已经移植到这里，并且在意 cold start 或与参考实现完全一致的输出时，用这个。

## 设计决策

每个回答都说明我们选了什么、为什么，以及代价是什么。

### 为什么把 cold start 放在第一位？

一个视频请求要独占一整张 GPU 几十秒到几分钟。高峰到来时，我们按需加 GPU；新容器越快处理第一个请求，用户等得越少，我们为还在启动、没在生成的 GPU 付的钱也越少。高峰过后，GPU 就还回去。容量只在使用时才占用：用户等得少，我们花得少，GPU 也不会闲置。

设计从硬件的物理极限出发，并掌控通往这一极限的每一层。我们在 Modal 上测得 cold start 的读取速度稳定在 10–20 GB/s，所以读 47–96 GB 权重应该只需要 5–10 s。其余所有启动工作（CPU 准备、CUDA context、backend plan、conditioning）都必须放进这个时间窗口里，而不是叠加在后面。从第一个 DiT step 开始，一个请求就是几十秒的 GPU 计算；掌控每个算子意味着哪里慢了，我们可以直接到瓶颈所在的那一层去改，而不用绕着框架走。

### 为什么用原生代码而不是 PyTorch？为什么选 Rust？

原生代码提供了上面说的端到端控制：

- **启动。** 原生进程启动时不需要 import、编译或 warm-up request，在任何 CUDA 工作之前就绑定 socket，在 CUDA context 创建之前就开始读权重。每个启动步骤都声明给同一个调度器，由它把读取、CPU 工作、context 创建和 backend plan 重叠起来（[architecture](architecture.md#the-supported-startup-path)）。
- **数值。** 每个操作都是参数已知的已知 kernel，所以能做到与参考实现完全一致，而且每次优化后都还能检查。
- **所有权。** 线程、CUDA context、pinned buffer 和子进程都显式持有、显式 join，取消和出错时也一样。

PyTorch 的算术保持不变：ATen 的 kernel、cuBLAS 参数和随机数生成器都是移植过来的，而不是替换掉（[NOTICE](../licenses/NOTICE.md)）。Rust 内存安全，性能与 C 相当，我们也听说大公司的很多 ML 基础设施是用它写的。一开始选 Rust 是个试验；移植都通过了验收，用起来也顺手。团队之前给纯 C 的 ML 框架 [libnnc](https://libnnc.org/) 做过贡献，端到端的 ownership 比语言本身更重要。

### 为什么每个模型单独移植，而不做通用引擎？

主要是因为我们服务的内容。我们的产品需要几秒的 cold start 和与参考实现完全一致的输出，现有框架都做不到；而且我们的流量集中在三个模型上：Wan、H3 和 LTX。通用引擎要为我们用不到的广度付出代价。

完全一致也更适合单独移植：它需要每个参考实现自己的 kernel、运算顺序、随机数生成器和 CPU 预处理，而共享的抽象会带入它自己的数值。所以每个模型复制自己需要的算子，为一个模型做的改动不会改变另一个模型已验收的数值。核心只保留与模型无关的机制，不含模型名或 shape（[CONTRIBUTING](../CONTRIBUTING.md)）。

代价是覆盖面：每个模型都是一次完整的移植（Wan 在十个请求上各匹配了 499 个 tensor），加一个模型要花时间。通用引擎选的是这个取舍的另一边。

### 为什么要求逐字节一致，而不是允许误差？

没有误差容忍，验收就是比较哈希，不一致就是 [parity ladder](porting-a-model.md#3-climb-the-parity-ladder) 某一级上的 bug，而不是需要主观判断的问题。每次提速改动也用同样的方式检查，所以优化不会悄悄改变输出。并行版本也必须与单卡的 gold 相同，所以一张卡和两张卡的输出逐字节相同。

参考实现是线上部署的或官方的实现（[冻结参考实现](porting-a-model.md#1-freeze-the-reference)）。代价是近似优化，比如量化或舍入方式不同的更快的 attention kernel，不在验收规则之内；而且完全一致只限于一种 GPU 和一套 SDK。

### 为什么没有 warm-up request？

通用引擎服务很多模型和 shape，在运行时编译或调优 kernel，所以要靠 warm-up 来完成这些工作：SGLang 的服务器跑三个 warm-up request（约 96 s），vLLM-Omni 用 `torch.compile` 编译 DiT block 并跑一个。这里模型和 shape 都少，同样的工作在离线时就记录好了。cuDNN attention plan 按精确的 shape 存放，从记录好的 bundle 加载，而不用花几秒钟构建。于是第一个请求就能以 warm 的速度运行：在 H3 上，新进程里第一个请求的去噪阶段耗时 151.62 s，warm 请求是 151.58 s（[测量](#warm-request-耗时与-sglang-和-vllm-omni-相比如何)）。代价是每个 shape 都要一个 bundle：没有记录过的 shape，Wan 每次 cold start 要多花约 2 s 构建 plan。

### 为什么一个容器只处理一个请求、不做 batching？

Batching 在 LLM 的 decode 阶段划算：每一步都很小、受内存带宽限制，多个请求可以共享每一次权重读取。视频 DiT 的一步更像 LLM 的 prefill：Wan 的官方示例约有 75,000 个 token，一个 block 约 120 ms，GPU 已经是计算瓶颈，batching 收益很小。吞吐靠增加容器来扩展。vLLM-Omni 探索过 step 级别的 [diffusion continuous batching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/)；它的文档把这标为实验性功能，并建议稳妥起见一次只处理一个请求。

### 为什么每个请求最多用两张 GPU？

高峰流量靠按需 GPU 来承接。单卡和双卡的容器可以按需启动，但八卡节点一般需要预留，按需等一台可能要半小时左右，所以缩短它的 cold start 在高峰时帮不上忙。

在两张卡以内，每个模型都用能让输出与单卡完全一致的切分方式：Wan 用 CFG parallel，H3 用 context parallel，LTX 用 sequence parallel（[README](../README.zh-CN.md#cold-start)）。存储是 cold start 中最慢的部分，所以第二张卡通过 peer link 从第一张卡复制权重（Wan 32.8 GB 的 DiT 约 40 ms），而不是再读一遍；H3 的两个 rank 各读同一组文件中互不重叠的一半。

### 为什么不用 CUDA graph？

我们的重点是 cold start。对 warm request，标准是与通用引擎持平，而不是超过它们，而 CUDA graph 在这里得不偿失：

- **Wan 和 H3** 的 forward 都很长。一个 Wan forward 约 4.8 s（80 个 forward 共 387 s），每个 block 约 120 ms，CPU 发 launch 的速度远快于 GPU 执行的速度。在我们的实验中，两者去掉 graph 都没有变慢。
- **LTX** 跑 11 个短 forward（stage 1 有 2,093 个视频 token，stage 2 有 8,372 个），所以 graph 对它确实有一点帮助。但 graph 只能重放固定的 shape，每个 shape 都要单独一个：每个 profile 的每个 sampling stage 各要一个，双卡时每张卡分到的 token 数不同，又各要一个。不用 graph，LTX 得到 final latents 的耗时仍与 SGLang 相当（[对比](#warm-request-耗时与-sglang-和-vllm-omni-相比如何)），所以我们去掉了 graph，而不是为每个 shape 都留一个。

LTX 如何改为削减 host 端的工作，以及用和不用 graph 的耗时对比，见 [LTX 的 README](../models/ltx25/README.md#performance)。

## 速度

### cold start 与 SGLang 和 vLLM-Omni 相比如何？

计时从新容器里的第一行用户代码开始（在任何 import 之前），到请求的第一个 DiT forward 开始为止；终点由 DiT 自己 CUDA stream 上的回调记录，不做全设备同步。Modal 分配 GPU 和启动镜像的时间不计在内；第一个 DiT step 之前的所有权重读取、上传和 conditioning 都计在内（[architecture](architecture.md#readiness-has-several-meanings)）。

README 的 [cold start 表](../README.zh-CN.md#cold-start)列出了本仓库每个模型在单卡和双卡上的 cold start，测于 2026 年 11 月 9 日和 10 日：所有 P50 都在 6.0 到 8.7 s 之间，LTX 单卡 6.6 s、双卡 6.5 s。H3 的行合并了两轮、每轮 30 个容器，Wan 和 LTX 的行各是一轮，所以它们的 P90 包含还没缓存 Volume 的主机。每轮 P50 之间会相差 1–2 s。此前有两项研究把它与 SGLang 和 vLLM-Omni 做了对比，设置各不相同。

**Wan 2.1**，2026 年 11 月 1 日（[README](../README.zh-CN.md#与-sglang-和-vllm-omni-对比)）：每个框架都用文档中的服务器命令和默认设置，跑同一个请求，用同一个时钟，在交替进行的多轮新容器中测量。对 SGLang 和 vLLM-Omni，在 `WanTransformer3DModel` 上挂一个 forward pre-hook 来标记第一个 DiT step，并忽略它们的 warm-up forward。这样测得的 P50：这里 12.1 s，vLLM-Omni 0.30.0 为 140 s，SGLang 0.5.22 为 188 s。它们的大部分时间花在带 warm-up 的服务器启动上（[这里为什么没有](#为什么没有-warm-up-request)），而且两者都读 Diffusers checkpoint，其中 DiT 和 text encoder 以 FP32 存储（共 90 GB）；这里读的是从官方 checkpoint 一次性准备好的 BF16 布局（47 GB）。它们的离线 Python API 比服务器启动得快：更早的一次测量（版本和设置都更旧）中，P50 分别是 vLLM-Omni 41.9 s、SGLang 0.5.21 75.9 s，这里 8.9 s（[历史对比](fast-cold-start.md#historical-comparison-wan-21-i2v-14b-720p-on-one-b200)）。

**MiniMax H3**，2026 年 10 月 21 日：十种配置，每种在三个 Modal region 各跑 50 个冷容器（共 1,500 个），FL2VA，768×768，124 帧，28 步，单张 B200。计时从云端 handler 的第一条指令开始，到进入第一个 DiT forward 为止。SGLang 和 vLLM-Omni 的所有配置都关闭了 warm-up 和 `torch.compile`，所以它们的时间是 import、引擎启动、加载和第一个请求的准备：

| 配置 | P50 | P90 |
| --- | ---: | ---: |
| native-video-inference | 7.21 s | 19.88 s |
| SGLang `77ba54f`，短启动加直接加载到 GPU（它最快的配置） | 64.70 s | 119.31 s |
| SGLang，推荐配置 | 65.40 s | 121.16 s |
| vLLM-Omni `83afc05`，离线 API 默认设置（它最快的配置） | 71.50 s | 126.03 s |
| vLLM-Omni，推荐的 HTTP 服务器，4 个 loader 线程 | 73.98 s | 124.11 s |

SGLang 和 vLLM-Omni 全部九种配置的中位数都在 64.70 到 75.00 s 之间。主机 CPU 没有固定（ap-south 对所有引擎都把容器分到了较慢的主机上），缓存没有清空，这里跑的是一个早于本仓库的 build。[cold start 博文](blog-cold-start.md#appendix-a-how-we-measured)里有完整的测量约定、全部十种配置，以及每个引擎的时间花在了哪里。

两项研究中 SGLang 的数字差别很大（188 s 对 65 s），主要是因为 Wan 那项用的是服务器默认设置（包括 warm-up），而 H3 那项关掉了 warm-up。在两项研究中，差距更多反映的是[设计决策](#设计决策)，而不是 kernel 的速度。在 H3 研究中，Python 引擎在读第一个权重之前要花 20 到 31 s 做 import 和初始化，然后逐个加载组件；而这里的 worker 半秒内就开始读取，并行加载，读得也更少（约 95 GB，而 checkpoint 里的 DiT 和 text encoder 共 133 GB）。

### SGLang 或 vLLM-Omni 能启动得这么快吗？

很多做法并不依赖 Rust。我们在 SGLang Diffusion main 的一个分支上试了本仓库 cold start 背后的思路，在单张 B200 上服务 H3：worker 在 launcher 解析完参数之前就启动；helper 进程在 Python import 的同时读取 checkpoint；直接加载到 GPU；把 bytecode 和 kernel cache 打进镜像。2026 年 11 月 5 日，每组 50 个容器在同一轮中测量，新启动的服务器到达第一个 DiT step 的中位数：快主机上从 79.0 s 降到 20.1 s，两类较慢的主机上分别从 152.9 s 降到 31.9 s、从 161.6 s 降到 42.8 s。完整的 28 步运行生成的帧完全相同，每步仍是 1.29 s。那一轮用的是更新的 main，与上面的研究分开进行，两者不可比。

在单个 Python 进程里，有两件事更难：并行加载组件（因为 GIL 和进程级的加载状态），以及在加载权重的同时准备第一个请求（因为服务器要等就绪后才接收请求）。[proposal](sglang-cold-start-proposal.md) 把这个分支拆成 13 个小 PR 和一个 cookbook 页面，新的启动功能默认关闭，不改变数值，并列出了希望 SGLang 维护者先回答的问题。它是一份提案，不是完成的补丁集；我们很乐意按 SGLang 团队希望的方式贡献，也同样乐意帮助 vLLM-Omni 团队。

### 离开 Modal 还能有同样的 cold start 吗？

核心不依赖 Modal。按 [build.md](build.md) 在 Linux x86-64 和 B200 上构建，把固定版本的 cuBLAS 和 cuDNN 库放到 loader 路径上，然后从任意 Python 3.12 进程运行 `NativeService`；权重根目录就是普通目录，本地 NVMe、镜像层或网络文件系统都可以。这些机制不是 Modal 专属的，但数字取决于存储：几秒的 cold start 需要约 10 GB/s 的读取速度。读取路径上的结论，比如 buffered read 到 5 GB/s 左右就上不去、很多小文件比少数大文件读得快，都是在 Modal Volume 上测得的（[从 Modal Volume 读取权重](fast-cold-start.md#reading-weights-from-a-modal-volume)）。在其他存储上，用 `NATIVE_STARTUP_TRACE_DIR`（以及 `NATIVE_STARTUP_TRACE_LEVEL=io`）记录一次 cold start，再按同一份指南调整 `NATIVE_LOAD_*` 读取设置。

平台自身的启动同样重要。2026 年 10 月 19 日和 20 日测量的四个 B200 平台中，Modal 和 fal 几秒内就能启动新容器；Together AI 会先把权重 volume 复制到每个新节点，Nebius 每个任务都要启动一台虚拟机，所以在这两个平台上，光是平台本身就要几十秒到几分钟（[底层平台](blog-cold-start.md#appendix-b-the-platform-underneath)）。

### warm request 耗时与 SGLang 和 vLLM-Omni 相比如何？

这些对比不是为了比谁更快。我们的重点是 cold start；对 warm request，这些数字是为了说明这里的推理性能与 SGLang、vLLM-Omni 处在同一水平，同样可用。

一个请求的耗时在不同的 B200 主机之间会相差约 2%，所以下面每组对比都在同一个容器、同一张 GPU 上跑各个引擎。

**MiniMax H3。** FL2VA，1376×768，124 帧，49 次 DiT forward（NFE 49），seed 42，2026 年 10 月 22 日测量。十个 B200 容器各自跑所有引擎，引擎的先后顺序在容器之间做了平衡：每个引擎先跑两个 warm-up request，再跑一个计时请求，从输入处理计到保存 MP4。各引擎的 attention 一致，CUDA graph、`torch.compile` 和 step cache 全部关闭。前两列数字是十个 warm request 的均值，最后一列是 30 个新进程的均值：

| Attention | 引擎 | Warm request | Warm 去噪 | 新进程中的去噪 |
| --- | --- | ---: | ---: | ---: |
| cuDNN SDPA | native-video-inference | 154.68 s | 151.58 s | 151.62 s |
| cuDNN SDPA | SGLang `ad7a9b5` | 156.98 s | 153.78 s | 154.98 s |
| cuDNN SDPA | vLLM-Omni `7cbc7e4`（vLLM 0.31.0） | 155.99 s | 152.63 s | 153.25 s |
| cuDNN SDPA | PyTorch eager，Diffusers `48cdbae` | 200.87 s | 193.83 s | 194.22 s |
| SubBlock | native-video-inference | 115.03 s | 111.94 s | 112.07 s |
| SubBlock | SGLang `ad7a9b5` | 117.73 s | 114.54 s | 118.32 s |

按 GPU 配对，用 cuDNN attention 时，这里的 warm request 比 SGLang 少用 1.47% 的时间（95% 区间 1.36–1.55%），比 vLLM-Omni 少 0.84%（0.72–0.94%），比 PyTorch eager 少 22.99%；用 SubBlock 时比 SGLang 少 2.30%。十组配对中的每一组都是这里更快。SGLang 和 vLLM-Omni 与这里相差不到 1.5%；大的差距在 eager 运行的 Diffusers pipeline 那里。我们和 SGLang 的 sampler 输出在每个请求中都逐比特相同；vLLM-Omni 和 PyTorch eager 的则不同，所以这两项只是延迟对比。在固定的版本上，vLLM-Omni 没有能直接用于原版模型的 SubBlock 类稀疏配置，所以只跑了 cuDNN attention。这只是单卡上的一种 FL2VA 请求 shape，不含 Ref2VA，测的是这个 H3 移植在 10 月 22 日的 build，早于本仓库；此后仓库里的代码没有重新测过。

**Wan 2.1**，官方示例，与 SGLang 0.5.22 交替运行（[README](../README.zh-CN.md#与-sglang-和-vllm-omni-对比)）：

| 请求 | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| 一张 B200：整个请求 | 390.3–390.4 s | 398.2 s |
| 一张 B200：40 步 | 387.1–387.2 s | 391.3–391.4 s |
| 两张 B200：40 步 | 200.97–200.99 s | 200.91–201.02 s |

两者很接近。SGLang 也把无条件的 CFG pass 放在第二张卡上，两张 B200 时 40 步耗时相同。单卡时，这里的 40 步快 1%，因为两个 CFG pass 共享了 block 0 的部分计算；VAE decode 这里用 3.2 s，SGLang 用 5.4–5.8 s。

**LTX 2.5**，T2V，1472×832，与未作修改的 SGLang 0.5.22 在同一个 B200:2 容器中交替运行，2026 年 11 月 9 日测量，prompt、seed 和尺寸都相同（[performance](../models/ltx25/README.md#performance)）。每个 warm request 从请求开始计到 final latents：SGLang 取它自己日志里 decode 之前各阶段耗时之和，这里用 `ltx25 infer` 加 `LTX25_REPEAT=3`。两边都不计 decode：

| 请求 | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| 一张 B200，49 帧 | 1.92–2.22 s | 2.57–2.61 s |
| 一张 B200，57 帧 | 2.19–2.23 s | 2.65–2.83 s |

没有双卡对比。双卡时 SGLang 会把 latent 帧数向上取整到 GPU 数的倍数，49 帧变成 57 帧；它在那个容器里的双卡耗时为 3.8–3.9 s，比它自己的单卡还慢，说明我们的 SGLang 双卡配置可能不对。这里两张 B200 跑 57 帧用了 1.6–2.2 s。

SGLang 的输出不是官方 pipeline 的输出：它用确定性的 Euler 而不是 Euler ancestral 采样，噪声也不同；默认情况下它用 VAE 对 latents 做一次 decode，而不是跑 LTX 2.5 的 diffusion video decoder。那个 decoder 需要 NATTEN，而 NATTEN 没有适配 SGLang 所用 torch 2.14 的 wheel；没有它，SGLang 会退回 `flex_attention`，一个请求约需 110 s。

## License

### 能商用吗？

可以，但有一个条件。核心、我们对 Wan 2.1 的贡献以及工具以 source-available 方式在[社区许可证](../LICENSE)下提供，允许商业和非商业使用。如果你的商业产品和服务年收入超过 1000 万美元，无论用途为何，使用本软件前都必须事先获得 Creatify Lab, Inc. 单独的书面授权（[第 2.2 条](../LICENSE)）。请联系 support@creatify.ai；仅发送邮件本身并不构成授权。它不是 OSI 认可的开源许可证。`models/minimax_h3_official` 适用 MiniMax H3 Community License，`models/ltx25` 适用 LTX-2.x Community License。运行模型还需要你自己获得相应权重的许可（[README](../README.zh-CN.md#许可证)）。本节为概述，如有出入，以英文的 [LICENSE](../LICENSE) 为准。

## 术语对照

| 如果你熟悉 | 在这里对应的是 |
| --- | --- |
| `vllm serve`、`sglang serve` | 每个模型的常驻程序，例如 `wan21-server serve <socket>`，由 `core_infra.runtime.service.NativeService` 管理。它通过 Unix socket 接收 JSON-line 命令；没有 HTTP 服务器。 |
| `DiffusionPipeline.__call__` | 在 Rust 中显式写出的运算顺序：`pipeline.rs`（Wan）、`request.rs`（LTX）、`pipeline/`（H3）。 |
| 模型的 `load_weights()` | 一份组件声明：tensor 名称和运行时 dtype，由核心的 bounded reader 从固定了 SHA 的 checkpoint manifest 中原地读取。 |
| `torch.compile`、Inductor | 运行时不编译。Wan，以及 H3 的 text encoder 和 DiT，都是逐个算子写的。LTX 和 H3 的 VAE 重放从参考实现一次性录制的程序：ATen 数据流图，或带 Triton CUBIN 的 Inductor launch 序列。 |
| Warm-up request、CUDA graph capture | 没有空跑的请求，也没有 CUDA graph。cuDNN plan 从记录好的 bundle 加载，每个 forward 都以 eager 方式 launch（[为什么](#为什么不用-cuda-graph)）。 |
| Attention backend 参数 | 每个模型固定：参考实现使用的 cuDNN SDPA 引擎（Wan、LTX），FlashInfer 的 SM100 block-sparse kernel（H3 默认），NATTEN（LTX 的 video decoder）。 |
| `--tp`、Ulysses 和 ring 并行度、CFG parallel | `--gpus 2`，每个模型一种策略。见[为什么最多两张](#为什么每个请求最多用两张-gpu)。 |
| `atol`、`rtol` | 没有。每个记录下来的 tensor 都必须与参考实现的 SHA-256 相同。 |
