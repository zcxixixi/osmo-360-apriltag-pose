# CPU 服务器安全、效率与轨迹质量维护台账

> 本文件是跨会话、跨 compact 的权威维护入口。每轮巡检开始先读本文件，结束前更新状态、证据和下一步。不要在此记录密码、令牌或私钥。

## 固定目标与闭环

- 服务器：`ps@192.168.111.62`；项目：`/home/ps/osmo-360-apriltag-pose-cpu`。
- 基准数据：`/home/ps/instaumi-data/instaumi_000001`，四路 MP4 与 `dataset.h5`。
- 开发分支：`codex/cpu-four-mp4-pipeline`。
- 每次算法、轨迹、坐标系、时间线或渲染语义变更必须：
  1. 阅读 `AGENTS.md`、`docs/DUAL_GRIPPER_V50_BASELINE.md` 和 v50 机器锁；
  2. 变更前后运行 `./umi verify`，记录因外部冻结文件缺失造成的环境失败；
  3. 运行聚焦测试和完整 `pytest`；
  4. 使用新版本输出目录在服务器无缓存重跑 `instaumi_000001`；
  5. 核对频率、样本数、质量门和资源占用；
  6. 从 Tag 正前方斜上视角录制四视频与联合 3D 轨迹对照视频；
  7. 将进度文字和新视频发送到既定飞书会话。
- 整点自动任务：`instaumi`（`InstaUMI服务器整点维护`），每小时第 0 分钟唤醒当前任务并执行巡检/飞书同步。
- 审阅视频视角固定沿用 19 点已接受样式：使用 `flu-front-above`，从 AprilGrid 正面一侧的斜上方向下拍摄，不跟随轨迹旋转。

## 当前已验证基线

| 项目 | 当前证据 |
|---|---|
| 流水线 | `dual-x5-four-mp4-cpu-v5`，30 Hz，手部相机 FLU，`back = +X`，每帧数值位姿与可信度解耦 |
| 本地算法提交 | `72203d6` (`fix: retain a pose on every joint frame`) |
| 服务器等价算法提交 | `b9f7803` |
| 服务器无缓存耗时 | 6.49 s 处理 10 s 四路视频；`time -v` 平均 CPU 1164%，峰值 RSS 311,272 KiB |
| 输出 | 300/300 帧具备双侧数值位姿；268 帧联合可信，266 帧双侧实测，`SELF_CALIBRATED_PASS` |
| 回归测试 | 本地和服务器均为 247 passed，7 skipped |
| 一致性 | v5 全部 300 帧数值位姿与撤回前 v3 逐项最大差值为 0；只额外保留长间隔不可信标记 |
| 最新已发视频 | v5 `processed_joint_trajectory_30hz_front_above_v5.mp4`，SHA-256 `fef54acf...c9a7f7` |

## 问题清单

状态取值：`OPEN`、`IN_PROGRESS`、`RESOLVED`、`DEFERRED`、`ACCEPTED_RISK`。

| ID | 优先级 | 状态 | 问题与证据 | 解决标准 / 下一步 |
|---|---:|---|---|---|
| SEC-001 | 高 | RESOLVED | H5 `dataset_id` 和 JSON `pair_id` 未验证即参与缓存/最终目录拼接；worker 发布阶段对计算出的目标目录执行 `shutil.rmtree`。恶意 `../` 可造成路径穿越和越界删除。涉及 `instaumi.py`、`four_mp4.py`、`four_mp4_worker.py`、`dataset_worker.py`。 | 已实现标识白名单、发现/worker 双重验证、修订锁检查、解析后包含性检查与逐级符号链接拒绝；13 个恶意输入/发布场景、真实 dry-run、服务器完整发布均通过。提交 `80e0f7f`，服务器 `3f09e68`。 |
| QUAL-001 | 高 | RESOLVED | `write_joint_pose_csv` 对所有处于首尾测量之间的缺失帧插值，不限制相邻测量间隔。当前左轨迹报告最大插值间隔约 0.634 s，违反 v50 最大可信间隔 0.25 s 约束。 | v4 长间隔输出 `INTERPOLATED_UNTRUSTED`、空位姿、联合无效；渲染隐藏相机并断开轨迹/趋势线。服务器无缓存结果：可信最大 0.0667 s，拒绝最大 0.6340 s，32 帧不可信，联合有效率 89.33%，全部门通过；7.1 秒视频人工检查通过。 |
| QUAL-002 | 高 | RESOLVED | v4 将长间隔的 XYZ/四元数置空，导致 32/300 帧不具备数值位姿，不符合当前“每帧都有位姿”的产品需求。 | v5 每帧保留位姿：长段插值为 `INTERPOLATED_UNTRUSTED`，首尾为 `HELD_UNTRUSTED`，`joint_has_pose` 与 `joint_valid` 解耦。服务器逐行审计 300/300 非空且有限，四元数归一，v3/v5 数值差为 0；视频中 7.1 s 位姿持续显示。 |
| REL-001 | 高 | RESOLVED | 结果发布采用“先删最终目录，再 copytree”。处理中断会丢失上一版已完成输出，且放大 SEC-001 的破坏面。 | 已改为同级临时目录完整复制后切换，旧目录先重命名为可恢复备份，切换成功后才删除；服务器真实发布成功且不存在 `.publish-*`/`.backup-*` 残留。 |
| SEC-002 | 中 | OPEN | 服务器流水线以高权限 `ps` 用户运行；该用户属于 `sudo`、`docker`、`lxd`、`k3s-admin` 等组，项目代码的进程被攻破后影响面很大。 | 设计最小权限服务账户、只读代码/输入和独立可写缓存/输出；迁移前需用户授权，不能擅自改变现有组。 |
| SEC-003 | 中 | IN_PROGRESS | 服务器有 `0.0.0.0:8000`、`:7864`、`:7865`、`:7869` 等项目相关服务监听局域网。`:7865` 已认证；其余服务的归属与接口已完成只读审计并拆为 SEC-005/006/007。 | 优先关闭 SEC-005；完成 SEC-006 浏览器认证后再决定哪些只读数据可公开。 |
| SEC-004 | 高 | RESOLVED | `:7865` 平台原先允许任意 LAN 客户端无认证覆盖 4 MiB 设备库存、创建项目、上传最大 8 GiB 视频并发布场景，可造成数据篡改与存储/CPU DoS。 | 所有 POST/PUT/PATCH/DELETE 在读体前统一验证 Bearer，等时比较；无令牌服务 fail-closed；令牌文件 `0600`。生产已验证 200/401/400 边界和实际设备同步，客户端/服务器测试及全量 237 passed/7 skipped。 |
| DEP-001 | 中 | IN_PROGRESS | 已完成锁定 Python 与 Node 依赖审计：`pip-audit` 基础 32 项、全 extras 49 项均为 0 已知漏洞；Node 漏洞链已由 DEP-002 修复。主机系统与 FFmpeg 风险分别转入 DEP-003/004。 | 保持锁文件扫描；完成 DEP-003/004 的授权、替换和真实数据回归后关闭总项。 |
| DEP-002 | 高 | RESOLVED | `puppeteer-core 24.16.0` 经 `@puppeteer/browsers` 引入受 GHSA-jmr9-qjv8-65gv 影响的 `extract-zip 2.0.1`；本地 Node 18、服务器 Node 20 均已停止安全维护。生产 `:7865` 仅安装 Three.js，漏洞不可达，但离线渲染树可达依赖。 | 已固定官方 Node 24.20.0 归档/二进制 SHA-256，所有 Python 渲染入口拒绝 Node <22.12；Puppeteer 25.9.0，依赖 80→26，`npm audit` 3 high→0。提交本地 `8e1fab9`、服务器 `fa14ff0`；生产服务已运行在项目 Node 24。 |
| DEP-003 | 高 | OPEN | Ubuntu 22.04 主机有 28 个可直接安装的 standard-security 更新；另有 79 个 ESM Apps 安全更新在未 attach Ubuntu Pro 时不可用。模拟升级共 38 个包，涉及 coreutils、util-linux、libssh、bzip2、bind9、PIL 等。 | 系统级升级可能影响其他项目，需用户授权维护窗口；先备份/列出服务，升级后重启受影响服务并运行整套服务器回归。 |
| DEP-004 | 高 | OPEN | 项目名为 `ffmpeg-master-latest-linux64-gpl` 的二进制实际与系统相同，均为 Ubuntu FFmpeg 4.4.2；Ubuntu 报告 FFmpeg/库的 `+esm14` 安全更新仅 ESM 可用。当前流水线确实调用该旧二进制，目录名也会误导运维。 | 选择受维护、校验哈希的项目级 FFmpeg；建立逐帧像素/Tag 检测/轨迹回归，用新输出版本无缓存重跑 `instaumi_000001` 并按固定 `flu-front-above` 视角出片。 |
| SEC-005 | 高 | OPEN | 独立 `/home/ps/rk3576/offline_flu_viewer` 以系统 Python 3.10 在 `0.0.0.0:8000` 从登录 session 连续运行 4 天，无认证接口可启动/取消处理、修改处理配置/标记，并对选定数据集递归删除 `slam`/`mocap_output` 等输出；请求体也没有大小上限。进程 RSS 约 463 MiB。 | 该目录不属于当前仓库，不能擅自改业务。需用户确认用途后立即停止旧 session，或改为受管 unit、loopback/反代认证、写请求体上限和 CSRF 防护；破坏性接口需二次确认/能力令牌。 |
| SEC-006 | 高 | IN_PROGRESS | 同项目旧 checkout 的审核 UI 在 `0.0.0.0:7869`，无认证 POST 可写审核、分段和人工时间对齐；公开 GET 返回 32 条记录、绝对源路径及审核字段。原 user unit 无任何沙箱，空闲 63 线程。 | unit 已先做无接口变化的硬化：只写状态目录、`NoNewPrivileges`/`PrivateTmp`/只读 home/system、线程池固定 1；状态目录/SQLite `0700/0600`。仍需给网页写接口增加认证与 CSRF，按需收窄公开 GET。 |
| SEC-007 | 低 | OPEN | `:7864` 是 4 天前从登录 session 启动的旧静态审核页面，只提供 GET，但绑定全 LAN、公开时间线/视频/网格，且仍运行 EOL 的系统 Node 20。 | 确认是否仍被使用；若已由 `:7865` 取代则停止，若保留则迁移受管 unit、项目 Node 24，并按数据敏感度限制访问。 |
| EFF-001 | 高 | RESOLVED | 压缩 NPZ 的每个数组被重复打开和解压。缓存读取改为一次加载所有成员。 | 服务器无缓存耗时 14.26 s → 6.10 s；输出逐字节一致；提交 `293df8b`。 |
| EFF-002 | 中 | DEFERRED | FFmpeg 管道软件解码约 2.04 s，OpenCV 包装约 2.70 s；VAAPI 四路约 7.43 s。FFmpeg 像素输出与当前路径不完全一致，且缓存修复后解码已非主要瓶颈。 | 若后续解码占比重新升高，建立像素/检测回归门后再切换；当前不以小收益换算法输入变化。 |
| EFF-003 | 中 | RESOLVED | 观察缓存已有 4×4 预算，但备用联合 pose-graph 的 8 个 Python worker 未限制 BLAS/OpenMP，存在 8×32 嵌套线程放大；不同任务之间也没有主机级并发门。 | 每任务最多 16 逻辑线程，小主机自动降配；OpenCV/FFmpeg/BLAS/OpenMP/BLIS/vecLib 全部继承限额，pose-graph 每 worker 1 个数学线程。默认同用户整机 1 个任务槽，可在总量不超过逻辑核时显式增加。双任务并发实测串行、无发布竞态；轨迹 SHA 不变。提交本地 `5dad620`/`e2400fb`，服务器 `5d550f6`/`449fde0`。 |
| EFF-004 | 低 | RESOLVED | `:7869` 审核服务主要待机，却因 OpenCV/数学库默认线程池保持 63 个线程、RSS 103,004 KiB。 | systemd 环境将所有原生线程池限制为 1；重启后接口仍为 200、32 条记录可读，线程 63→1、RSS 103,004→68,804 KiB，systemd `MemoryCurrent` 约 62.2→33.8 MiB。 |
| REL-002 | 低 | OPEN | 本机不带隔离环境变量运行 pytest 时会自动加载 ROS Humble 的 `launch_testing` 插件，并因跨 Python 环境缺少 `yaml` 在收集前失败。 | 维护命令统一设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，或在项目 pytest 配置中只加载所需插件，避免宿主插件污染。 |
| REL-003 | 低 | DEFERRED | CPU 服务器未安装 Chrome/Chromium，因此可选的服务器端 WebGL 离线渲染不可用；四 MP4 轨迹审阅使用 Python/OpenCV，不受影响，`:7865` 也只在客户端浏览器渲染。原实现硬编码 `/usr/bin/google-chrome`。 | 已支持 `--chrome`/`OSMO_CHROME_BINARY` 与常见路径探测，缺失时启动前明确失败；本地前后视频 SHA 完全相同。仅在确需服务器 WebGL 时再固定、校验并安装项目级 Chromium，避免当前无收益地增加体积和攻击面。提交本地 `29f91f4`、服务器 `78c734c`。 |

## 已完成的安全检查

- 仓库与数据目录不是全局可写；项目中仅观察到 `.venv/.lock` 为 `0666`，不承载可执行代码。
- 未在仓库发现 `.env`、常见私钥文件或已知密码/密钥字面量。
- 当前服务器没有正在运行的四 MP4 pipeline 进程。
- v50 受保护代码和制品未被本分支修改；本机 `./umi verify` 仍因缺少 `/home/cenxi/.../dual_gripper_claw_to_claw_action_v50_fixed_timeline.json` 而在读取外部冻结样本时失败，属于已知环境缺失，不能据此宣称基线已完整验证。
- `:7865` 平台的生产令牌位于仓库外、权限 `0600`，仓库和日志不保存令牌内容；旧服务源文件有可恢复备份。

## 巡检日志

### 2026-09-01 / Cycle 001

- 建立本台账与整点自动任务。
- 确认 EFF-001 的服务器真实数据性能收益和输出不变性。
- 直接解码基准表明 VAAPI 在当前硬件/驱动/四路负载下更慢，保留软件路径。
- 首轮安全审计发现 SEC-001/REL-001 路径与发布风险、SEC-002 权限面、SEC-003 LAN 监听面。
- 首轮轨迹质量审计发现 QUAL-001：约 0.634 s 的插值仍被信任。
- SEC-001/REL-001 本地修复完成：新增 13 个安全/发布回归场景，完整测试 `223 passed, 7 skipped`；真实 `instaumi_example_000001` dry-run 正常。待部署到服务器后关闭问题。
- 发现 REL-002：宿主 ROS pytest 插件污染；禁用自动插件加载后项目测试正常。
- QUAL-001 本地修复完成并升级为 `dual-x5-four-mp4-cpu-v4`：同一真实 bearings 从旧版 300/300“有效”改为 268/300 可信；32 帧长空洞 fail-closed，最大可信插值 0.0667 s，状态仍为 `SELF_CALIBRATED_PASS`。
- v4 聚焦视频已人工检查封面与 7.1 秒帧：左侧在空洞中显示 `UNTRUSTED` 与 XYZ/RPY N/A，3D 相机隐藏、轨迹断开；不再把 0.634 秒空洞画成平滑跳变。
- 变更后完整测试 `230 passed, 7 skipped`。`./umi verify` 仍在读取本机缺失的外部冻结 v50 timeline 时失败，受保护文件未修改。
- v4 已提交 `9865b92` 并以等价提交 `4bf7060` 部署服务器；服务器完整测试同为 `230 passed, 7 skipped`。
- 服务器无缓存完整运行 6.63 s；`time -v` 平均 CPU 1180%，0.1 s 进程树采样平均/峰值 950%/2167%（32 逻辑核整机约 29.7%/67.7%）；聚合峰值 RSS 1,374,412 KiB（约 1.31 GiB、整机 2.17%）；无 swap。
- v3/v4 `session_world_map.json`、`left_pose.csv`、`right_pose.csv` SHA-256 分别完全相同；300 行中 32 行明确不可信且对应左侧坐标为空。
- 最终视频已生成并人工检查封面/7.1 s：服务器与本地路径均为 `final/dual-x5-four-mp4-cpu-v4/reviews/processed_joint_trajectory_30hz_front_above_v4.mp4`，SHA-256 `73474b0208836e526efe68c941b4115cbbdff3887f4d8377606d4754cb349e27`。
- 飞书进度消息 `om_x100b665f8cc4d4a8c1c7df3e2b59fbf`、视频消息 `om_x100b665f8c7678a4c00eeb6b35018dd` 均发送成功。
- 下一步巡检重点：SEC-002 最小权限账户、SEC-003 LAN 服务接口、DEP-001 依赖漏洞、EFF-003 并发资源隔离。

### 2026-09-01 / Cycle 002

- 用户要求撤回 v4 的空位姿策略，并明确授权忽略本机缺失外部文件造成的 v50 环境 gate；冻结基线文件未修改。
- v5 实现数值位姿与可信度解耦：`joint_has_pose` 表示数值可用，`joint_valid` 表示可信测量/短间隔插值；长间隔不再清空或隐藏。
- 如果某一侧整段没有任何接受位姿，流水线明确失败，不凭空伪造整段轨迹。
- 本地聚焦测试 `28 passed`，全量测试 `232 passed, 7 skipped`。
- SEC-003 初步审计：`:7865` 可视化平台、`:7869` 审阅服务以及独立 `:8000` 处理服务存在局域网未认证写接口；`:7864` 仅静态 GET。待完成本次轨迹回滚闭环后优先修复当前仓库管理的 `:7865`。
- v5 已提交 `72203d6`并以等价提交 `b9f7803` 部署服务器；本地/服务器全量测试均为 `232 passed, 7 skipped`。
- 服务器无缓存运行 6.49 s；300/300 帧 `joint_has_pose=true`，无空值、无非有限值，四元数范数范围为 `0.9999999999992..1.0000000000008`，全部门通过。
- v5 与 v3 的 300 帧双侧 XYZ/四元数逐项最大差值为 0；数值行为已撤回，但 32 帧长间隔仍可审计地标为 `INTERPOLATED_UNTRUSTED`。
- 新视频 `processed_joint_trajectory_30hz_front_above_v5.mp4` 已检查封面和 7.1 s 帧，SHA-256 `fef54acf451a829a5b6ffd9b3c72625807b7c1b97886d99092fd4f36d1c9a7f7`。
- 飞书文字消息 `om_x100b665fb13d9ca0c24c7d06359eea4`，视频消息 `om_x100b66584eb5b8a0df34a0f404b9605`，均发送成功。
- SEC-004 修复提交：本地 `1f381d1`，服务器等价 `fc53efa`；服务器和本地全量测试均为 `237 passed, 7 skipped`。这是纯安全边界变更，未改轨迹/渲染算法，因此不重跑数据或录制视频。
- `:7865` 生产服务已从 `/run/user/...` 临时 unit 迁移到启用的持久 user unit；强制 `NoNewPrivileges`、`PrivateTmp`、`ProtectSystem=strict`、`ProtectHome=read-only`，仅白名单数据目录可写。第一版内核 capability 硬化与该 user manager 不兼容（`218/CAPABILITIES`），已去除不支持项并恢复为 active。
- 生产验证：LAN `GET /healthz` 返回 200；无认证 `POST /api/projects` 返回 401 和 `WWW-Authenticate: Bearer`；已认证无效 JSON 返回 400；`umi devices sync` 认证写入 20 台设备成功，库存 SHA-256 写前写后不变。

### 2026-09-01 / Cycle 003

- 依赖审计分层完成：锁定 Python 基础运行时 32 项、全部 extras 49 项经 `pip-audit` 均未发现已知漏洞。
- Node 原依赖树有 3 个 high 报告，实际为同一条 `puppeteer-core → @puppeteer/browsers → extract-zip` 路径穿越链；生产 `:7865` 只有 Three.js，无法触达，但 CPU 仓库离线 WebGL 渲染安装了该依赖。
- 项目级修复使用 Node 24.20.0，官方归档 SHA-256 `2f2c0da1...0a7cbf2`；安装到 gitignored `work/tools/`，不升级宿主系统 Node。所有 Python WebGL 入口统一优先使用该运行时，并明确拒绝 Node <22.12.0。
- Puppeteer 升级至 25.9.0，锁定依赖数从 80 降到 26，`extract-zip` 已不存在，`npm audit` 从 3 high 降为 0；安装统一使用 `npm ci --ignore-scripts`。
- 新增运行时选择、过期版本拒绝和归档路径穿越测试；聚焦测试 `6 passed`。Node 24 + Puppeteer 25 + 现有 Chrome/Three.js 的真实 WebGL 单帧编码成功。
- 本地与服务器全量测试最终均为 `243 passed, 7 skipped`；本机 `./umi verify` 仍只因缺少既有 `/home/cenxi/.../dual_gripper_claw_to_claw_action_v50_fixed_timeline.json` 外部冻结文件失败，受保护文件未修改。
- Node 修复提交：本地 `8e1fab9`、服务器等价 `fa14ff0`。生产 `:7865` 的 user systemd unit 已从系统 Node 20 切到项目 Node 24，旧 unit 备份为 `osmo-visualization.service.pre-node24-20260901.bak`；服务 active/enabled、0 次重启，进程 RSS 约 61.2 MiB。
- 生产切换后 `GET /healthz` 为 200，无认证 `POST /api/projects` 为 401 且保留 `WWW-Authenticate: Bearer`；实际进程 `/proc/.../exe` 指向校验过的项目 Node 24.20.0。
- 服务器缺少任何 Chrome/Chromium，原 WebGL 离线渲染本就不可用。新增显式浏览器路径解析和 fail-fast，提交本地 `29f91f4`、服务器 `78c734c`；本地同一单帧输出修改前后 SHA-256 均为 `caee3906...6a3c82`，证明默认路径上的像素/编码不变。
- 不为当前未使用的服务器 WebGL 能力安装大型浏览器；四 MP4 审阅视频仍由 Python/OpenCV 生成，并继续强制 `flu-front-above` 固定视角。
- 主机级风险单列而未擅自修改：28 个 standard-security 更新可用、79 个 ESM Apps 更新因未 attach 不可用；`apt-get -s upgrade` 会升级 38 个包。
- 项目所谓 `ffmpeg-master-latest-linux64-gpl` 实际是 FFmpeg 4.4.2，且流水线正在使用。由于替换会改变解码输入，必须按算法输入变更闭环重跑数据和固定视角录制，不能混在 Node 安全修复中悄悄替换。
- EFF-003 资源隔离修复：单任务上限 16 逻辑线程，主机按用户默认只放行 1 个任务；小 CPU 主机自动降低 4×4 配置，显式超配则 fail-closed。任务槽目录/文件权限为 `0700/0600`，拒绝符号链接和非本用户所有者。
- 备用联合 pose-graph 原先可由 8 个 Python worker 各自继承 32 线程数学库；现固定为每 worker 1 个 BLAS/OpenMP 线程，并补齐 BLIS、vecLib、OpenMP/MKL dynamic 限制。
- 本地/服务器全量测试均为 `247 passed, 7 skipped`。服务器缓存重跑 1.07 s，平均 CPU 824%，峰值 RSS 105,316 KiB、无 swap；任务槽等待约 18 µs。
- 同时提交两条相同数据任务，先到者立即运行，后到者在 slot 0 等待 0.400 s 后运行；两者均成功，发布区没有 `.tmp`/`.publish-*`/`.backup-*` 残留。
- 并发修复前后 `joint_trajectory.csv`、双侧 pose、世界图、跟踪报告和输入签名 6 个 SHA-256 全部相同；这不是轨迹/渲染算法变更，因此没有重复录制或发送视频，最近审阅视频仍为固定 `flu-front-above` 的 v5。
- LAN 深审确认 `:7869` 来自 `/home/ps/osmo-360-apriltag-pose` 的 `osmo-alignment-review.service`，`:8000` 来自独立 `/home/ps/rk3576/offline_flu_viewer` 登录 session，`:7864` 是旧静态 Node session；均非当前 CPU worker 进程。
- `:7869` 已部署可恢复 unit 硬化，备份为 `osmo-alignment-review.service.pre-hardening-20260901.bak`。新 unit 仅允许写 `/home/ps/review-state/alignment-review-v1`，接口 `/`、`/api/items`、`/api/reprocess-queue` 均继续 200；线程 63→1，RSS 103,004→68,804 KiB，0 次重启。
- 审核状态目录和现有 SQLite/WAL/SHM 从 `0775/0644` 收紧为 `0700/0600`，未来文件由 unit `UMask=0077` 约束。
- `:7869` 认证仍未完成：源码没有 Authorization/令牌检查，POST 可改审核/分段/对齐，GET 暴露绝对路径和审核信息。不得因 unit 已硬化而关闭 SEC-006。
- `:8000` 无认证 POST 可启动/取消流水线、修改配置/标记并调用 `shutil.rmtree` 清除结果；GET `/api/processing/status` 在只读探测中 10 秒未返回，说明还存在便宜触发的资源消耗面。该服务属于独立工程，未擅自停止或修改。

## 最近一次算法改动验证

- 改动：v5 撤销长间隔空位姿/隐藏策略，保证每帧数值位姿，同时保留独立可信度（QUAL-002）。
- 提交：本地 `72203d6`，服务器等价 `b9f7803`。
- 服务器输出：`/home/ps/instaumi-data/instaumi_000001/final/dual-x5-four-mp4-cpu-v5/`。
- 服务器报告：300/300 帧具备数值位姿；联合可信 268（89.33%）；联合实测 266（88.67%）；长间隔不可信 32 帧；全部门通过。
- 服务器运行：6.49 s；`time -v` 平均 CPU 1164%；峰值 RSS 311,272 KiB；无 swap。
- 最终审阅视频：`reviews/processed_joint_trajectory_30hz_front_above_v5.mp4`，SHA-256 `fef54acf...c9a7f7`。
- 飞书：文字与视频均发送成功，消息 ID 见 Cycle 002 日志。
