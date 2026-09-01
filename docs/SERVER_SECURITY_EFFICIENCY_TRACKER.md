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
- 审阅视频以飞书 CLI 19:03 原视频 `processed_joint_trajectory_30hz_front_above_v1.mp4` 为唯一模板：世界坐标保持原生 `Tag Map`，手部相机仅作为 `Camera FLU` 子坐标且 `back = +X`，渲染必须使用 `tag-map-front-above`。机位始终位于两个 AprilGrid 正面一侧，从斜上方向下拍且镜头面朝网格，两块网格在画面中横向排列；不跟随轨迹、不自动旋转，禁止用会重表达世界坐标的 `flu-front-above` 代替。

## 当前已验证基线

| 项目 | 当前证据 |
|---|---|
| 流水线 | `dual-x5-four-mp4-cpu-v6`，30 Hz，手部相机 FLU，`back = +X`，每帧数值位姿与可信度解耦，项目 FFmpeg 9.0.1 哈希锁定 |
| 本地算法提交 | `72203d6` (`fix: retain a pose on every joint frame`) |
| 服务器等价算法提交 | `b9f7803` |
| 运行时提交 | 本地 `0e0a2d4`，服务器等价 `45802c5` |
| 服务器无缓存耗时 | 无竞争 v5 基线 6.49 s；v6 在两条独立 ORB-SLAM 各占约 335% CPU、load 18 时为 12.21 s，峰值 RSS 310,236 KiB |
| 输出 | 300/300 帧具备双侧数值位姿；268 帧联合可信，266 帧双侧实测，`SELF_CALIBRATED_PASS` |
| 回归测试 | 本地为 270 passed，7 skipped；服务器待部署本轮兼容入口与默认视角后复测；Bandit 0 high/0 medium |
| 一致性 | v6/v5 `joint_trajectory.csv` SHA-256 同为 `52c3e192...b82ed`；视角纠正只重新渲染，未改轨迹文件 |
| 最新已发视频 | v6 `processed_joint_trajectory_30hz_tag_map_front_above_v6.mp4`，固定 `tag-map-front-above`，SHA-256 `ba80d1d7...127238` |

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
| DEP-004 | 高 | RESOLVED | 旧兼容目录名为 `ffmpeg-master-latest-linux64-gpl`，实际是 Ubuntu FFmpeg 4.4.2。主像素解码虽已由 OpenCV 内置 FFmpeg 8.1.2 完成，外部旧版仍参与 MP4 探测/音频/审阅编码。 | 已从官方签名源构建项目 FFmpeg 9.0.1，PGP 指纹 `FCF9...58D8`；源归档、离线归档和二进制 SHA 均锁定，关闭网络协议，拒绝旧版/哈希篡改/可写文件/符号链接；旧兼容入口也改为受校验运行时的包装器。四路共 2396 帧像素、同一视角编码回归的 299 帧、轨迹 SHA 均完全一致；服务器 v6 无缓存通过。 |
| SEC-005 | 高 | OPEN | 独立 `/home/ps/rk3576/offline_flu_viewer` 以系统 Python 3.10 在 `0.0.0.0:8000` 从登录 session 连续运行 4 天，无认证接口可启动/取消处理、修改处理配置/标记，并对选定数据集递归删除 `slam`/`mocap_output` 等输出；请求体也没有大小上限。进程 RSS 约 463 MiB。 | 该目录不属于当前仓库，不能擅自改业务。需用户确认用途后立即停止旧 session，或改为受管 unit、loopback/反代认证、写请求体上限和 CSRF 防护；破坏性接口需二次确认/能力令牌。 |
| SEC-006 | 高 | IN_PROGRESS | 同项目旧 checkout 的审核 UI 在 `0.0.0.0:7869`，无认证 POST 可写审核、分段和人工时间对齐；公开 GET 返回 32 条记录、绝对源路径及审核字段。原 user unit 无任何沙箱，空闲 63 线程。 | unit 已先做无接口变化的硬化：只写状态目录、`NoNewPrivileges`/`PrivateTmp`/只读 home/system、线程池固定 1；状态目录/SQLite `0700/0600`。仍需给网页写接口增加认证与 CSRF，按需收窄公开 GET。 |
| SEC-007 | 低 | OPEN | `:7864` 是 4 天前从登录 session 启动的旧静态审核页面，只提供 GET，但绑定全 LAN、公开时间线/视频/网格，且仍运行 EOL 的系统 Node 20。 | 确认是否仍被使用；若已由 `:7865` 取代则停止，若保留则迁移受管 unit、项目 Node 24，并按数据敏感度限制访问。 |
| SEC-008 | 中 | RESOLVED | Bandit 首轮扫描 32,217 行代码为 0 high/8 medium；实际输入面包括任意 urllib 协议、Node 下载来源/大小、XML 实体、PyTorch pickle checkpoint 和旧 INSV 共享临时目录。四 MP4 任务槽的 `/tmp` 告警已有同用户属主、`0700/0600`、拒绝符号链接和 `O_NOFOLLOW` 防护，属于已验证误报。 | HTTP 客户端现只接受无凭据/片段/反斜线的明确 HTTP(S)；Node 只接受 `nodejs.org` HTTPS、重定向同源、128 MiB 上限且继续校验 SHA；checkpoint 使用 `weights_only=True`；URDF 使用 `defusedxml`；旧 INSV 默认 scratch 进入各节点 checkout 的 `work/` 并校验属主/符号链接。复扫 32,308 行为 0 high/0 medium，提交本地 `bbfe7ac`、服务器 `8aa6fba`。 |
| SEC-009 | 高 | RESOLVED | `:7865` 虽已保护写接口，但设备库存 GET 仍向任意 LAN 客户端公开 20 台相机的序列号、固件和左右/Tag 分配；上传 scene/video 会在内容验证前替换正式文件，无效上传可破坏仍标为 ready 的已发布项目。服务也缺少 CSP/`nosniff`，历史 50 个项目目录和 167 个资产为 `0775/0644` 或 `0664`。Node 对 `Expect: 100-continue` 默认先放行正文，未认证客户端可在 401 前开始发送超大请求。 | 设备库存读写均要求 Bearer；scene 与 MP4（ISO BMFF `ftyp`）在唯一 `0600` 临时文件中验证后才原子替换，元数据同样原子发布；拒绝数据根/项目/资产符号链接和非服务属主；返回 CSP、`nosniff`、拒绝嵌入/权限策略。生产启动将 50 个目录、168 个文件收紧到 `0700/0600`；`checkContinue` 在 100 前鉴权，畸形 URL 返回 400 且不刷内部错误。提交本地 `bdacd36`/`86714b1`、服务器 `fa1d4a2`/`554eeae`。 |
| SEC-010 | 中 | OPEN | `:7865` 仍允许任意 LAN 客户端列出 50 个可视化项目，并直接读取每个 ready 项目的 scene、完整轨迹 timeline 和相机视频。这是当前“点击链接直接审阅”的既有设计，但视频/轨迹可能属于敏感采集数据，不能因为写接口已认证就视为完整访问控制。 | 需要明确产品策略：若只供王浩/授权人员查看，增加独立读会话或短期签名 capability，避免令牌进入 URL/日志；若确认局域网公开是需求，则记录数据分级、网络边界和接受风险。未经选择不擅自让现有 50 个审阅链接失效。 |
| SEC-011 | 高 | RESOLVED | CPython `urllib` 的默认 302 处理会把原请求的 `Authorization` 原样转发到跨域 Location；本地双 HTTP 服务已复现 Bearer 到达第二个域。设备同步和可视化 JSON API 使用默认 `urlopen`，上传客户端还信任创建响应中的绝对 `links`，错误配置或被攻破的平台可窃取写令牌。 | 所有带认证的 urllib API 请求使用拒绝重定向 opener，302 直接报错且第二服务未收到请求；上传 URL 不再读取响应中的绝对 links，而是校验 `[a-z0-9-]{1,64}` 项目 ID 后从用户配置的 server 本地构造四个端点。新增真实双服务泄露回归、恶意 links 和路径 ID 测试；本地 `5ae4c84`、服务器 `2c7f2d7`。 |
| EFF-001 | 高 | RESOLVED | 压缩 NPZ 的每个数组被重复打开和解压。缓存读取改为一次加载所有成员。 | 服务器无缓存耗时 14.26 s → 6.10 s；输出逐字节一致；提交 `293df8b`。 |
| EFF-002 | 中 | DEFERRED | FFmpeg 管道软件解码约 2.04 s，OpenCV 包装约 2.70 s；VAAPI 四路约 7.43 s。主路径 OpenCV 4.14 已内置 FFmpeg 8.1.2。外部 FFprobe 9.0.1 的 800 次探测由 23.44 s 降至 4.34 s，但每任务仅探测四个文件，端到端收益很小。 | 若后续解码占比重新升高，建立完整像素/Tag/轨迹回归门后再调整主 `cv2.VideoCapture`；当前继续使用已验证的软件解码和线程上限。 |
| EFF-003 | 中 | RESOLVED | 观察缓存已有 4×4 预算，但备用联合 pose-graph 的 8 个 Python worker 未限制 BLAS/OpenMP，存在 8×32 嵌套线程放大；不同任务之间也没有主机级并发门。 | 每任务最多 16 逻辑线程，小主机自动降配；OpenCV/FFmpeg/BLAS/OpenMP/BLIS/vecLib 全部继承限额，pose-graph 每 worker 1 个数学线程。默认同用户整机 1 个任务槽，可在总量不超过逻辑核时显式增加。双任务并发实测串行、无发布竞态；轨迹 SHA 不变。提交本地 `5dad620`/`e2400fb`，服务器 `5d550f6`/`449fde0`。 |
| EFF-004 | 低 | RESOLVED | `:7869` 审核服务主要待机，却因 OpenCV/数学库默认线程池保持 63 个线程、RSS 103,004 KiB。 | systemd 环境将所有原生线程池限制为 1；重启后接口仍为 200、32 条记录可读，线程 63→1、RSS 103,004→68,804 KiB，systemd `MemoryCurrent` 约 62.2→33.8 MiB。 |
| EFF-005 | 低 | OPEN | `:7869 /api/items` 会在读取 32 条记录时同步扫描/汇总数据；本轮冷探测 5 s 超时，随后稳定返回 200 但耗时 3.06 s。请求期间进程为 2 线程、RSS 80,092 KiB，`MemoryCurrent` 约 123 MiB，明显高于空闲基线；根页面仅 0.0006 s。 | 先在其可维护源码迁移后做端点分段计时、结果缓存和失效策略；不得为了提速放宽已设置的原生线程池或认证边界。与 SEC-006 一起处理。 |
| EFF-006 | 低 | OPEN | `:7865` 目前保存 50 个项目、总计 4.2 GiB（文件字节 4,486,238,512），单项目中位约 69.6 MiB、最大约 447.5 MiB；没有保留期限、容量配额或清理工作流。当前磁盘仍有 1.4 TiB，项目列表 50 次实测中位 3.55 ms、p95 8.87 ms，暂非实时瓶颈。 | 在不自动删除正式审阅数据的前提下增加只读容量告警和按项目创建时间/最后访问时间的清理候选报告；真正删除必须经用户确认并提供可恢复窗口。 |
| REL-002 | 低 | RESOLVED | 本机不带隔离环境变量运行 pytest 时会自动加载 ROS Humble 的 `launch_testing` 插件，并因跨 Python 环境缺少 `yaml` 在收集前失败。 | `pyproject.toml` 明确屏蔽 7 个宿主 ROS/ament pytest entry point；本地和服务器均以普通 `pytest -q` 得到 `247 passed, 7 skipped`。提交本地 `598175b`、服务器 `61c1d02`。 |
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
- 不为当前未使用的服务器 WebGL 能力安装大型浏览器；四 MP4 审阅视频仍由 Python/OpenCV 生成。此轮当时沿用了 `flu-front-above`；Cycle 006 读取 19:03 原视频后确认该预设并非用户指定模板，现已纠正为 `tag-map-front-above`。
- 主机级风险单列而未擅自修改：28 个 standard-security 更新可用、79 个 ESM Apps 更新因未 attach 不可用；`apt-get -s upgrade` 会升级 38 个包。
- 项目所谓 `ffmpeg-master-latest-linux64-gpl` 实际是 FFmpeg 4.4.2，且流水线正在使用。由于替换会改变解码输入，必须按算法输入变更闭环重跑数据和固定视角录制，不能混在 Node 安全修复中悄悄替换。
- EFF-003 资源隔离修复：单任务上限 16 逻辑线程，主机按用户默认只放行 1 个任务；小 CPU 主机自动降低 4×4 配置，显式超配则 fail-closed。任务槽目录/文件权限为 `0700/0600`，拒绝符号链接和非本用户所有者。
- 备用联合 pose-graph 原先可由 8 个 Python worker 各自继承 32 线程数学库；现固定为每 worker 1 个 BLAS/OpenMP 线程，并补齐 BLIS、vecLib、OpenMP/MKL dynamic 限制。
- 本地/服务器全量测试均为 `247 passed, 7 skipped`。服务器缓存重跑 1.07 s，平均 CPU 824%，峰值 RSS 105,316 KiB、无 swap；任务槽等待约 18 µs。
- 同时提交两条相同数据任务，先到者立即运行，后到者在 slot 0 等待 0.400 s 后运行；两者均成功，发布区没有 `.tmp`/`.publish-*`/`.backup-*` 残留。
- 并发修复前后 `joint_trajectory.csv`、双侧 pose、世界图、跟踪报告和输入签名 6 个 SHA-256 全部相同；这不是轨迹/渲染算法变更，因此当时没有重复录制或发送视频。该轮引用的 `flu-front-above` 后经 Cycle 006 核对确认为错误模板。
- LAN 深审确认 `:7869` 来自 `/home/ps/osmo-360-apriltag-pose` 的 `osmo-alignment-review.service`，`:8000` 来自独立 `/home/ps/rk3576/offline_flu_viewer` 登录 session，`:7864` 是旧静态 Node session；均非当前 CPU worker 进程。
- `:7869` 已部署可恢复 unit 硬化，备份为 `osmo-alignment-review.service.pre-hardening-20260901.bak`。新 unit 仅允许写 `/home/ps/review-state/alignment-review-v1`，接口 `/`、`/api/items`、`/api/reprocess-queue` 均继续 200；线程 63→1，RSS 103,004→68,804 KiB，0 次重启。
- 审核状态目录和现有 SQLite/WAL/SHM 从 `0775/0644` 收紧为 `0700/0600`，未来文件由 unit `UMask=0077` 约束。
- `:7869` 认证仍未完成：源码没有 Authorization/令牌检查，POST 可改审核/分段/对齐，GET 暴露绝对路径和审核信息。不得因 unit 已硬化而关闭 SEC-006。
- `:8000` 无认证 POST 可启动/取消流水线、修改配置/标记并调用 `shutil.rmtree` 清除结果；GET `/api/processing/status` 在只读探测中 10 秒未返回，说明还存在便宜触发的资源消耗面。该服务属于独立工程，未擅自停止或修改。
- REL-002 已关闭：项目 pytest 配置在插件自动加载阶段屏蔽宿主 ROS/ament 的 `launch_testing`、`launch_ros` 和 5 个 ament lint entry point；不设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD` 时，本地/服务器仍稳定通过 `247 passed, 7 skipped`。
- 用户再次确认审阅机位不可变化：严格沿用 19 点版本，从两个 AprilGrid 正面一侧的斜上方向下拍，镜头面朝 AprilGrid；已同时写入固定目标和整点自动任务约束。
- Bandit 对 `src/`、`tools/` 的首轮结果为 0 high/8 medium；修复真实输入面并给经过属主/权限/防符号链接验证的四 MP4 `/tmp` fallback 添加理由后，复扫 32,308 行为 0 high/0 medium。低级告警 100 项留作后续按可达性分批审阅，不能把静态告警数量直接当成漏洞数量。
- 安全补丁提交本地 `bbfe7ac`、服务器 `8aa6fba`；本地和服务器完整测试均为 `260 passed, 7 skipped`。锁定 Python 基础/全 extras 依赖再次经 `pip-audit` 验证均为 0 已知漏洞。
- 服务器第一次 `uv sync` 因外部网络下载 SciPy/kiwisolver 长时间无进展而中止；仅终止本轮启动的两个同步进程。随后从本地锁定 wheel/cache 离线补齐并成功执行 `uv sync --frozen --extra test --offline`，服务器最终版本为 `defusedxml 0.7.1`、`kiwisolver 1.5.0`、`scipy 1.18.0`，普通 `./.venv/bin/pytest` 已恢复并通过。
- 安全补丁未修改四 MP4 检测、位姿、时间线或审阅渲染语义；服务器 `joint_trajectory.csv` SHA-256 仍为 `52c3e192...b82ed`，v5 视频仍为 `fef54acf...c9a7f7`，因此没有重复处理数据或发送旧视频。
- `:7865`、`:7869` unit 均保持 active，`:7865 /healthz` 正常。`:7869 /api/items` 冷探测出现一次 5 s 超时，复测 3.06 s 返回 200，新增 EFF-005 跟踪，未擅自改变独立审核服务业务代码。
- 21:00 飞书文字进度发送成功，消息 ID `om_x100b66588b3788a0dfed92c99568203`；本轮无新视频。

### 2026-09-01 / Cycle 004

- 深审 `:7865` 生产平台的读边界、上传原子性、路径和浏览器响应头。修复前未认证 GET 可读取 20 台 X5 的序列号、固件及角色/Tag 分配；同时公开 50 个项目。
- 发现已发布项目的 scene 上传先覆盖正式文件、后检查两个关键字符串；无效但已认证的上传会留下坏 scene，同时旧元数据仍可能保持 `ready`。video 也只检查非空，任意内容可作为 `video/mp4` 发布。
- 本地提交 `bdacd36`、服务器等价提交 `fa1d4a2`：设备库存 GET 改为 Bearer；scene/MP4 先在随机唯一临时文件验证再原子替换；MP4 至少验证 ISO BMFF `ftyp`；项目元数据和设备库存改为 `0600` 临时文件原子发布。
- 服务启动与读取现在拒绝数据根、项目目录、元数据、库存和资产符号链接/错属主；忽略请求的 Host/X-Forwarded-Proto，未配置公开 origin 时从实际 socket 派生链接；`--public-base-url` 只允许无凭据、路径、查询和片段的 HTTP(S) origin。
- 所有响应增加 `nosniff`、no-referrer、DENY frame、same-origin resource policy；HTML 额外使用 CSP，限制脚本/媒体/连接为同源并禁止 object/base/form/frame。现有版本化 scene 只使用同源 Three.js、timeline 和视频，生产复测正常。
- 新增伪造 Host、无认证库存、危险 public origin、符号链接数据根、无效 scene/MP4 保留旧 ready 项目、临时文件清理和权限回归；本地/服务器完整测试均为 `262 passed, 7 skipped`。Bandit 仍为 0 high/0 medium，锁定 Node 依赖 `npm audit` 仍为 0。
- 生产源码切换前确认与上一提交逐字节一致，备份为 `/home/ps/osmo-360-visualization/platform_server.mjs.pre-bundle-hardening-20260901-2119.bak`；切换后源码 SHA-256 与 CPU 仓库同为 `3baa7bde...fe2e9`，unit active、0 次重启。
- 生产动态验证：未认证 `/api/devices` 为 401 且带 Bearer challenge，认证读取仍为 200/20 台；现有项目 scene 为 200 且有 CSP，真实 MP4 `bytes=0-11` 为 206、`ftyp` 正常且有 `nosniff`。
- 历史数据修复前 50 个项目目录和 167 个子文件中共有 217 个条目对组/其他用户开放写权限；启动自愈后 50 个目录、168 个文件均无组/其他权限，符号链接、错属主、组/其他可写计数均为 0。
- 仍有 50 个项目的 timeline/视频/scene 对 LAN 公开读取，拆为 SEC-010，等待明确“公开审阅”还是“读认证/签名链接”策略；没有在本轮让已有链接失效。
- 继续审计 HTTP 协议边界发现 Node 默认的 `Expect: 100-continue` 会在处理 request 回调前先发 100；新增本地 `86714b1`、服务器 `554eeae`，写请求在 100 前验证 Bearer，其他 Expect 返回 417，非法 percent-encoding URL 明确返回 400且不记录内部错误栈。
- 生产第二阶段备份为 `/home/ps/osmo-360-visualization/platform_server.mjs.pre-expect-auth-20260901-2123.bak`；当前生产/CPU 仓库源码 SHA-256 同为 `0b0502f5...8c5b`，unit active、0 重启、`MemoryCurrent` 约 19.1 MiB。
- 生产原始 socket 验证：伪造 1,000,000,000 字节 Content-Length 且带 `Expect: 100-continue` 的未认证 POST 直接返回 `HTTP/1.1 401`，响应中没有 `100 Continue`；畸形 URL 为 400，库存 GET 仍为 401。
- `:7865` 容量/列表效率基线：50 项共 4.2 GiB，项目中位约 69.6 MiB、最大约 447.5 MiB；磁盘可用 1.4 TiB。`/api/projects` 50 次本机请求中位 3.55 ms、p95 8.87 ms，CPU 当前无优化必要；无保留策略的长期增长拆为 EFF-006。
- 客户端认证流深审复现 CPython `urllib` 会将 Bearer 随跨域 302 转发。修复提交本地 `5ae4c84`、服务器 `2c7f2d7`：设备同步和 JSON API 禁止重定向；bundle 上传忽略服务端绝对 links，仅接受安全项目 ID 并从已验证 server 构造同源端点。
- 新增两个真实本地 HTTP 服务的回归：修复前第二服务收到 `Authorization`，修复后首个 302 直接作为错误、第二服务请求数为 0；恶意 `https://attacker.invalid/steal-token` links 不会被使用，`../escape` 项目 ID 被拒绝。完整测试本地/服务器均为 `265 passed, 7 skipped`，Bandit 仍为 0 high/0 medium。
- 服务器真实 `umi devices sync` 成功；写前/写后文件 SHA 不同，因此不能宣称字节不变，且没有保留写前语义快照。写后库存为 20 台，与 CPU 仓库 JSON 语义逐项相等、规范化 SHA-256 同为 `5cc0fb49...25d30`，文件 SHA-256 同为 `8f3cf1e6...7b557`，权限 `0600`，服务 active。差异最可能来自此前序列化/字段顺序，但不把该推断当成已证明事实。
- 本轮只改变平台安全/发布边界，不改变四 MP4 检测、位姿、坐标、时间线内容或录制渲染。v5 轨迹 SHA 仍为 `52c3e192...b82ed`，视频 SHA 仍为 `fef54acf...c9a7f7`，因此不重跑数据集、不生成或发送重复视频。

### 2026-09-01 / Cycle 005

- DEP-004 深审确认旧外部运行时是 FFmpeg 4.4.2；但主帧读取由 OpenCV 4.14 内置 FFmpeg 8.1.2（avcodec 62.28.102）完成，因此旧版影响探测、可选音频提取和审阅编码，不是主 Tag 像素解码器。
- 官方当前稳定版 9.0.1 源归档经 PGP 验证；签名指纹 `FCF9 86EA 15E6 E293 A564 4F10 B432 2F04 D676 58D8`，源 SHA-256 `cf38e0e2...7f635`。构建关闭网络/共享库/调试/文档，保留 GPL libx264；离线归档 SHA `6d221609...0487`，ffmpeg/ffprobe SHA 分别为 `91f3138d...1143`、`cc11804f...15c`。
- 新增项目内离线安装器、运行时解析与哈希/版本/权限/属主/符号链接 fail-closed，流水线锁记录完整运行时身份。聚焦测试 24 passed；完整测试本地/服务器均为 `269 passed, 7 skipped`；Bandit 32,643 行仍为 0 high/0 medium。
- 兼容性回归：四个输入各 599 帧，共 2396 帧的 `framemd5` 完全相同；同一 v5 审阅视频转码的旧/新编码结果 299 帧解码像素完全相同。FFprobe 800 次探测旧/新为 23.44/4.34 s；完整视频解码新版快约 5–7%，但 CLI RSS 约 224–230 MiB，高于旧版约 112 MiB，主路径不使用该 CLI 解码。
- v6 提交本地 `0e0a2d4`、服务器等价 `45802c5`。服务器从同一离线归档安装，运行时哈希与本地相同；旧 4.4.2 不再进入四 MP4 流水线。
- 服务器 v6 无缓存处理 12.21 s，平均 CPU 921%、峰值 RSS 310,236 KiB、无 swap。当时系统 load 18.10，另有两条独立 ORB-SLAM 各占约 335% CPU，所以不能将本次墙钟时间与无竞争 6.49 s 基线作为版本回归比较。
- v6 结果仍为 300/300 双侧数值位姿、268 联合可信、266 双侧实测、32 长间隔不可信、`SELF_CALIBRATED_PASS`；`joint_trajectory.csv` SHA 与 v5 同为 `52c3e192...b82ed`。
- 21:47 视频 `processed_joint_trajectory_30hz_front_above_v6.mp4` 为 1920×1080、30 FPS、299 帧、SHA `5fca5373...f79b0e`；audit 明确 `view_preset=flu-front-above`。它只证明与此前同一错误预设的 v5 像素一致，不能证明复现了 19:03 模板；Cycle 006 已将该视频标记为作废并发送更正版。
- 21:47 飞书文字 `om_x100b66593b0430b0df3dababcc942fe`、视频 `om_x100b665938b15ca4c2cc95ee1ae8c26` 发送成功，但均被 Cycle 006 的更正消息取代。

### 2026-09-01 / Cycle 006

- 通过飞书 CLI 读取 19:03 原始消息和媒体：文字 `om_x100b665f4046cca4c3b32975f3ae7a8`，视频 `om_x100b665f40716cacc264ada80e2ff9d`，文件名 `processed_joint_trajectory_30hz_front_above_v1.mp4`。
- 原视频 SHA-256 为 `f816a988...450f83`，封面 SHA-256 为 `afcace27...a8b42`；直接检查封面确认标题为 `TAG MAP + CAM FLU`，视角注记为 `TAG-WALL FRONT + PHYSICAL ABOVE / UP = -Y MAP`。因此权威模板是 `tag-map-front-above`，不是 `flu-front-above`。
- 语义差异：`Tag Map` 必须继续作为世界坐标；`Camera FLU/back=+X` 只定义手部相机子坐标。`flu-front-above` 会把世界视角重表达为 FLU，造成两个 AprilGrid 垂直堆叠，不符合 19:03 构图。
- 在服务器用同一 v6 轨迹重新渲染 `processed_joint_trajectory_30hz_tag_map_front_above_v6.mp4`；audit 为 `view_preset=tag-map-front-above`、`coordinate_frame=TAG MAP`、`camera_frame=CAMERA FLU`，1920×1080、30 FPS、299 帧、SHA-256 `ba80d1d7...127238`。
- 人工检查封面与 7.1 s 帧：两块 AprilGrid 横向朝向观察者；固定机位未跟随/旋转；7.1 s 左侧仍显示真实的 `INTERPOLATED_UNTRUSTED` 数值位姿。只纠正渲染语义，`joint_trajectory.csv` 未修改。
- 飞书更正文字 `om_x100b6659c54deca0c3e103b57ba0ff7`、视频 `om_x100b6659c5740cacc3e63fcd9fc94b4` 发送成功；21:47 的 `flu-front-above` 视频正式作废。
- 整点自动任务已同步改为强制 `tag-map-front-above`，并明确禁止 `flu-front-above`；后续算法变更出片必须匹配 19:03 的世界坐标和构图。
- 渲染器默认预设也由 `legacy-oblique` 改为 `tag-map-front-above`，即使人工漏写参数也不会偏离 19:03 模板；聚焦测试 9 passed、完整测试 270 passed/7 skipped、Bandit 0 high/0 medium。

## 最近一次流水线版本变更验证

- 改动：v6 将外部探测/音频/审阅编码切到哈希锁定的项目 FFmpeg 9.0.1；轨迹算法仍是 v5 的每帧数值位姿 + 独立可信度。
- 提交：本地 `0e0a2d4`，服务器等价 `45802c5`；算法提交仍为本地 `72203d6`、服务器 `b9f7803`。
- 服务器输出：`/home/ps/instaumi-data/instaumi_000001/final/dual-x5-four-mp4-cpu-v6/`。
- 服务器报告：300/300 帧具备数值位姿；联合可信 268（89.33%）；联合实测 266（88.67%）；长间隔不可信 32 帧；全部门通过。
- 服务器运行：竞争负载下 12.21 s；`time -v` 平均 CPU 921%；峰值 RSS 310,236 KiB；无 swap。无竞争基线仍采用 v5 的 6.49 s，待服务器空闲时再做同条件 v6 测量。
- 最终审阅视频：`reviews/processed_joint_trajectory_30hz_tag_map_front_above_v6.mp4`，SHA-256 `ba80d1d7...127238`，固定 `tag-map-front-above`；21:47 的 `flu-front-above` 文件仅保留作拒绝样本。
- 飞书：更正文字和视频均发送成功，消息 ID 见 Cycle 006 日志。
