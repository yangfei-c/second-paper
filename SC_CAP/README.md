# SC-CAP

SC-CAP 的统一研究工程。算法原型、Web 后端和前端已合并到同一目录，第二篇 Web 调用唯一的 `SCCAPPlanner`，不再维护部署版副本。

## 目录

- `sc_cap/`：canonical SC-CAP 规划器、策略约束、离线评价与冻结 Text-VA 推理。
- `backend/`：FastAPI 接口；保留第一篇单曲推荐链，并将第二篇请求适配到 canonical core。
- `frontend/`：Vue 3 双页面界面。
- `configs/default.yaml`：第二篇 SC-CAP 唯一规划参数配置。
- `tests/`：核心、会话、离线接口和 Web/core 一致性测试。
- `outputs/`：运行时会话与反馈日志，不纳入版本控制。

冻结模型和音乐目录仍从 `D:/AAAAAAAAAA_emotion/Code/MSMMR` 与 `D:/AAAAAAAAAA_emotion/Code/EMMR` 只读加载，不重新训练、不复制、不修改。

## 启动 Web

在本目录打开两个 PowerShell 终端：

```powershell
D:\Anaconda3\envs\music\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8081
```

```powershell
npm --prefix frontend run dev
```

浏览器访问 <http://127.0.0.1:5173>，健康检查为 <http://127.0.0.1:8081/api/health>。

## 验证

```powershell
python -m pytest -q
npm --prefix frontend run build
python scripts/run_demo.py --synthetic --strategy energize
```

第二篇闭环状态固定为 $c_1=q_0$、$c_t=u_{t-1}$（$t\ge2$）。策略约束与累计前缀约束均为硬约束；交集为空时返回并记录 `infeasible`，不会静默放宽。`open_loop_sc_cap` 仅是显式实验基线，使用 $c_{t+1}=p_t$，不属于部署闭环。
