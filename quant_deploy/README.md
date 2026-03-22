# 量化Alpha · A股策略系统

真实A股行情数据 + 技术指标分析 + 策略回测

## 快速启动

### Windows（推荐）
双击运行 `启动.bat`，自动安装依赖并打开浏览器。

### 手动启动
```bash
# 1. 安装依赖（只需一次）
pip install flask flask-cors akshare pandas numpy

# 2. 启动后端
python server.py

# 3. 浏览器访问
http://127.0.0.1:5000
```

## 功能说明

| 功能 | 说明 |
|------|------|
| K线图 | 真实历史行情，支持日K/周K |
| 均线 | MA5 / MA20 / MA60，自动判断金叉死叉 |
| RSI | 14日RSI，超买超卖信号 |
| MACD | DIF/DEA/柱状图 |
| 布林带 | Bollinger Band 通道 |
| 回测 | 双均线策略回测，计算胜率/夏普/最大回撤 |
| 信号扫描 | 一键扫描全部股票池信号 |

## 股票池

| 代码 | 名称 |
|------|------|
| 300418 | 昆仑万维 ★ |
| 600519 | 贵州茅台 |
| 000858 | 五粮液 |
| 300750 | 宁德时代 |
| 601318 | 中国平安 |
| 000333 | 美的集团 |

## 新增股票

在 `server.py` 的 `STOCK_POOL` 字典中加入股票代码和名称即可：
```python
STOCK_POOL = {
    "300418": "昆仑万维",
    "600519": "贵州茅台",
    "你的代码": "股票名称",   # ← 在这里添加
}
```

## 数据来源

- 历史K线：东方财富（通过 AKShare 接口）
- 完全免费，无需注册，无需 API Key

## 目录结构

```
quant_system/
├── server.py          # Flask 后端
├── requirements.txt   # 依赖列表
├── 启动.bat           # Windows 一键启动
├── README.md
└── static/
    └── index.html     # 前端页面
```
