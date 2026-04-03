# 敏捷洞察1.0

一个面向培训资料与行业白皮书的结构化总结助手。

## 当前版本能力

- 支持 PDF 上传
- 自动保存原文件
- 自动解析 PDF 文本
- 自动生成一页摘要
- 自动生成三级逻辑大纲
- 支持基于原文的问答
- 返回引用片段，提升回答可追溯性

## 当前项目形态

这是一个本地 MVP 版本，主要用于验证完整产品闭环。

当前技术实现：
- 后端：FastAPI
- PDF 解析：PyMuPDF
- 模型调用：OpenAI 兼容接口
- 数据存储：本地文件系统

## 项目目录结构

Agile-Insights-v1.0/
├─ app/
│  ├─ main.py
│  ├─ routes/
│  ├─ services/
│  ├─ static/
│  ├─ templates/
│  └─ __init__.py
├─ data/
│  ├─ uploads/
│  ├─ parsed/
│  └─ outputs/
├─ docs/
├─ tests/
├─ .env
├─ .env.example
├─ .gitignore
├─ README.md
└─ requirements.txt

## 本地运行方式

### 1. 创建虚拟环境
python -m venv .venv

### 2.安装依赖
.venv\Scripts\python.exe -m pip install -r requirements.txt

### 3.配置环境变量
API_KEY=真实key
API_BASE=模型服务地址
MODEL_NAME=模型名

### 4.启动服务
.venv\Scripts\python.exe -m uvicorn app.main:app --reload

### 5.打开本地页面
访问：http://127.0.0.1:8000

## 当前限制

- 当前仅支持 PDF
- 当前只支持单文件分析
- 当前数据保存于本地 `data/` 目录
- 当前引用片段使用的是最小检索逻辑，不是完整向量检索方案
- 当前版本更适合 demo 与功能验证，不是正式生产版本

## 下一步计划

- 优化问答引用质量
- 优化页面展示体验
- 部署到公网
- 后续考虑接入对象存储与更稳的检索方案

## 部署说明

### 当前项目部署启动命令为：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
### 当前依赖的环境变量为：
API_KEY=
API_BASE=
MODEL_NAME=

## 在线体验

Demo 地址：
https://agile-insights-v1-0.onrender.com/

当前已验证能力：
- 首页访问
- /health 健康检查
- PDF 上传
- 一页摘要
- 三级逻辑大纲
- 基于原文问答
- 引用片段返回