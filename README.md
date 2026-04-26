# 职业共生工坊

这是一个围绕“职业知识库数据包 + AI 生成体验内容”的文字游戏 Demo 项目。

## 项目文件

- demo.html：前端文字游戏界面
- backend_server.py：本地后端与 DeepSeek 代理
- packs/：官方职业包示例
- AI时代下的「职业共生工坊」游戏构思方案

## 本地运行

1. 启动后端：

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python313/python.exe backend_server.py
```

2. 打开页面：

- http://127.0.0.1:8787/demo.html

## Demo 使用方式

1. 点击首页“开始游戏”。
2. 在开局配置页选择职业包来源：

- 选择官方职业包
- 上传自定义 JSON
- 输入自然语言后由 AI 生成职业包 JSON

3. 配置模型参数并填写 API Key（如需真实生成）。
4. 点击“生成并进入游戏”。
5. 在每个阶段做出选择，点击“进入下一阶段”推进剧情。
6. 最终阶段后生成 AI 结局报告，查看路线总结、评级、标签和建议。
7. 可点击“重新开局”快速体验其他职业包。

## 自定义职业包格式

职业包支持三种优先级输入方式：

1. 直接粘贴 JSON
2. 上传 JSON 文件
3. 官方职业包

JSON 需包含以下字段：

- profession
- type
- audience
- coreSkills
- workflow
- scenarios
- constraints
- cases
- style

## 常见问题

1. 第二阶段提示 404

请确认后端已重启到最新版本，续阶段接口为 /api/continue。

2. 生成失败后进入模拟模式

说明后端请求失败，Demo 仍可继续演示流程。

3. AI 生成职业包无响应

请检查 API Key 是否填写，以及后端终端日志是否有报错。

4. 没有出现结局报告

请确认已到最后阶段并完成当前阶段选择后再推进。
