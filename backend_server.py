import argparse
from datetime import datetime
import json
import os
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


def log_debug(trace_id, stage, message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{trace_id}] [{stage}] {message}")


def mask_secret(value):
    text = str(value or "").strip()
    if not text:
        return "<empty>"
    if len(text) <= 8:
        return text[:2] + "***" + text[-2:]
    return text[:4] + "***" + text[-4:]


def normalize_api_key(raw_key):
    key = str(raw_key or "").strip().strip('"').strip("'")
    if not key:
        raise ValueError("DeepSeek API Key 为空")

    try:
        key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "DeepSeek API Key 包含无法写入 HTTP 头的非拉丁字符，请重新粘贴为纯英文/数字形式"
        ) from exc

    return key


def build_prompt(job_data):
    return "\n".join(
        [
            "你是一个游戏Demo生成器，需要把职业知识库数据包转成一段可玩的文字模拟体验。",
            "请只输出严格 JSON，不要输出额外解释。",
            "输出字段必须包含：title, hook, mission, choices, feedbackByChoice, knowledgeTips, nextScene。",
            "choices 必须是 3 个选项，每个选项包含 label 和 detail。",
            "feedbackByChoice 必须是 3 条反馈，对应三个选项。",
            "knowledgeTips 必须是 3 到 5 条简短知识点。",
            "nextScene 用一句话描述下一步体验。",
            "职业数据包如下：",
            json.dumps(job_data, ensure_ascii=False, indent=2),
        ]
    )


def build_continue_prompt(job_data, previous_result, choice_index, stage):
    return "\n".join(
        [
            "你是一个游戏Demo续生成器，需要在上一阶段结果基础上，生成下一阶段职业体验。",
            "请只输出严格 JSON，不要输出额外解释。",
            "输出字段必须包含：title, hook, mission, choices, feedbackByChoice, knowledgeTips, nextScene。",
            "choices 必须是 3 个选项，每个选项包含 label 和 detail。",
            "feedbackByChoice 必须是 3 条反馈，对应三个选项。",
            "knowledgeTips 必须是 3 到 5 条简短知识点。",
            f"当前要生成第 {stage} 阶段内容，必须体现玩家上一阶段选择的影响。",
            "职业数据包如下：",
            json.dumps(job_data, ensure_ascii=False, indent=2),
            "上一阶段结果如下：",
            json.dumps(previous_result, ensure_ascii=False, indent=2),
            f"玩家上一阶段选择索引：{choice_index}",
        ]
    )


def build_packify_prompt(user_text):
    return "\n".join(
        [
            "你是一个职业知识库整理器。",
            "请把用户输入的自然语言职业设定，整理成严格 JSON。",
            "只输出 JSON，不要输出解释。",
            "JSON 字段必须包含：profession, type, audience, coreSkills, workflow, scenarios, constraints, cases, style。",
            "其中 coreSkills, workflow, scenarios, constraints, cases 都必须是数组，且每项为简短中文句子或词组。",
            "用户输入如下：",
            user_text,
        ]
    )


def build_finish_prompt(job_data, history, current_result, stage, max_stage):
    return "\n".join(
        [
            "你是一个游戏结局报告生成器。",
            "请根据职业包、每一阶段的玩家选择、以及当前阶段结果，生成一份结局报告。",
            "只输出严格 JSON，不要输出额外解释。",
            "JSON 字段必须包含：title, summary, route, evaluation, nextStep, tags, rating。",
            "title 是结局标题。",
            "summary 是 2-4 段中文总结，概括玩家在整个职业体验中的走向。",
            "route 是数组，每一项描述一个阶段的选择与影响。",
            "evaluation 是对玩家整体倾向的简短评价。",
            "nextStep 是下一轮建议或继续体验的方向。",
            "tags 是数组，返回 3-5 个标签。",
            "rating 是结局评级，例如 S、A、B+、B、C。",
            f"当前总阶段数：{max_stage}，当前阶段：{stage}。",
            "职业数据包如下：",
            json.dumps(job_data, ensure_ascii=False, indent=2),
            "阶段历史如下：",
            json.dumps(history, ensure_ascii=False, indent=2),
            "当前阶段结果如下：",
            json.dumps(current_result, ensure_ascii=False, indent=2),
        ]
    )


def parse_first_json_object(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("模型返回内容为空")

    # 去掉 markdown 代码块包装，提升容错率。
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError("无法从模型输出中提取 JSON 起始位置")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end < 0:
        raise ValueError("模型输出 JSON 结构不完整")

    return json.loads(text[start : end + 1])


def _pick(result, keys, default=None):
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    return default


def normalize_result(result):
    if not isinstance(result, dict):
        raise ValueError("模型输出不是对象结构")

    normalized = {
        "title": str(_pick(result, ["title", "sceneTitle", "阶段标题"], "未命名阶段")),
        "hook": str(_pick(result, ["hook", "opening", "开场"], "")),
        "mission": str(_pick(result, ["mission", "task", "goal", "任务"], "")),
        "choices": _pick(result, ["choices", "options", "选项"], []),
        "feedbackByChoice": _pick(
            result,
            ["feedbackByChoice", "feedback", "optionFeedback", "选项反馈"],
            [],
        ),
        "knowledgeTips": _pick(result, ["knowledgeTips", "tips", "知识点"], []),
        "nextScene": str(_pick(result, ["nextScene", "next", "下一步"], "")),
    }

    # choices 支持字符串数组或对象数组。
    fixed_choices = []
    for item in normalized["choices"] if isinstance(normalized["choices"], list) else []:
        if isinstance(item, dict):
            label = str(_pick(item, ["label", "title", "name"], "未命名选择"))
            detail = str(_pick(item, ["detail", "desc", "description"], ""))
            fixed_choices.append({"label": label, "detail": detail})
        else:
            fixed_choices.append({"label": str(item), "detail": ""})

    if not fixed_choices:
        raise ValueError("模型输出缺少可用 choices")
    normalized["choices"] = fixed_choices

    feedback = normalized["feedbackByChoice"]
    if not isinstance(feedback, list):
        feedback = [str(feedback)] if feedback else []
    feedback = [str(x) for x in feedback]
    if not feedback:
        feedback = ["系统已记录你的选择。"] * len(fixed_choices)
    if len(feedback) < len(fixed_choices):
        feedback.extend([feedback[-1]] * (len(fixed_choices) - len(feedback)))
    normalized["feedbackByChoice"] = feedback[: len(fixed_choices)]

    tips = normalized["knowledgeTips"]
    if not isinstance(tips, list):
        tips = [str(tips)] if tips else []
    tips = [str(x) for x in tips if str(x).strip()]
    if not tips:
        tips = ["职业体验已完成一轮有效决策。", "你正在建立可迁移的职业判断框架。"]
    normalized["knowledgeTips"] = tips

    if not normalized["hook"].strip():
        normalized["hook"] = "你进入了新的职业阶段。"
    if not normalized["mission"].strip():
        normalized["mission"] = "请根据当前信息做出下一步职业决策。"
    if not normalized["nextScene"].strip():
        normalized["nextScene"] = "完成当前决策后，将进入下一阶段。"

    return normalized


def normalize_pack(result):
    if not isinstance(result, dict):
        raise ValueError("模型输出不是对象结构")

    normalized = {
        "profession": str(_pick(result, ["profession", "job", "职业"], "未命名职业")),
        "type": str(_pick(result, ["type", "category", "职业类型"], "未知类型")),
        "audience": str(_pick(result, ["audience", "target", "受众"], "职业探索者")),
        "coreSkills": _pick(result, ["coreSkills", "skills", "核心技能"], []),
        "workflow": _pick(result, ["workflow", "steps", "流程"], []),
        "scenarios": _pick(result, ["scenarios", "cases", "场景"], []),
        "constraints": _pick(result, ["constraints", "limits", "限制"], []),
        "cases": _pick(result, ["cases", "examples", "示例"], []),
        "style": str(_pick(result, ["style", "tone", "风格"], "偏纪实、偏入门教学")),
    }

    for key in ["coreSkills", "workflow", "scenarios", "constraints", "cases"]:
        value = normalized[key]
        if not isinstance(value, list):
            value = [str(value)] if value else []
        value = [str(x).strip() for x in value if str(x).strip()]
        normalized[key] = value

    if not normalized["coreSkills"]:
        normalized["coreSkills"] = ["观察", "判断", "沟通"]
    if not normalized["workflow"]:
        normalized["workflow"] = ["接收任务", "分析现场", "给出建议"]
    if not normalized["scenarios"]:
        normalized["scenarios"] = ["初始场景"]
    if not normalized["constraints"]:
        normalized["constraints"] = ["信息不完整", "需要做出初步判断"]
    if not normalized["cases"]:
        normalized["cases"] = ["一件需要现场判断的真实问题"]

    return normalized


def _extract_text_from_model_response(data):
    text = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message") or {}
        text = message.get("content") or first.get("text") or ""
    return text


def call_deepseek(job_data, api_key, model, temperature, endpoint, trace_id="-"):
    prompt = build_prompt(job_data)
    api_key = normalize_api_key(api_key)
    body = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是一个游戏Demo生成器，请严格输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    log_debug(
        trace_id,
        "generate",
        f"calling deepseek endpoint={endpoint} model={body['model']} temp={temperature} prompt_chars={len(prompt)} api_key={mask_secret(api_key)}",
    )

    req = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            log_debug(trace_id, "generate", f"deepseek http_status={resp.status} response_chars={len(raw)}")
    except HTTPError as exc:
        error_detail = exc.read().decode("utf-8", errors="replace")
        log_debug(trace_id, "generate", f"deepseek http_error={exc.code} detail={error_detail[:320]}")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {error_detail}") from exc
    except URLError as exc:
        log_debug(trace_id, "generate", f"deepseek url_error={exc}")
        raise RuntimeError(f"DeepSeek 连接失败: {exc}") from exc

    data = json.loads(raw)
    text = _extract_text_from_model_response(data)
    log_debug(trace_id, "generate", f"model_text_chars={len(text)} text_preview={text[:180].replace(chr(10), ' ')}")

    try:
        parsed = parse_first_json_object(text)
    except Exception as exc:
        snippet = (text or "")[:280].replace("\n", " ")
        log_debug(trace_id, "generate", f"parse_json_failed snippet={snippet}")
        raise RuntimeError(f"模型返回无法解析为 JSON，片段: {snippet}") from exc

    normalized = normalize_result(parsed)
    log_debug(trace_id, "generate", f"normalize_ok title={normalized.get('title','')} choices={len(normalized.get('choices', []))}")
    return normalized


def call_deepseek_continue(job_data, previous_result, choice_index, stage, api_key, model, temperature, endpoint, trace_id="-"):
    prompt = build_continue_prompt(job_data, previous_result, choice_index, stage)
    api_key = normalize_api_key(api_key)
    body = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是一个游戏Demo续生成器，请严格输出 JSON。"},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    log_debug(
        trace_id,
        "continue",
        f"calling deepseek endpoint={endpoint} model={body['model']} temp={temperature} stage={stage} choice_index={choice_index} prompt_chars={len(prompt)} api_key={mask_secret(api_key)}",
    )

    req = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            log_debug(trace_id, "continue", f"deepseek http_status={resp.status} response_chars={len(raw)}")
    except HTTPError as exc:
        error_detail = exc.read().decode("utf-8", errors="replace")
        log_debug(trace_id, "continue", f"deepseek http_error={exc.code} detail={error_detail[:320]}")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {error_detail}") from exc
    except URLError as exc:
        log_debug(trace_id, "continue", f"deepseek url_error={exc}")
        raise RuntimeError(f"DeepSeek 连接失败: {exc}") from exc

    data = json.loads(raw)
    text = _extract_text_from_model_response(data)
    log_debug(trace_id, "continue", f"model_text_chars={len(text)} text_preview={text[:180].replace(chr(10), ' ')}")

    try:
        parsed = parse_first_json_object(text)
    except Exception as exc:
        snippet = (text or "")[:280].replace("\n", " ")
        log_debug(trace_id, "continue", f"parse_json_failed snippet={snippet}")
        raise RuntimeError(f"续阶段模型返回无法解析为 JSON，片段: {snippet}") from exc

    normalized = normalize_result(parsed)
    log_debug(trace_id, "continue", f"normalize_ok title={normalized.get('title','')} choices={len(normalized.get('choices', []))}")
    return normalized


def call_deepseek_packify(user_text, api_key, model, temperature, endpoint, trace_id="-"):
    prompt = build_packify_prompt(user_text)
    api_key = normalize_api_key(api_key)
    body = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是一个职业知识库整理器，请严格输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    log_debug(
        trace_id,
        "packify",
        f"calling deepseek endpoint={endpoint} model={body['model']} temp={temperature} text_chars={len(user_text)} api_key={mask_secret(api_key)}",
    )

    req = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            log_debug(trace_id, "packify", f"deepseek http_status={resp.status} response_chars={len(raw)}")
    except HTTPError as exc:
        error_detail = exc.read().decode("utf-8", errors="replace")
        log_debug(trace_id, "packify", f"deepseek http_error={exc.code} detail={error_detail[:320]}")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {error_detail}") from exc
    except URLError as exc:
        log_debug(trace_id, "packify", f"deepseek url_error={exc}")
        raise RuntimeError(f"DeepSeek 连接失败: {exc}") from exc

    data = json.loads(raw)
    text = _extract_text_from_model_response(data)
    log_debug(trace_id, "packify", f"model_text_chars={len(text)} text_preview={text[:180].replace(chr(10), ' ')}")

    try:
        parsed = parse_first_json_object(text)
    except Exception as exc:
        snippet = (text or "")[:280].replace("\n", " ")
        log_debug(trace_id, "packify", f"parse_json_failed snippet={snippet}")
        raise RuntimeError(f"模型返回无法解析为 JSON，片段: {snippet}") from exc

    normalized = normalize_pack(parsed)
    log_debug(trace_id, "packify", f"normalize_ok profession={normalized.get('profession','')} coreSkills={len(normalized.get('coreSkills', []))}")
    return normalized


def call_deepseek_finish(job_data, history, current_result, stage, max_stage, api_key, model, temperature, endpoint, trace_id="-"):
    prompt = build_finish_prompt(job_data, history, current_result, stage, max_stage)
    api_key = normalize_api_key(api_key)
    body = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是一个游戏结局报告生成器，请严格输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    log_debug(
        trace_id,
        "finish",
        f"calling deepseek endpoint={endpoint} model={body['model']} temp={temperature} stage={stage} max_stage={max_stage} history_count={len(history)} api_key={mask_secret(api_key)}",
    )

    req = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            log_debug(trace_id, "finish", f"deepseek http_status={resp.status} response_chars={len(raw)}")
    except HTTPError as exc:
        error_detail = exc.read().decode("utf-8", errors="replace")
        log_debug(trace_id, "finish", f"deepseek http_error={exc.code} detail={error_detail[:320]}")
        raise RuntimeError(f"DeepSeek HTTPError {exc.code}: {error_detail}") from exc
    except URLError as exc:
        log_debug(trace_id, "finish", f"deepseek url_error={exc}")
        raise RuntimeError(f"DeepSeek 连接失败: {exc}") from exc

    data = json.loads(raw)
    text = _extract_text_from_model_response(data)
    log_debug(trace_id, "finish", f"model_text_chars={len(text)} text_preview={text[:180].replace(chr(10), ' ')}")

    try:
        parsed = parse_first_json_object(text)
    except Exception as exc:
        snippet = (text or "")[:280].replace("\n", " ")
        log_debug(trace_id, "finish", f"parse_json_failed snippet={snippet}")
        raise RuntimeError(f"结局报告无法解析为 JSON，片段: {snippet}") from exc

    normalized = {
        "title": str(_pick(parsed, ["title", "reportTitle", "结局标题"], "结局报告")),
        "summary": str(_pick(parsed, ["summary", "report", "总结"], "")),
        "route": _pick(parsed, ["route", "path", "路线"], []),
        "evaluation": str(_pick(parsed, ["evaluation", "judge", "评价"], "")),
        "nextStep": str(_pick(parsed, ["nextStep", "next", "建议"], "")),
        "tags": _pick(parsed, ["tags", "labels", "标签"], []),
        "rating": str(_pick(parsed, ["rating", "score", "结局评级"], "B")),
    }

    if not isinstance(normalized["route"], list):
        normalized["route"] = [str(normalized["route"])] if normalized["route"] else []
    if not isinstance(normalized["tags"], list):
        normalized["tags"] = [str(normalized["tags"])] if normalized["tags"] else []
    normalized["route"] = [str(x) for x in normalized["route"] if str(x).strip()]
    normalized["tags"] = [str(x) for x in normalized["tags"] if str(x).strip()]

    if not normalized["summary"].strip():
        normalized["summary"] = "你已经完成了一轮职业体验，系统正在整理你的决策路径。"
    if not normalized["evaluation"].strip():
        normalized["evaluation"] = "你在本轮体验中展现了稳定的职业判断。"
    if not normalized["nextStep"].strip():
        normalized["nextStep"] = "你可以重新开始，体验另一条职业路线。"
    if not normalized["tags"]:
        normalized["tags"] = ["结局", "职业判断", "路线总结"]
    if not normalized["rating"].strip():
        normalized["rating"] = "B"

    log_debug(trace_id, "finish", f"normalize_ok title={normalized.get('title','')} route={len(normalized.get('route', []))} tags={len(normalized.get('tags', []))} rating={normalized.get('rating','')}")
    return normalized


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def _write_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"ok": True, "service": "职业共生工坊后端"})
            return
        return super().do_GET()

    def do_POST(self):
        trace_id = uuid.uuid4().hex[:8]
        path_only = self.path.split("?", 1)[0]
        normalized_path = path_only.rstrip("/") or "/"
        log_debug(trace_id, "request", f"method=POST path={self.path} normalized={normalized_path}")

        if normalized_path not in ("/api/generate", "/api/continue", "/api/packify", "/api/finish"):
            log_debug(trace_id, "request", "rejected unknown endpoint")
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "接口不存在"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            log_debug(trace_id, "request", "invalid json body")
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "请求体不是有效 JSON"})
            return

        api_key = (payload.get("apiKey") or "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
        if normalized_path in ("/api/generate", "/api/continue", "/api/finish", "/api/packify"):
            if not api_key:
                log_debug(trace_id, "request", "api key missing")
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "缺少 DeepSeek API Key，请在前端填写或设置环境变量 DEEPSEEK_API_KEY",
                    },
                )
                return

            try:
                api_key_preview = mask_secret(normalize_api_key(api_key))
            except ValueError as exc:
                log_debug(trace_id, "request", f"api key invalid reason={exc}")
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

        model = (payload.get("model") or DEFAULT_MODEL).strip()
        try:
            temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
        except (TypeError, ValueError):
            temperature = DEFAULT_TEMPERATURE
        endpoint = (payload.get("deepseekEndpoint") or DEFAULT_DEEPSEEK_ENDPOINT).strip()

        try:
            if normalized_path == "/api/packify":
                user_text = str(payload.get("text") or payload.get("inputText") or payload.get("content") or "").strip()
                if not user_text:
                    log_debug(trace_id, "packify", "missing input text")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "text 不能为空"})
                    return
                log_debug(trace_id, "request", f"mode=packify model={model} temp={temperature} endpoint={endpoint} api_key={api_key_preview}")
                result = call_deepseek_packify(user_text, api_key, model, temperature, endpoint, trace_id=trace_id)
                self._write_json(HTTPStatus.OK, {"ok": True, "source": "后端 DeepSeek", "result": result})
                return

            job_data = payload.get("jobData")
            if not isinstance(job_data, dict):
                log_debug(trace_id, "request", "missing jobData object")
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "jobData 必须是对象"})
                return

            profession = str(job_data.get("profession", "未知职业"))
            log_debug(
                trace_id,
                "request",
                f"mode={'generate' if normalized_path == '/api/generate' else 'continue' if normalized_path == '/api/continue' else 'finish'} profession={profession} model={model} temp={temperature} endpoint={endpoint} api_key={api_key_preview}",
            )

            if normalized_path == "/api/generate":
                result = call_deepseek(job_data, api_key, model, temperature, endpoint, trace_id=trace_id)
            elif normalized_path == "/api/continue":
                previous_result = payload.get("previousResult")
                choice_index = payload.get("choiceIndex")
                stage = payload.get("stage")
                if isinstance(choice_index, str) and choice_index.isdigit():
                    choice_index = int(choice_index)
                if isinstance(stage, str) and stage.isdigit():
                    stage = int(stage)
                if not isinstance(previous_result, dict):
                    log_debug(trace_id, "continue", "invalid previousResult")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "previousResult 必须是对象"})
                    return
                if not isinstance(choice_index, int):
                    log_debug(trace_id, "continue", f"invalid choiceIndex={choice_index}")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "choiceIndex 必须是整数"})
                    return
                if not isinstance(stage, int) or stage < 1:
                    log_debug(trace_id, "continue", f"invalid stage={stage}")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "stage 必须是大于等于 1 的整数"})
                    return

                result = call_deepseek_continue(
                    job_data,
                    previous_result,
                    choice_index,
                    stage,
                    api_key,
                    model,
                    temperature,
                    endpoint,
                    trace_id=trace_id,
                )
            else:
                previous_result = payload.get("previousResult")
                history = payload.get("history") or []
                stage = payload.get("stage")
                max_stage = payload.get("maxStage")
                if not isinstance(previous_result, dict):
                    log_debug(trace_id, "finish", "invalid currentResult")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "currentResult 必须是对象"})
                    return
                if not isinstance(history, list):
                    log_debug(trace_id, "finish", "invalid history")
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history 必须是数组"})
                    return
                if isinstance(stage, str) and stage.isdigit():
                    stage = int(stage)
                if isinstance(max_stage, str) and max_stage.isdigit():
                    max_stage = int(max_stage)
                if not isinstance(stage, int) or stage < 1:
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "stage 必须是大于等于 1 的整数"})
                    return
                if not isinstance(max_stage, int) or max_stage < 1:
                    self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "maxStage 必须是大于等于 1 的整数"})
                    return

                log_debug(
                    trace_id,
                    "request",
                    f"mode=finish profession={profession} model={model} temp={temperature} endpoint={endpoint} history_count={len(history)} api_key={api_key_preview}",
                )
                result = call_deepseek_finish(
                    job_data,
                    history,
                    previous_result,
                    stage,
                    max_stage,
                    api_key,
                    model,
                    temperature,
                    endpoint,
                    trace_id=trace_id,
                )
        except Exception as exc:
            log_debug(trace_id, "response", f"failed error={exc}")
            self._write_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})
            return

        log_debug(trace_id, "response", f"ok title={result.get('title','')} choices={len(result.get('choices', []))}")

        self._write_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "source": "后端 DeepSeek",
                "result": result,
            },
        )


def main():
    parser = argparse.ArgumentParser(description="职业共生工坊 Demo 后端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    static_dir = str(Path(args.dir).resolve())
    server = ThreadingHTTPServer(
        (args.host, args.port),
        lambda *handler_args, **handler_kwargs: DemoHandler(
            *handler_args, directory=static_dir, **handler_kwargs
        ),
    )
    print(f"[demo-backend] serving {static_dir}")
    print(f"[demo-backend] open http://{args.host}:{args.port}/demo.html")
    print("[demo-backend] health check: /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
