# -*- coding: utf-8 -*-
"""按题型构造对话消息（提示词模板可在配置中覆盖）。"""

CHOICE_CATEGORIES = ("chinese_knowledge", "english_knowledge", "logic_reasoning")


def format_options(options):
    return "".join("%s. %s\n" % (k, v) for k, v in (options or {}).items())


def build_messages(item, cfg):
    """根据题目与配置构造 messages 列表。"""
    prompts = cfg.get("prompts") or {}
    system = (prompts.get("system") or "").strip()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    cat = item.get("category")
    if cat in CHOICE_CATEGORIES:
        template = prompts.get("choice") or ""
        user = template.format(question=item["question"],
                               options=format_options(item.get("options")).strip())
    elif cat == "math_reasoning":
        template = prompts.get("math") or ""
        user = template.format(question=item["question"])
    elif cat == "code_generation":
        template = prompts.get("code") or ""
        user = template.format(prompt=item["prompt"])
    else:
        # instruction_following / safety_alignment / creative_writing：原样发送
        user = item.get("instruction") or item.get("prompt") or ""
    messages.append({"role": "user", "content": user})
    return messages
