---
title: 模板渲染失败排查
tags: troubleshooting, template, send
---

# 模板渲染失败排查

模板渲染失败通常与变量、语种或内容格式有关。先检查 messageTemplateContentVOList 中目标通道和语种是否存在，再检查模板变量是否能从业务数据取到值。

常见问题包括变量名大小写不一致、`{{orderNo}}`、`{amount}` 或 `[md#field]` 这类变量没有对应业务字段、HTML 标签未闭合、JSON 内容被错误转义、富文本内容被二次编码。多语种模板要确认 sourceLanguage 和 targetLanguage，目标语种已存在时应跳过或更新，不要重复新增。

如果模板预览正常但发送失败，继续检查发送时的 payload 是否和预览一致，尤其是业务系统运行时是否缺少字段、字段为空字符串，或字段在不同租户下使用了不同编码。
