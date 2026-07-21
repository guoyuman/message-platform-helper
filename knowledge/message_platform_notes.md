---
title: Message Platform Notes
tags: platform, endpoint, strategy
---

# Message Platform Notes

业务消息配置应复用已有 message-agent，保存前生成 `businessMessageConfigVO` 和 `businessMessageConditionVOList`。

邮箱通道保存接口是 `/msgTemplateChannel/saveChannelConfig`。邮件测试接口是 `/test/sendTestWithChannelConfig`。真实发送测试需要 SMTP 地址、端口、用户名、密码和验证邮箱。

发送策略建议按通道限制、定时发送、频次控制的顺序执行。频次控制应优先使用 Redis 计数，并按租户、通道、接收人或模板维度隔离 key。
