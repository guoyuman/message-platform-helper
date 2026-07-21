---
title: 消息平台接口映射
tags: endpoint, platform
---

# 消息平台接口映射

业务消息配置可以复用 message-agent，也可以调用 Java 平台接口 `/bizMessageConfig/save` 保存。保存前要生成 businessMessageConfigVO 和 businessMessageConditionVOList，确保业务单据、动作、租户、应用编码、启用状态和接收人规则齐全。

邮件通道配置保存接口是 `/msgTemplateChannel/saveChannelConfig`。保存前可以用 `/test/sendTestWithChannelConfig` 带临时通道配置做真实测试，也可以用 `/test/sendTest` 对已保存通道做测试。

模板保存接口是 `/msgTemplate/saveTempInfo`。保存模板时重点校验 messageTempVO、messageTemplateContentVOList、channelType、language、title、content、raw 和 contentType。
