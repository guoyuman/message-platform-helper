---
title: 邮件发送失败排查
tags: troubleshooting, mail, channel
---

# 邮件发送失败排查

邮件发送失败优先检查通道配置字段：mailHost、mailPort、mailUsername、mailPwd、verifyUser、smtpSSL、smtpTLS、mailSender。先判断失败发生在连接、认证、收件人、内容渲染还是第三方限流。

常见原因包括 SMTP 地址或端口错误、SSL/TLS 组合不匹配、使用登录密码而不是授权码、发件账号未开启 SMTP、verifyUser 不可达、网络或防火墙阻断、第三方邮箱服务限流。企业邮箱还可能要求发件账号、域名和 sender 显示名满足安全策略。

排查顺序建议：

1. 调用 `/test/sendTestWithChannelConfig` 做测试发送。
2. 根据返回错误定位认证、连接、收件人或内容问题。
3. 查看 Java 平台日志里的 channelType、templateCode、receiver 和第三方响应。
4. 如果测试发送成功但业务发送失败，再回到业务配置、模板变量和接收人解析链路排查。
