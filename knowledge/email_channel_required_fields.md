---
title: 邮件通道必填字段
tags: channel, mail
---

# 邮件通道必填字段

配置邮件通道时，保存前至少要准备 configName、mailHost、mailPort、mailUsername、mailPwd 和 verifyUser。发件账号与展示发件人分离时，还要确认 mailSender 是否符合邮箱服务商要求。

SMTP 相关字段要成组校验：smtpSSL、smtpTLS、mailPort 和服务商文档必须匹配。常见组合是 465 + SSL 或 587 + TLS，但最终以企业邮箱服务商配置为准。使用个人邮箱或企业邮箱时，mailPwd 通常应该填授权码，而不是登录密码。

真实测试发送前先确认 verifyUser 可达、发件账号已开启 SMTP、网络和防火墙允许访问 mailHost:mailPort。测试接口可以使用 `/test/sendTestWithChannelConfig`，它适合在保存通道前验证临时配置。
