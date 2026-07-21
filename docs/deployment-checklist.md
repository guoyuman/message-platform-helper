# Deployment Checklist

- 配置独立数据目录：`MESSAGE_HELPER_DATA_DIR`。
- 生产启用 Redis：`MESSAGE_HELPER_REDIS_URL`，频控规则依赖原子计数。
- 配置 LLM 网关和模型，要求 JSON 输出稳定。
- 配置 Java 平台 Base URL 和鉴权 Header。
- 配置已有 `message-agent` URL，用于业务消息配置闭环。
- 默认保持 `dryRun=true`，灰度验证 DTO、审计轨迹、权限和租户 Header。
- 对 `/api/chat` 增加网关鉴权、租户隔离、请求体大小限制。
- 审计存储 `decision`、`results[].steps`、平台响应和用户确认记录。
- 对真实邮箱测试增加权限控制，避免被滥用为邮件发送入口。
- 频控策略上线前明确 Redis key 前缀、TTL、跨租户隔离和降级策略。
