# API Contract

## POST /api/chat

Request:

```json
{
  "sessionId": "demo",
  "userId": "u001",
  "tenantId": "tenant-a",
  "locale": "zh_CN",
  "text": "同步采购订单下所有模板到阿拉伯语",
  "payload": {
    "template": {
      "sourceLanguage": "en_US",
      "targetLanguage": "ar_SA",
      "documentId": "purchase_order",
      "groupCode": "businessprocess",
      "templates": [
        {
          "messageTempVO": {
            "id": "tpl-001",
            "templateName": "Purchase order notice",
            "templateCode": "purchase_order_notice",
            "languageDict": {
              "templateName": {
                "en_US": {"id": "", "value": "Purchase order notice"}
              }
            }
          },
          "messageTemplateContentVOList": [
            {
              "id": "content-001",
              "channelType": "mail",
              "language": "en_US",
              "title": "Business approval notification",
              "content": "Hello, you have a business message to process. {{orderNo}}"
            }
          ]
        }
      ]
    }
  },
  "dryRun": true
}
```

Response:

```json
{
  "ok": true,
  "decision": {
    "need_retrieval": false,
    "request_type": "action",
    "domain": "template",
    "operation": "execute",
    "selected_agents": ["template"],
    "confidence": 0.78
  },
  "results": [
    {
      "agent": "template",
      "ok": true,
      "output": {
        "summary": "template language sync translated 1 template(s), skipped 0 existing template(s)",
        "sourceLanguage": "en_US",
        "targetLanguage": "ar_SA",
        "translatedTemplates": [
          {
            "templateId": "tpl-001",
            "templateCode": "purchase_order_notice",
            "targetLanguage": "ar_SA",
            "translatedContentCount": 1,
            "saveResponse": {
              "ok": true,
              "status": "dry-run"
            }
          }
        ],
        "skippedTemplates": [],
        "failedTemplates": []
      },
      "steps": []
    }
  ]
}
```

## POST /api/knowledge/ingest

```json
{
  "title": "邮件通道配置规范",
  "content": "邮箱通道需要 mailHost、mailPort、mailUsername、mailPwd、verifyUser。",
  "tags": ["channel", "mail"],
  "source": "ops-doc"
}
```

## POST /api/strategy/evaluate

```json
{
  "strategy": {
    "name": "工作时间发送",
    "enabled": true,
    "rules": [
      {"kind": "channel_limit", "config": {"allowedChannels": ["mail"]}},
      {"kind": "frequency", "config": {"scope": "recipient", "limit": 2, "windowSeconds": 3600}}
    ]
  },
  "message": {
    "channels": ["mail"],
    "receivers": ["u001"],
    "templateCode": "approval"
  }
}
```
