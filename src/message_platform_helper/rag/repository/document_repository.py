"""Document repository."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..db import DocumentRecord
from ..models import Document
from .utils import stable_uuid


@dataclass
class DocumentRepository:
    session: Session

    def save(self, document: Document) -> DocumentRecord:
        record = self.session.get(DocumentRecord, stable_uuid(document.id))
        if record is None:
            record = DocumentRecord(id=stable_uuid(document.id))
            self.session.add(record)
        record.title = document.title
        record.content = document.content
        record.source = document.metadata.source
        record.metadata_ = document.metadata.to_dict()
        return record
