"""Compatibility wrapper for document loader interfaces."""

from ...loader.base import DocumentLoader, LoaderFactory, UnsupportedDocumentFormat, default_loader_factory

__all__ = ["DocumentLoader", "LoaderFactory", "UnsupportedDocumentFormat", "default_loader_factory"]
