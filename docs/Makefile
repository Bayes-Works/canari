# Minimal Makefile for Sphinx

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: html clean help

help:
	@echo "Usage:"
	@echo "  make html  — build HTML docs"
	@echo "  make clean — remove built docs"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
