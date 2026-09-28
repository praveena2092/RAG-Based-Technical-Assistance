.PHONY: install install-pipeline scrape-course scrape-discourse chunk embed upload serve eval test pipeline

install:            ## runtime deps only
	pip install -r requirements.txt

install-pipeline:   ## runtime + scraping/chunking deps + pytest
	pip install -r requirements-pipeline.txt

# ── RAG steps, in order ──────────────────────────────────────────────────
scrape-course:      ## Step 1a: clone course repo
	python -m tds_rag.ingestion.course_scraper

scrape-discourse:   ## Step 1b: scrape Discourse threads (needs cookies in .env)
	python -m tds_rag.ingestion.discourse_scraper

chunk:              ## Step 2: markdown -> data/processed/chunks.json
	python -m tds_rag.chunking.chunker

embed:              ## Step 3: chunks + threads -> data/embeddings/vectors.json
	python -m tds_rag.embedding.embedder

upload:             ## Step 4: vectors -> Qdrant
	python -m tds_rag.storage.qdrant_uploader

serve:              ## Steps 5-6: run retrieval + generation API on :8000
	uvicorn api.main:app --reload

eval:               ## Step 7: promptfoo evaluation (API must be running)
	promptfoo eval -c evaluation/promptfoo.yaml --clear-cache

pipeline: scrape-course scrape-discourse chunk embed upload   ## Steps 1-4 end to end

test:
	pytest -q
