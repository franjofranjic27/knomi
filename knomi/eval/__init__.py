"""Retrieval evaluation harness.

Measures how well the current configuration (embedding model × chunking strategy
× vector store) retrieves the right documents for a set of gold questions, so
that profile choices become measurable instead of guessed.

Public surface:
- :func:`knomi.eval.dataset.load_eval_set`
- :func:`knomi.eval.runner.run_eval`
"""
