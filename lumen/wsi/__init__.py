"""Whole-slide inference: tiling and tile scoring.

``segment_classifier`` tiles a slide and scores every tile that passes the
tissue filter; ``slide_benchmark`` pools cached tile embeddings into slide
scores. Tiling and scoring are deliberately separate stages, so prompts can be
swept without re-reading a slide.
"""
