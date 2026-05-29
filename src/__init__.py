"""Learning Agent Source Package.

Keep package initialization lightweight so importing ``src.*`` modules does
not eagerly load optional heavy dependencies like sentence-transformers,
NLTK, or scikit-learn.
"""