"""NexusFeed — real-time AI recommendation and personalization engine."""
import os

# faiss and lightgbm each bundle their own OpenMP runtime. The API process
# loads both (retrieval + ranking share one process to hit the 50ms budget),
# and on macOS/arm64 wheels the two bundled libomp copies collide and
# segfault lightgbm's native training/inference calls — reliably reproduced
# by `import faiss` before `import lightgbm`. Setting KMP_DUPLICATE_LIB_OK
# does NOT fix this particular crash (verified); what does is forcing
# lightgbm's OpenMP runtime to initialize first, before faiss's. Since this
# module is the first thing executed for any `nexusfeed.*` import, doing it
# here guarantees the safe order regardless of which submodule imports which
# library first elsewhere in the codebase.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import lightgbm as _lightgbm  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
