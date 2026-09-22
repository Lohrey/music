"""Files copied verbatim from the upstream YuE repository's ``skills/yue2-music/scripts``
(Apache License 2.0, multimodal-art-projection/YuE).  They are standard-library-only
helpers that the upstream wheel does not ship, so they are vendored here to keep this
package self-contained.  ``yue2cpp_convert.py`` is yue2.cpp's ``convert.py`` (MIT), byte-identical
at the commit ``gguf_engine.YUE2CPP_PIN`` names; the GGUF engine drives it with its module globals
patched (paths), never by editing it.  See NOTICE at the repository root.
"""
