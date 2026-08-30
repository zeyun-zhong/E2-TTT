# -*- coding: utf-8 -*-
"""Baselines shipped with E2-TTT.
"""

import warnings

try:
    from e2_ttt.baselines import lact  # noqa: F401  (registers with HF Auto*)
except Exception as exc:  # pragma: no cover - depends on the installed fla
    warnings.warn(
        f"E2-TTT: the LaCT baseline could not be imported ({exc!r}); "
        "the E2-TTT models themselves are unaffected.",
        RuntimeWarning,
        stacklevel=2,
    )
    lact = None

__all__ = ['lact']
