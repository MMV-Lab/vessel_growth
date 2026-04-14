"""Region Grow: A napari plugin for 3D vessel segmentation."""

__version__ = "0.1.0"

# Suppress "leaked semaphore" warnings on macOS.  These are caused by
# Python's multiprocessing resource_tracker being overly strict when
# napari / Qt spins up worker threads that import multiprocessing
# indirectly.  Setting the start method to "fork" (macOS default for
# Python < 3.14) avoids the resource_tracker entirely; on other
# platforms this is harmless.
import multiprocessing as _mp
import sys as _sys
import warnings as _warnings

if _sys.platform == "darwin":
    try:
        _mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass  # already set — ignore

    # Some macOS/Python combinations over-report leaked semaphores from
    # framework internals; avoid tracking semaphore resources entirely.
    try:
        from multiprocessing import resource_tracker as _resource_tracker

        _rt_register = _resource_tracker.register
        _rt_unregister = _resource_tracker.unregister

        def _register(name, rtype):
            if rtype == "semaphore":
                return
            _rt_register(name, rtype)

        def _unregister(name, rtype):
            if rtype == "semaphore":
                return
            _rt_unregister(name, rtype)

        _resource_tracker.register = _register
        _resource_tracker.unregister = _unregister
    except Exception:
        pass

    # Python 3.11 on macOS may report false-positive leaked semaphore
    # warnings at interpreter shutdown when GUI/thread pools are active.
    _warnings.filterwarnings(
        "ignore",
        message=(
            r"resource_tracker: There appear to be \d+ leaked semaphore "
            r"objects to clean up at shutdown"
        ),
        category=UserWarning,
    )
