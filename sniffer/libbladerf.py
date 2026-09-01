"""Minimal ctypes binding for libbladeRF 2.x.

Nuand's own Python bindings are not on PyPI and need a CFFI build step, which is
awkward on current CPython.  The shared library ships with the standard bladeRF
installer, exports a plain cdecl C ABI, and we only need about twenty of its 194
entry points, so binding it directly here keeps the application dependency-free
and portable across Python versions.

Only receive-side calls are bound.  `bladerf_sync_tx`, `bladerf_enable_module`
on a TX channel, and every other transmit entry point are deliberately absent so
that no code path in this application can key a transmitter.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int16,
    c_int32,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_uint,
    c_void_p,
)

# --------------------------------------------------------------------------
# enums / macros mirrored from libbladeRF.h and bladeRF2.h
# --------------------------------------------------------------------------

BLADERF_MODULE_RX = 0
BLADERF_MODULE_TX = 1

BLADERF_RX = 0  # bladerf_direction
BLADERF_TX = 1

# bladerf_channel_layout
BLADERF_RX_X1 = 0
BLADERF_TX_X1 = 1
BLADERF_RX_X2 = 2
BLADERF_TX_X2 = 3

# bladerf_format
BLADERF_FORMAT_SC16_Q11 = 0
BLADERF_FORMAT_SC16_Q11_META = 1
BLADERF_FORMAT_PACKET_META = 2

# bladerf_gain_mode
BLADERF_GAIN_DEFAULT = 0
BLADERF_GAIN_MGC = 1  # manual gain control
BLADERF_GAIN_FASTATTACK_AGC = 2
BLADERF_GAIN_SLOWATTACK_AGC = 3
BLADERF_GAIN_HYBRID_AGC = 4

# bladerf_correction
BLADERF_CORR_DCOFF_I = 0
BLADERF_CORR_DCOFF_Q = 1
BLADERF_CORR_PHASE = 2
BLADERF_CORR_GAIN = 3

# bladerf_clock_select
CLOCK_SELECT_ONBOARD = 0
CLOCK_SELECT_EXTERNAL = 1

# metadata status bits
BLADERF_META_STATUS_OVERRUN = 1 << 0
BLADERF_META_STATUS_UNDERRUN = 1 << 1
BLADERF_META_FLAG_RX_NOW = 1 << 31

# log verbosity
BLADERF_LOG_LEVEL_VERBOSE = 0
BLADERF_LOG_LEVEL_DEBUG = 1
BLADERF_LOG_LEVEL_INFO = 2
BLADERF_LOG_LEVEL_WARNING = 3
BLADERF_LOG_LEVEL_ERROR = 4
BLADERF_LOG_LEVEL_CRITICAL = 5
BLADERF_LOG_LEVEL_SILENT = 6


def CHANNEL_RX(n: int) -> int:
    """BLADERF_CHANNEL_RX(n) -- RX0 -> 0, RX1 -> 2."""
    return (n << 1) | 0x0


# --------------------------------------------------------------------------
# structs
# --------------------------------------------------------------------------

class BladerfVersion(Structure):
    _fields_ = [
        ("major", c_uint16),
        ("minor", c_uint16),
        ("patch", c_uint16),
        ("describe", c_char_p),
    ]

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class BladerfMetadata(Structure):
    _fields_ = [
        ("timestamp", c_uint64),
        ("flags", c_uint32),
        ("status", c_uint32),
        ("actual_count", c_uint),
        ("reserved", c_uint8 * 32),
    ]


class BladerfRange(Structure):
    _fields_ = [
        ("min", c_int64),
        ("max", c_int64),
        ("step", c_int64),
        ("scale", c_float),
    ]

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.min * self.scale, self.max * self.scale, self.step * self.scale)


class BladerfDevinfo(Structure):
    _fields_ = [
        ("backend", c_int),
        ("serial", c_char * 33),
        ("usb_bus", c_uint8),
        ("usb_addr", c_uint8),
        ("instance", c_uint),
        ("manufacturer", c_char * 33),
        ("product", c_char * 33),
    ]


# --------------------------------------------------------------------------
# library loading
# --------------------------------------------------------------------------

_SEARCH_PATHS_WINDOWS = [
    r"C:\Program Files\bladeRF\x64\bladeRF.dll",
    r"C:\Program Files\bladeRF\x86\bladeRF.dll",
    r"C:\Program Files (x86)\bladeRF\x64\bladeRF.dll",
]


class BladeRFError(RuntimeError):
    """A libbladeRF call returned a negative status code."""

    def __init__(self, fn: str, code: int, detail: str = ""):
        self.code = code
        msg = f"{fn} failed: {detail or 'error'} ({code})"
        super().__init__(msg)


def _find_library() -> str:
    env = os.environ.get("BLADERF_LIBRARY")
    if env and os.path.exists(env):
        return env
    if sys.platform == "win32":
        for p in _SEARCH_PATHS_WINDOWS:
            if os.path.exists(p):
                return p
        # fall back to whatever is on PATH
        return "bladeRF.dll"
    found = ctypes.util.find_library("bladeRF")
    if found:
        return found
    return "libbladeRF.so.2"


_lib = None


def lib():
    """Load (once) and return the configured CDLL handle."""
    global _lib
    if _lib is not None:
        return _lib

    path = _find_library()
    if sys.platform == "win32":
        # bladeRF.dll pulls in libusb-1.0.dll and pthreadVC2.dll from its own
        # directory; add it to the DLL search path or the load fails obscurely.
        d = os.path.dirname(path)
        if d and hasattr(os, "add_dll_directory") and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    try:
        handle = ctypes.CDLL(path)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise BladeRFError(
            "load",
            -1,
            f"could not load libbladeRF from {path!r} ({exc}). Install the bladeRF "
            "package or set BLADERF_LIBRARY to the shared library path.",
        ) from exc

    _declare(handle)
    _lib = handle
    return _lib


def _declare(L) -> None:
    """Attach argtypes/restype to every entry point we use."""
    dev_p = c_void_p

    sigs = {
        # ---- lifecycle -------------------------------------------------
        "bladerf_open": ([POINTER(dev_p), c_char_p], c_int),
        "bladerf_close": ([dev_p], None),
        "bladerf_get_device_list": ([POINTER(POINTER(BladerfDevinfo))], c_int),
        "bladerf_free_device_list": ([POINTER(BladerfDevinfo)], None),
        "bladerf_get_serial": ([dev_p, c_char_p], c_int),
        "bladerf_get_board_name": ([dev_p], c_char_p),
        "bladerf_get_fpga_size": ([dev_p, POINTER(c_int)], c_int),
        "bladerf_is_fpga_configured": ([dev_p], c_int),
        "bladerf_fw_version": ([dev_p, POINTER(BladerfVersion)], c_int),
        "bladerf_fpga_version": ([dev_p, POINTER(BladerfVersion)], c_int),
        "bladerf_version": ([POINTER(BladerfVersion)], None),
        "bladerf_device_speed": ([dev_p], c_int),
        "bladerf_strerror": ([c_int], c_char_p),
        "bladerf_log_set_verbosity": ([c_int], None),
        # ---- tuning ----------------------------------------------------
        "bladerf_set_frequency": ([dev_p, c_int, c_uint64], c_int),
        "bladerf_get_frequency": ([dev_p, c_int, POINTER(c_uint64)], c_int),
        "bladerf_set_sample_rate": ([dev_p, c_int, c_uint, POINTER(c_uint)], c_int),
        "bladerf_get_sample_rate": ([dev_p, c_int, POINTER(c_uint)], c_int),
        "bladerf_set_bandwidth": ([dev_p, c_int, c_uint, POINTER(c_uint)], c_int),
        "bladerf_get_bandwidth": ([dev_p, c_int, POINTER(c_uint)], c_int),
        "bladerf_set_gain": ([dev_p, c_int, c_int], c_int),
        "bladerf_get_gain": ([dev_p, c_int, POINTER(c_int)], c_int),
        "bladerf_set_gain_mode": ([dev_p, c_int, c_int], c_int),
        "bladerf_get_gain_mode": ([dev_p, c_int, POINTER(c_int)], c_int),
        "bladerf_get_gain_range": ([dev_p, c_int, POINTER(POINTER(BladerfRange))], c_int),
        "bladerf_set_correction": ([dev_p, c_int, c_int, c_int16], c_int),
        "bladerf_get_correction": ([dev_p, c_int, c_int, POINTER(c_int16)], c_int),
        # ---- streaming (RX only) ---------------------------------------
        "bladerf_enable_module": ([dev_p, c_int, c_bool], c_int),
        "bladerf_sync_config": (
            [dev_p, c_int, c_int, c_uint, c_uint, c_uint, c_uint],
            c_int,
        ),
        "bladerf_sync_rx": (
            [dev_p, c_void_p, c_uint, POINTER(BladerfMetadata), c_uint],
            c_int,
        ),
        "bladerf_get_timestamp": ([dev_p, c_int, POINTER(c_uint64)], c_int),
        # ---- bladeRF 2.0 specifics -------------------------------------
        "bladerf_get_rfic_temperature": ([dev_p, POINTER(c_float)], c_int),
        "bladerf_get_rfic_rssi": (
            [dev_p, c_int, POINTER(c_int32), POINTER(c_int32)],
            c_int,
        ),
        "bladerf_set_bias_tee": ([dev_p, c_int, c_bool], c_int),
        "bladerf_get_bias_tee": ([dev_p, c_int, POINTER(c_bool)], c_int),
        "bladerf_set_clock_select": ([dev_p, c_int], c_int),
        "bladerf_get_clock_select": ([dev_p, POINTER(c_int)], c_int),
        "bladerf_set_pll_enable": ([dev_p, c_bool], c_int),
        "bladerf_get_pll_enable": ([dev_p, POINTER(c_bool)], c_int),
        "bladerf_set_pll_refclk": ([dev_p, c_uint64], c_int),
        "bladerf_get_pll_refclk": ([dev_p, POINTER(c_uint64)], c_int),
        "bladerf_get_pll_lock_state": ([dev_p, POINTER(c_bool)], c_int),
        "bladerf_get_rf_port": ([dev_p, c_int, POINTER(c_char_p)], c_int),
        "bladerf_set_rf_port": ([dev_p, c_int, c_char_p], c_int),
    }
    for name, (argtypes, restype) in sigs.items():
        try:
            fn = getattr(L, name)
        except AttributeError:  # pragma: no cover - older library
            continue
        fn.argtypes = argtypes
        fn.restype = restype


def strerror(code: int) -> str:
    try:
        s = lib().bladerf_strerror(code)
        return s.decode("utf-8", "replace") if s else ""
    except Exception:  # pragma: no cover
        return ""


def check(code: int, fn: str) -> int:
    """Raise on a negative libbladeRF status, otherwise pass the code through."""
    if code is not None and code < 0:
        raise BladeRFError(fn, code, strerror(code))
    return code


def has(name: str) -> bool:
    """True when the loaded library exports `name`."""
    try:
        getattr(lib(), name)
        return True
    except AttributeError:
        return False


def library_version() -> str:
    v = BladerfVersion()
    lib().bladerf_version(byref(v))
    return str(v)


def list_devices() -> list[dict]:
    """Enumerate attached bladeRF devices without opening them."""
    L = lib()
    arr = POINTER(BladerfDevinfo)()
    n = L.bladerf_get_device_list(byref(arr))
    if n < 0:
        return []
    out = []
    try:
        for i in range(n):
            d = arr[i]
            out.append(
                {
                    "serial": d.serial.decode("ascii", "replace").strip("\x00"),
                    "usb_bus": d.usb_bus,
                    "usb_addr": d.usb_addr,
                    "instance": d.instance,
                    "product": d.product.decode("ascii", "replace").strip("\x00"),
                    "manufacturer": d.manufacturer.decode("ascii", "replace").strip("\x00"),
                }
            )
    finally:
        L.bladerf_free_device_list(arr)
    return out


__all__ = [n for n in dir() if n.startswith(("BLADERF", "CLOCK", "Bladerf"))] + [
    "lib",
    "check",
    "has",
    "strerror",
    "library_version",
    "list_devices",
    "CHANNEL_RX",
    "BladeRFError",
]
