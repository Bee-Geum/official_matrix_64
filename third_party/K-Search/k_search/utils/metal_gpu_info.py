"""Metal / Apple Silicon GPU auto-detection helpers.

These utilities are best-effort and must be safe to import on non-macOS hosts.
They are used for MLX tasks to provide an accurate hardware profile in prompts
without requiring a user-supplied --target-gpu hint.
"""

from __future__ import annotations

from functools import lru_cache


# Best-effort mapping for Metal GPU families.
# The integer codes are Metal constants; they may vary by OS / PyObjC bindings.
# This table is intentionally partial and should degrade gracefully to Unknown.
#
# Tuple layout:
#   (label, occupancy_model, bandwidth_estimate, memory_type, has_mpp)
_GPU_FAMILY_INFO: dict[int, tuple[str, str, str, str, bool]] = {
    # Apple GPU families (approx):
    1007: ("Apple 7", "simdgroup32", "~200-400 GB/s", "Unified", False),
    1008: ("Apple 8", "simdgroup32", "~200-600 GB/s", "Unified", False),
    1009: ("Apple 9", "simdgroup32", "~200-600+ GB/s", "Unified", True),
    1010: ("Apple 10", "simdgroup32", "~200-600+ GB/s", "Unified", True),
}


@lru_cache(maxsize=1)
def _get_gpu_core_count() -> int | None:
    """Best-effort GPU core count on macOS.

    Uses `system_profiler SPDisplaysDataType -json` and attempts to parse the
    reported core count. Returns None if unavailable.
    """

    import json
    import subprocess

    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        obj = json.loads(out.stdout)
        displays = obj.get("SPDisplaysDataType") or obj.get("SPDisplaysDataTypeData")
        if not isinstance(displays, list):
            return None
        for d in displays:
            if not isinstance(d, dict):
                continue
            # Common keys seen across macOS versions / devices.
            for key in (
                "sppci_cores",
                "spdisplays_cores",
                "spdisplays_num_cores",
                "spdisplays_gpu_cores",
                "spdisplays_total_cores",
            ):
                v = d.get(key)
                if v is None:
                    continue
                try:
                    if isinstance(v, str):
                        v = "".join(ch for ch in v if ch.isdigit())
                    cores = int(v)
                    if cores > 0:
                        return cores
                except Exception:
                    continue
        return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_metal_device_name() -> str:
    """Return Metal default device name, or empty string if unavailable."""

    try:
        import Metal  # type: ignore[import-untyped]
    except ImportError:
        return ""

    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return ""
        name = device.name()
        return str(name or "")
    except Exception:
        return ""


@lru_cache(maxsize=1)
def get_gpu_info() -> str:
    """Query the Metal device and return a formatted hardware profile string.

    Returns an empty string if Metal is unavailable (e.g. Linux, CI).
    """

    try:
        import Metal  # type: ignore[import-untyped]
    except ImportError:
        return ""

    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return ""
    except Exception:
        return ""

    def _safe_device_call(method_name: str, default):
        """Best-effort call for optional/OS-version-dependent Metal device methods."""
        try:
            fn = getattr(device, method_name, None)
            if not callable(fn):
                return default
            return fn()
        except Exception:
            return default

    name: str = str(_safe_device_call("name", "") or "")
    max_tg = _safe_device_call("maxThreadsPerThreadgroup", None)
    max_threads_per_tg: int = max_tg.width if hasattr(max_tg, "width") else 1024
    tg_mem: int = int(_safe_device_call("maxThreadgroupMemoryLength", 0) or 0)
    unified: bool = bool(_safe_device_call("hasUnifiedMemory", False))
    working_set_raw = _safe_device_call("recommendedMaxWorkingSetSize", 0)
    try:
        working_set_gb = float(working_set_raw) / (1024**3)
    except Exception:
        working_set_gb = 0.0

    # Detect highest supported GPU family
    family_code: int | None = None
    family_label = "Unknown"
    for code in sorted(_GPU_FAMILY_INFO.keys(), reverse=True):
        try:
            if device.supportsFamily_(code):
                family_code = code
                family_label = _GPU_FAMILY_INFO[code][0]
                break
        except Exception:
            continue

    # Get system RAM
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        ram_gb = int(out.stdout.strip()) / (1024**3)
    except Exception:
        ram_gb = working_set_gb

    gpu_cores = _get_gpu_core_count()

    # Derive family-specific properties
    if family_code and family_code in _GPU_FAMILY_INFO:
        _, occ_model, bw_est, mem_type, has_mpp = _GPU_FAMILY_INFO[family_code]
    else:
        occ_model = "Unknown"
        bw_est = "Unknown"
        mem_type = "Unknown"
        has_mpp = False

    is_m1 = family_code == 1007
    has_async_copy = family_code is not None and family_code >= 1007
    has_simdgroup_matrix = family_code is not None and family_code >= 1007

    lines: list[str] = [
        "═══ TARGET GPU (auto-detected) ═══════════════════════════════════════════",
        f"  Chip:                  {name}",
    ]
    if gpu_cores is not None:
        lines.append(f"  GPU cores:             {gpu_cores}")
    lines += [
        f"  GPU family:            {family_label}",
        f"  Occupancy model:       {occ_model}",
        f"  Max threads/TG:        {max_threads_per_tg}",
        f"  Threadgroup memory:    {tg_mem} bytes ({tg_mem // 1024} KB)",
        f"  Unified memory:        {'Yes' if unified else 'No'} — {ram_gb:.0f} GB",
        f"  Memory bandwidth:      {bw_est} ({mem_type})",
        f"  SIMD width:            32",
        f"  simdgroup_matrix:      {'Yes' if has_simdgroup_matrix else 'No'} (Apple 7+)",
    ]
    if has_async_copy:
        async_note = "Yes (Apple 7+)"
        if is_m1:
            async_note += ", ⚠ M1 hardware bug — copied data MUST be read before kernel ends"
        lines.append(f"  simdgroup_async_copy:  {async_note}")
    else:
        lines.append("  simdgroup_async_copy:  No (requires Apple 7+)")
    lines += [
        f"  MetalPerformancePrimitives: {'Yes (Apple 9+ / Metal 4)' if has_mpp else 'No (requires Apple 9+)'}",
        "  F16/F32 ALU parity:    Yes (256 OPs/core-cycle both, M1+)",
        (
            "  Low-occ F32 penalty:   Yes — 2.9× slower FFMA than F16 at min occupancy"
            if (family_code and family_code >= 1007)
            else "  Low-occ F32 penalty:   Unknown"
        ),
        "  Register spill target: Device memory (not threadgroup) — all M-series",
        "═══════════════════════════════════════════════════════════════════════════",
        "",
    ]
    return "\n".join(lines) + "\n"
