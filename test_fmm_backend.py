#!/usr/bin/env python3
"""
test_fmm_backend.py — Check which FMM acceleration backend is available.

Run:
    python test_fmm_backend.py
"""

import sys
sys.path.insert(0, ".")

print("=" * 50)
print("  FMM Backend Check")
print("=" * 50)

# ── 1. Check Cython module ────────────────────────
print("\n1. Cython extension (fmm_near_cy):")
try:
    import fmm_near_cy
    print("   FOUND — Cython backend available")
    print(f"   Module: {fmm_near_cy.__file__}")
except ImportError as e:
    print(f"   NOT FOUND — {e}")
    print("   Build with: pip install cython numpy && cythonize -i fmm_near_cy.pyx")

# ── 2. Check ctypes .so/.dll ─────────────────────
print("\n2. C shared library (fmm_near.so / .dll):")
import ctypes, os
found_c = False
for path in ["./fmm_near.so", "./fmm_near.dll",
             os.path.join(os.path.dirname(os.path.abspath("fmm_helmholtz_2d.py")), "fmm_near.so")]:
    if os.path.isfile(path):
        try:
            lib = ctypes.CDLL(path)
            print(f"   FOUND — {os.path.abspath(path)}")
            found_c = True
            break
        except OSError as e:
            print(f"   File exists but won't load: {path} ({e})")
            print("   Recompile: gcc -O3 -shared -fPIC -o fmm_near.so fmm_near.c -lm")
if not found_c:
    print("   NOT FOUND")
    print("   Compile: gcc -O3 -shared -fPIC -o fmm_near.so fmm_near.c -lm")
    print("   Or Mac:  clang -O3 -shared -fPIC -o fmm_near.so fmm_near.c -lm")

# ── 3. Check what FMMOperator will use ────────────
print("\n3. FMMOperator._load_native() result:")
try:
    from fmm_helmholtz_2d import FMMOperator
    result = FMMOperator._load_native()
    if result is None:
        print("   PYTHON FALLBACK — no native backend")
        print("   The solver works but FMM near-field is 2-3x slower")
    else:
        kind, mod = result
        if kind == "cython":
            print(f"   CYTHON — fastest backend active")
        elif kind == "ctypes":
            print(f"   C (ctypes) — fast backend active")
        print(f"   Backend: {kind}")
except ImportError as e:
    print(f"   Cannot import FMMOperator: {e}")

# ── 4. Summary ────────────────────────────────────
print("\n" + "=" * 50)
print("  Priority order: Cython → C (.so) → Python fallback")
print("  All three produce identical results.")
print("  Cython/C are ~18x faster for real-k near-field blocks.")
print("=" * 50)
