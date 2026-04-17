"""
Convention self-check.

For each of the five segment types we draw a canonical simple case,
pass it through the solver's panel builder, and assert that the
final panel has its normal pointing in the direction the solver's
internal semantics expect.

The solver's internal convention: panel.normal points TOWARD the plus side.
  TYPE 1: plus = virtual sheet region  (normal direction physically arbitrary)
  TYPE 2: plus = PEC interior          (normal should point INTO the conductor)
  TYPE 3: plus = IPN1 dielectric       (normal should point INTO the dielectric)
  TYPE 4: plus = IPN1 dielectric       (normal should point INTO the dielectric)
  TYPE 5: plus = IPN1                  (normal should point INTO the IPN1 region)

User-facing convention: user draws the normal (from endpoint order)
pointing into air for types 2/3, and into the IPN1 side for types 4/5.

After the internal convention flip, TYPE 2/3 normals should come out
pointing INTO the solid/dielectric (opposite of how the user drew),
while TYPE 4/5 normals should come out pointing toward IPN1 (same
as how the user drew).
"""

import numpy as np
from rcs_solver import _build_panels, _unit_scale_to_meters


def one_panel_snapshot(seg_type, ipn1=1, ipn2=0, ibc_flag=0,
                       x1=0.0, y1=0.0, x2=1.0, y2=0.0):
    """Build a geometry snapshot with a single straight primitive."""
    return {
        "segments": [{
            "name": f"test_type{seg_type}",
            "seg_type": seg_type,
            "properties": [str(seg_type), "-20", "0.0",
                           str(ibc_flag), str(ipn1), str(ipn2)],
            "point_pairs": [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}],
        }],
        "ibcs": [["1", "100.0", "0.0"]] if ibc_flag else [],
        "dielectrics": [["1", "3.0", "0.0", "1.0", "0.0"],
                        ["2", "5.0", "0.0", "1.0", "0.0"]],
    }


def check(description, seg_type, drawn_dir, expected_normal_dir,
          ipn1=1, ipn2=0, ibc_flag=0):
    """Draw a segment in drawn_dir and check final normal direction."""
    # Canonical horizontal segment drawn left->right (dir = +x).
    # We rotate the endpoints according to drawn_dir to test different
    # draw orientations.
    dx, dy = drawn_dir
    snap = one_panel_snapshot(
        seg_type,
        ipn1=ipn1, ipn2=ipn2, ibc_flag=ibc_flag,
        x1=0.0, y1=0.0,
        x2=dx, y2=dy,
    )
    panels = _build_panels(snap, _unit_scale_to_meters("meters"),
                           min_wavelength=100.0, max_panels=10)
    n = panels[0].normal
    ex, ey = expected_normal_dir
    # Expected normal is a unit vector too; test dot product > 0.999.
    dot = n[0] * ex + n[1] * ey
    status = "PASS" if dot > 0.999 else "FAIL"
    print(f"  {status}: {description}")
    print(f"         drawn = {drawn_dir}, normal = ({n[0]:+.3f}, {n[1]:+.3f}),"
          f" expected ~= ({ex:+.1f}, {ey:+.1f})")
    return dot > 0.999


print("=" * 70)
print("Convention self-check")
print("=" * 70)

all_passed = True

print("\nTYPE 2 (PEC body): user draws normal INTO AIR; internal normal")
print("should end up pointing INTO the PEC interior after flip.")
print("Canonical: top of a PEC body drawn L->R, air is UP, PEC is DOWN.")
print("User draws with normal up; solver internal normal should be DOWN.")
all_passed &= check(
    "top surface of PEC body, drawn L->R",
    seg_type=2,
    drawn_dir=(1.0, 0.0),
    expected_normal_dir=(0.0, -1.0),
)
all_passed &= check(
    "bottom surface of PEC body, drawn R->L",
    seg_type=2,
    drawn_dir=(-1.0, 0.0),
    expected_normal_dir=(0.0, +1.0),
)

print("\nTYPE 3 (air/dielectric): user draws normal INTO AIR; internal")
print("normal should end up pointing INTO the dielectric after flip.")
all_passed &= check(
    "top of dielectric body, drawn L->R (normal user-side = UP = air)",
    seg_type=3, ipn1=1,
    drawn_dir=(1.0, 0.0),
    expected_normal_dir=(0.0, -1.0),
)
all_passed &= check(
    "bottom of dielectric body, drawn R->L (normal user-side = DOWN = air)",
    seg_type=3, ipn1=1,
    drawn_dir=(-1.0, 0.0),
    expected_normal_dir=(0.0, +1.0),
)

print("\nTYPE 4 (dielectric/PEC): user draws normal FROM PEC INTO IPN1")
print("dielectric.  No flip; internal normal should still point into IPN1.")
all_passed &= check(
    "PEC-backed dielectric coating on top, drawn L->R, normal UP into diel",
    seg_type=4, ipn1=1,
    drawn_dir=(1.0, 0.0),
    expected_normal_dir=(0.0, +1.0),
)
all_passed &= check(
    "PEC-backed coating on bottom, drawn R->L, normal DOWN into diel",
    seg_type=4, ipn1=1,
    drawn_dir=(-1.0, 0.0),
    expected_normal_dir=(0.0, -1.0),
)

print("\nTYPE 5 (dielectric/dielectric): user draws normal INTO IPN1.")
print("No flip; internal normal should still point into IPN1.")
all_passed &= check(
    "diel/diel interface, drawn L->R, normal UP into IPN1 (diel flag 1)",
    seg_type=5, ipn1=1, ipn2=2,
    drawn_dir=(1.0, 0.0),
    expected_normal_dir=(0.0, +1.0),
)
all_passed &= check(
    "diel/diel interface, drawn R->L, normal DOWN into IPN1",
    seg_type=5, ipn1=1, ipn2=2,
    drawn_dir=(-1.0, 0.0),
    expected_normal_dir=(0.0, -1.0),
)

print("\nTYPE 1 (free sheet): no flip, user endpoint order preserved.")
print("Normal direction is physically arbitrary for a symmetric card.")
all_passed &= check(
    "free resistive card drawn L->R, normal UP (user's choice preserved)",
    seg_type=1, ibc_flag=1,
    drawn_dir=(1.0, 0.0),
    expected_normal_dir=(0.0, +1.0),
)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
print("=" * 70)
