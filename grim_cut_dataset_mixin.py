from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from grim_dataset import RcsGrid

POLARIZATION_DISPLAY_ORDER = ("VV", "TE", "HH", "TM", "VH", "HV")
_POLARIZATION_DISPLAY_RANK = {
    polarization: index for index, polarization in enumerate(POLARIZATION_DISPLAY_ORDER)
}


def _polarization_display_sort_key(value: object, original_index: int) -> tuple[int, int]:
    label = str(value).strip().upper()
    rank = _POLARIZATION_DISPLAY_RANK.get(label, len(POLARIZATION_DISPLAY_ORDER))
    return rank, original_index


def _sorted_polarization_indices(values, indices) -> list[int]:
    return sorted(
        (int(idx) for idx in indices),
        key=lambda idx: _polarization_display_sort_key(values[idx], idx),
    )


def _sorted_polarization_values(values) -> list:
    return [values[idx] for idx in _sorted_polarization_indices(values, range(len(values)))]


class AxisCropDialog(QDialog):
    """Single dialog for axis crop: shows per-axis min/max spinboxes with live shape preview."""

    def __init__(
        self,
        dataset: RcsGrid,
        n_datasets: int = 1,
        presel_az=None,
        presel_el=None,
        presel_freq=None,
        presel_pol=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Axis Crop")
        self._dataset = dataset
        layout = QVBoxLayout(self)

        desc = QLabel(
            f"Cropping {n_datasets} dataset(s).  Adjust ranges for each axis — "
            "leave at full extent to keep all samples on that axis."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)

        def _make_spin(lo, hi, val):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(6)
            s.setValue(val)
            return s

        az_lo = float(dataset.azimuths.min())
        az_hi = float(dataset.azimuths.max())
        el_lo = float(dataset.elevations.min())
        el_hi = float(dataset.elevations.max())
        fr_lo = float(dataset.frequencies.min())
        fr_hi = float(dataset.frequencies.max())

        self._az_lo_full, self._az_hi_full = az_lo, az_hi
        self._el_lo_full, self._el_hi_full = el_lo, el_hi
        self._fr_lo_full, self._fr_hi_full = fr_lo, fr_hi

        self.spin_az_min = _make_spin(-1e9, 1e9, az_lo)
        self.spin_az_max = _make_spin(-1e9, 1e9, az_hi)
        self.spin_el_min = _make_spin(-1e9, 1e9, el_lo)
        self.spin_el_max = _make_spin(-1e9, 1e9, el_hi)
        self.spin_fr_min = _make_spin(-1e9, 1e9, fr_lo)
        self.spin_fr_max = _make_spin(-1e9, 1e9, fr_hi)

        def _info(n, lo, hi):
            return QLabel(f"  {n} samples  ({lo:.6g} – {hi:.6g})")

        grid.addWidget(QLabel("Azimuth"), 0, 0)
        grid.addWidget(QLabel("Min"), 0, 1)
        grid.addWidget(self.spin_az_min, 0, 2)
        grid.addWidget(QLabel("Max"), 0, 3)
        grid.addWidget(self.spin_az_max, 0, 4)
        grid.addWidget(_info(len(dataset.azimuths), az_lo, az_hi), 0, 5)

        grid.addWidget(QLabel("Elevation"), 1, 0)
        grid.addWidget(QLabel("Min"), 1, 1)
        grid.addWidget(self.spin_el_min, 1, 2)
        grid.addWidget(QLabel("Max"), 1, 3)
        grid.addWidget(self.spin_el_max, 1, 4)
        grid.addWidget(_info(len(dataset.elevations), el_lo, el_hi), 1, 5)

        grid.addWidget(QLabel("Frequency"), 2, 0)
        grid.addWidget(QLabel("Min"), 2, 1)
        grid.addWidget(self.spin_fr_min, 2, 2)
        grid.addWidget(QLabel("Max"), 2, 3)
        grid.addWidget(self.spin_fr_max, 2, 4)
        grid.addWidget(_info(len(dataset.frequencies), fr_lo, fr_hi), 2, 5)

        layout.addLayout(grid)

        pol_group = QGroupBox("Polarization  (check = keep)")
        pol_row = QHBoxLayout(pol_group)
        self._pol_checks: list[tuple[QCheckBox, object]] = []
        for pol in _sorted_polarization_values(dataset.polarizations):
            chk = QCheckBox(str(pol))
            chk.setChecked(True)
            chk.toggled.connect(self._update_preview)
            pol_row.addWidget(chk)
            self._pol_checks.append((chk, pol))
        pol_row.addStretch(1)
        layout.addWidget(pol_group)

        self._lbl_preview = QLabel()
        layout.addWidget(self._lbl_preview)

        btn_reset = QPushButton("Reset to Full Range")
        btn_reset.clicked.connect(self._reset)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(btn_reset)
        bottom.addStretch(1)
        bottom.addWidget(btn_box)
        layout.addLayout(bottom)

        for spin in (
            self.spin_az_min, self.spin_az_max,
            self.spin_el_min, self.spin_el_max,
            self.spin_fr_min, self.spin_fr_max,
        ):
            spin.valueChanged.connect(self._update_preview)

        # Pre-fill from parameter list selections (if any)
        self._prefill_axis(self.spin_az_min, self.spin_az_max, presel_az)
        self._prefill_axis(self.spin_el_min, self.spin_el_max, presel_el)
        self._prefill_axis(self.spin_fr_min, self.spin_fr_max, presel_freq)
        if presel_pol is not None:
            presel_strs = {str(v) for v in presel_pol}
            for chk, pol in self._pol_checks:
                chk.setChecked(str(pol) in presel_strs)

        self._update_preview()

    @staticmethod
    def _prefill_axis(spin_min, spin_max, values) -> None:
        if not values:
            return
        arr = np.asarray(values, dtype=float)
        spin_min.setValue(float(arr.min()))
        spin_max.setValue(float(arr.max()))

    def _reset(self) -> None:
        self.spin_az_min.setValue(self._az_lo_full)
        self.spin_az_max.setValue(self._az_hi_full)
        self.spin_el_min.setValue(self._el_lo_full)
        self.spin_el_max.setValue(self._el_hi_full)
        self.spin_fr_min.setValue(self._fr_lo_full)
        self.spin_fr_max.setValue(self._fr_hi_full)
        for chk, _ in self._pol_checks:
            chk.setChecked(True)

    @staticmethod
    def _count_in_range(arr, lo, hi, tol: float = 1e-6) -> int:
        a = np.asarray(arr, dtype=float)
        return int(np.sum((a >= lo - tol) & (a <= hi + tol)))

    def _update_preview(self) -> None:
        ds = self._dataset
        n_az = self._count_in_range(ds.azimuths, self.spin_az_min.value(), self.spin_az_max.value())
        n_el = self._count_in_range(ds.elevations, self.spin_el_min.value(), self.spin_el_max.value())
        n_fr = self._count_in_range(ds.frequencies, self.spin_fr_min.value(), self.spin_fr_max.value())
        n_pol = sum(1 for chk, _ in self._pol_checks if chk.isChecked())
        orig = f"{len(ds.azimuths)} × {len(ds.elevations)} × {len(ds.frequencies)} × {len(ds.polarizations)}"
        result = f"{n_az} × {n_el} × {n_fr} × {n_pol}"
        self._lbl_preview.setText(
            f"Result (reference dataset):  {orig}  →  {result}  (az × el × freq × pol)"
        )

    def get_crop_params(self) -> dict:
        """Return kwargs suitable for RcsGrid.axis_crop()."""
        ds = self._dataset
        tol = 1e-6

        def _range_or_none(lo, hi, full_lo, full_hi):
            if lo <= full_lo + tol and hi >= full_hi - tol:
                return None
            return (lo, hi)

        az_range = _range_or_none(
            self.spin_az_min.value(), self.spin_az_max.value(),
            self._az_lo_full, self._az_hi_full,
        )
        el_range = _range_or_none(
            self.spin_el_min.value(), self.spin_el_max.value(),
            self._el_lo_full, self._el_hi_full,
        )
        fr_range = _range_or_none(
            self.spin_fr_min.value(), self.spin_fr_max.value(),
            self._fr_lo_full, self._fr_hi_full,
        )

        checked_pols = {str(pol) for chk, pol in self._pol_checks if chk.isChecked()}
        all_pols = {str(pol) for _, pol in self._pol_checks}
        if checked_pols >= all_pols or not checked_pols:
            pol_values = None
        else:
            pol_values = [pol for _, pol in self._pol_checks if str(pol) in checked_pols]

        return {
            "azimuth_range": az_range,
            "elevation_range": el_range,
            "frequency_range": fr_range,
            "polarizations": pol_values,
        }


class AlignDialog(QDialog):
    """Choose alignment mode when aligning datasets to a reference."""

    def __init__(self, ref_name: str, n_others: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align Datasets")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Reference: <b>{ref_name}</b>  —  aligning {n_others} other dataset(s) to it."
        ))

        grp = QGroupBox("Alignment Mode")
        grp_layout = QVBoxLayout(grp)
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in both datasets (exact match, no interpolation)"
        )
        self._radio_interp = QRadioButton(
            "Interpolate — linearly interpolate to the reference axes (no extrapolation)"
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        grp_layout.addWidget(self._radio_intersect)
        grp_layout.addWidget(self._radio_interp)
        layout.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "interp" if self._radio_interp.isChecked() else "intersect"


class ScaleDialog(QDialog):
    """Scale RCS values by a linear multiplier or a dB offset."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale Dataset")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Apply a scaling factor to all RCS values:"))

        self._btn_group = QButtonGroup(self)
        self._radio_linear = QRadioButton("Linear multiplier:")
        self._radio_db = QRadioButton("dB offset (applied as a log-power shift, e.g. +3 dB ≈ ×2.0):")
        self._radio_linear.setChecked(True)
        self._btn_group.addButton(self._radio_linear, 0)
        self._btn_group.addButton(self._radio_db, 1)

        self._spin_linear = QDoubleSpinBox()
        self._spin_linear.setRange(1e-12, 1e12)
        self._spin_linear.setDecimals(6)
        self._spin_linear.setValue(1.0)

        self._spin_db = QDoubleSpinBox()
        self._spin_db.setRange(-300.0, 300.0)
        self._spin_db.setDecimals(4)
        self._spin_db.setSingleStep(1.0)
        self._spin_db.setValue(0.0)
        self._spin_db.setEnabled(False)

        row1 = QHBoxLayout()
        row1.addWidget(self._radio_linear)
        row1.addWidget(self._spin_linear)
        row2 = QHBoxLayout()
        row2.addWidget(self._radio_db)
        row2.addWidget(self._spin_db)
        layout.addLayout(row1)
        layout.addLayout(row2)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._radio_linear.toggled.connect(self._update_enabled)

    def _update_enabled(self, linear_checked: bool) -> None:
        self._spin_linear.setEnabled(linear_checked)
        self._spin_db.setEnabled(not linear_checked)

    def get_factor(self) -> complex:
        """Return the complex linear multiplier to apply to RCS."""
        if self._radio_linear.isChecked():
            return complex(self._spin_linear.value())
        return complex(10.0 ** (self._spin_db.value() / 10.0))


class ResampleDialog(QDialog):
    """Resample dataset axes to a user-specified number of evenly-spaced points."""

    def __init__(self, dataset, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resample Dataset")
        self._dataset = dataset
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Interpolate each axis to a new number of evenly-spaced points "
            "(within the existing range — no extrapolation)."
        ))

        grid = QGridLayout()
        self._spins: dict[str, QDoubleSpinBox] = {}

        axes = [
            ("Azimuth", dataset.azimuths),
            ("Elevation", dataset.elevations),
            ("Frequency", dataset.frequencies),
        ]
        for row_idx, (label, arr) in enumerate(axes):
            n = len(arr)
            lo, hi = float(arr.min()), float(arr.max())
            spin = QDoubleSpinBox()
            spin.setRange(2 if n > 1 else 1, 8192)
            spin.setDecimals(0)
            spin.setValue(float(n))
            spin.setEnabled(n > 1)
            grid.addWidget(QLabel(f"{label}:"), row_idx, 0)
            grid.addWidget(spin, row_idx, 1)
            info = f"currently {n} pts,  {lo:.6g} – {hi:.6g}"
            if n == 1:
                info += "  (single-point axis, locked)"
            grid.addWidget(QLabel(info), row_idx, 2)
            self._spins[label] = spin

        layout.addLayout(grid)
        layout.addWidget(QLabel(
            "Polarization axis is unchanged (interpolation requires identical polarizations)."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_target_counts(self) -> tuple[int, int, int]:
        return (
            int(self._spins["Azimuth"].value()),
            int(self._spins["Elevation"].value()),
            int(self._spins["Frequency"].value()),
        )


class ExportCsvDialog(QDialog):
    """Options for exporting RCS data to a CSV file."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to CSV")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Magnitude:"), 0, 0)
        self._combo_scale = QComboBox()
        self._combo_scale.addItem("Linear", "linear")
        self._combo_scale.addItem("dBsm", "dbsm")
        self._combo_scale.addItem("dBke", "dbke")
        self._combo_scale.addItem("Both (Linear + dBsm + dBke)", "both")
        grid.addWidget(self._combo_scale, 0, 1)

        layout.addLayout(grid)

        self._chk_phase = QCheckBox("Include phase column (degrees)")
        self._chk_phase.setChecked(False)
        layout.addWidget(self._chk_phase)

        layout.addWidget(QLabel(
            "Columns: azimuth, elevation, frequency, polarization, [magnitude], [phase].\n"
            "For dBke export, frequency-dependent conversion uses the dataset frequency axis.\n"
            "One row per sample — all combinations of selected axes."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_options(self) -> tuple[str, bool]:
        """Return (scale, include_phase)."""
        return (
            self._combo_scale.currentData(),
            self._chk_phase.isChecked(),
        )


class StatisticsDialog(QDialog):
    """Single dialog for statistics dataset: all options in one place."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics Dataset")
        layout = QVBoxLayout(self)

        params_grid = QGridLayout()

        params_grid.addWidget(QLabel("Statistic:"), 0, 0)
        self.combo_stat = QComboBox()
        self.combo_stat.addItems(["mean", "median", "min", "max", "std", "percentile"])
        params_grid.addWidget(self.combo_stat, 0, 1)

        params_grid.addWidget(QLabel("Percentile:"), 0, 2)
        self.spin_pct = QDoubleSpinBox()
        self.spin_pct.setRange(0.0, 100.0)
        self.spin_pct.setDecimals(1)
        self.spin_pct.setSingleStep(5.0)
        self.spin_pct.setValue(50.0)
        self.spin_pct.setEnabled(False)
        self.spin_pct.setToolTip("Only used when Statistic = percentile")
        params_grid.addWidget(self.spin_pct, 0, 3)

        params_grid.addWidget(QLabel("Domain:"), 1, 0)
        self.combo_domain = QComboBox()
        self.combo_domain.addItem("Magnitude (linear)", "magnitude")
        self.combo_domain.addItem("dBsm", "dbsm")
        self.combo_domain.addItem("dBke", "dbke")
        self.combo_domain.addItem("Complex", "complex")
        params_grid.addWidget(self.combo_domain, 1, 1, 1, 3)

        layout.addLayout(params_grid)

        axes_group = QGroupBox("Axes to Reduce")
        axes_row = QHBoxLayout(axes_group)
        self.chk_az = QCheckBox("Azimuth")
        self.chk_az.setChecked(True)
        self.chk_el = QCheckBox("Elevation")
        self.chk_el.setChecked(True)
        self.chk_freq = QCheckBox("Frequency")
        self.chk_freq.setChecked(True)
        self.chk_pol = QCheckBox("Polarization")
        self.chk_pol.setChecked(False)
        for chk in (self.chk_az, self.chk_el, self.chk_freq, self.chk_pol):
            axes_row.addWidget(chk)
        axes_row.addStretch(1)
        layout.addWidget(axes_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.combo_stat.currentTextChanged.connect(
            lambda t: self.spin_pct.setEnabled(t == "percentile")
        )

    def get_params(self) -> tuple[str, float, str, list[str]]:
        """Return (statistic, percentile, domain, axes)."""
        statistic = self.combo_stat.currentText()
        percentile = self.spin_pct.value()
        domain = self.combo_domain.currentData()
        axes = [
            name
            for chk, name in (
                (self.chk_az, "azimuth"),
                (self.chk_el, "elevation"),
                (self.chk_freq, "frequency"),
                (self.chk_pol, "polarization"),
            )
            if chk.isChecked()
        ]
        return statistic, percentile, domain, axes


def _resample_grid(dataset: "RcsGrid", n_az: int, n_el: int, n_freq: int) -> "RcsGrid":
    """Interpolate a dataset's numeric axes to the given sample counts."""
    from scipy.interpolate import RegularGridInterpolator

    az = np.asarray(dataset.azimuths, dtype=float)
    el = np.asarray(dataset.elevations, dtype=float)
    fr = np.asarray(dataset.frequencies, dtype=float)
    new_az = np.linspace(az[0], az[-1], n_az) if len(az) > 1 else az.copy()
    new_el = np.linspace(el[0], el[-1], n_el) if len(el) > 1 else el.copy()
    new_fr = np.linspace(fr[0], fr[-1], n_freq) if len(fr) > 1 else fr.copy()

    rcs = dataset.rcs
    n_pol = rcs.shape[3]
    new_rcs = np.empty((len(new_az), len(new_el), len(new_fr), n_pol), dtype=rcs.dtype)

    coords = np.array(np.meshgrid(new_az, new_el, new_fr, indexing="ij")).reshape(3, -1).T
    for p in range(n_pol):
        vol = rcs[:, :, :, p]
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        re = RegularGridInterpolator((az, el, fr), vol.real, **kw)(coords)
        im = RegularGridInterpolator((az, el, fr), vol.imag, **kw)(coords)
        new_rcs[:, :, :, p] = (re + 1j * im).reshape(len(new_az), len(new_el), len(new_fr))

    return RcsGrid(
        new_az,
        new_el,
        new_fr,
        dataset.polarizations,
        new_rcs,
        rcs_power=dataset.rcs_to_linear(new_rcs),
        rcs_domain=dataset.rcs_domain,
    )


def _dataset_with_rcs(
    dataset: "RcsGrid",
    rcs,
    *,
    rcs_power=None,
    rcs_domain: str | None = None,
) -> "RcsGrid":
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs,
        rcs_power=rcs_power,
        rcs_domain=(dataset.rcs_domain if rcs_domain is None else rcs_domain),
        units=dataset.units,
    )


def _write_dataset_csv(
    dataset: "RcsGrid",
    path: str,
    *,
    scale: str = "linear",
    sep: str = ",",
    include_phase: bool = False,
) -> None:
    """Write a flat az×el×freq×pol CSV from a dataset."""
    az = dataset.azimuths
    el = dataset.elevations
    fr = dataset.frequencies
    pol = dataset.polarizations
    rcs = dataset.rcs
    header = ["azimuth", "elevation", "frequency", "polarization"]
    if scale in ("linear", "both"):
        header.append("magnitude_linear")
    if scale in ("dbsm", "both"):
        header.append("magnitude_dbsm")
    if scale in ("dbke", "both"):
        header.append("magnitude_dbke")
    if include_phase:
        header.append("phase_deg")

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(sep.join(header) + "\n")
        for ai, az_v in enumerate(az):
            for ei, el_v in enumerate(el):
                for fi, fr_v in enumerate(fr):
                    for pi, pol_v in enumerate(pol):
                        val = rcs[ai, ei, fi, pi]
                        mag = float(dataset.rcs_to_linear(val))
                        row = [str(az_v), str(el_v), str(fr_v), str(pol_v)]
                        if scale in ("linear", "both"):
                            row.append(f"{mag:.10g}")
                        if scale in ("dbsm", "both"):
                            row.append(f"{float(dataset.rcs_to_dbsm(val)):.6f}")
                        if scale in ("dbke", "both"):
                            row.append(f"{float(dataset.rcs_to_dbke(val, fr_v)):.6f}")
                        if include_phase:
                            row.append(f"{np.degrees(np.angle(val)):.6f}")
                        f.write(sep.join(row) + "\n")


def _load_dataset_csv(path: str) -> "RcsGrid":
    """Load a dataset from a delimited text file exported by _write_dataset_csv()."""
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("missing CSV header row")

        field_map: dict[str, str] = {}
        for raw_name in reader.fieldnames:
            if raw_name is None:
                continue
            key = str(raw_name).strip().lower()
            if key and key not in field_map:
                field_map[key] = raw_name

        required = ["azimuth", "elevation", "frequency", "polarization"]
        missing = [name for name in required if name not in field_map]
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")

        has_linear = "magnitude_linear" in field_map
        has_dbsm = "magnitude_dbsm" in field_map
        has_dbke = "magnitude_dbke" in field_map
        if not has_linear and not has_dbsm and not has_dbke:
            raise ValueError("missing magnitude column (need magnitude_linear and/or magnitude_dbsm and/or magnitude_dbke)")
        has_phase = "phase_deg" in field_map

        def _cell(row: dict[str, str], key: str) -> str:
            source = field_map[key]
            raw = row.get(source, "")
            return str(raw).strip() if raw is not None else ""

        records: list[tuple[float, float, float, str, float, float]] = []
        freq_values_seen: list[float] = []
        pol_order: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            az_text = _cell(row, "azimuth")
            el_text = _cell(row, "elevation")
            fr_text = _cell(row, "frequency")
            pol_text = _cell(row, "polarization")
            if not (az_text or el_text or fr_text or pol_text):
                continue
            if not pol_text:
                raise ValueError(f"line {line_no}: polarization is blank")
            try:
                az = float(az_text)
                el = float(el_text)
                fr = float(fr_text)
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid axis value ({exc})") from exc
            freq_values_seen.append(fr)

            lin_value: float | None = None
            if has_linear:
                linear_text = _cell(row, "magnitude_linear")
                if linear_text:
                    try:
                        lin_value = float(linear_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_linear ({exc})") from exc
            if lin_value is None and has_dbsm:
                db_text = _cell(row, "magnitude_dbsm")
                if db_text:
                    try:
                        lin_value = float(10.0 ** (float(db_text) / 10.0))
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_dbsm ({exc})") from exc
            if lin_value is None and has_dbke:
                db_text = _cell(row, "magnitude_dbke")
                if db_text:
                    try:
                        dbke_val = float(db_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_dbke ({exc})") from exc
                    # Infer frequency units similarly to grim_dataset CSV/TXT loaders.
                    typical = float(np.nanmedian(np.abs(np.asarray(freq_values_seen, dtype=float)))) if freq_values_seen else float(abs(fr))
                    if typical >= 1.0e6:
                        freq_hz = fr
                    elif typical >= 1.0e3:
                        freq_hz = fr * 1.0e6
                    else:
                        freq_hz = fr * 1.0e9
                    if np.isfinite(freq_hz) and freq_hz > 0.0:
                        lin_value = float((2.998e8 / (2.0 * np.pi * freq_hz)) * (10.0 ** (dbke_val / 10.0)))
                    else:
                        lin_value = float("nan")
            if lin_value is None:
                lin_value = float("nan")
            elif np.isfinite(lin_value):
                lin_value = max(lin_value, 0.0)

            phase_rad = float("nan")
            if has_phase:
                phase_text = _cell(row, "phase_deg")
                if phase_text:
                    try:
                        phase_rad = float(np.deg2rad(float(phase_text)))
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid phase_deg ({exc})") from exc

            if pol_text not in pol_order:
                pol_order.append(pol_text)
            records.append((az, el, fr, pol_text, lin_value, phase_rad))

    if not records:
        raise ValueError("CSV contains no data rows")

    az_values = np.asarray(sorted({r[0] for r in records}), dtype=float)
    el_values = np.asarray(sorted({r[1] for r in records}), dtype=float)
    fr_values = np.asarray(sorted({r[2] for r in records}), dtype=float)
    pol_values = np.asarray(pol_order, dtype=object)

    az_index = {float(v): i for i, v in enumerate(az_values.tolist())}
    el_index = {float(v): i for i, v in enumerate(el_values.tolist())}
    fr_index = {float(v): i for i, v in enumerate(fr_values.tolist())}
    pol_index = {str(v): i for i, v in enumerate(pol_values.tolist())}

    shape = (len(az_values), len(el_values), len(fr_values), len(pol_values))
    power = np.full(shape, np.nan, dtype=np.float32)
    phase = np.full(shape, np.nan, dtype=np.float32)

    for az, el, fr, pol, lin_value, phase_rad in records:
        ai = az_index[float(az)]
        ei = el_index[float(el)]
        fi = fr_index[float(fr)]
        pi = pol_index[str(pol)]
        power[ai, ei, fi, pi] = np.float32(lin_value)
        phase[ai, ei, fi, pi] = np.float32(phase_rad)

    if not np.isfinite(power).any():
        raise ValueError("CSV contains no finite magnitude values")

    return RcsGrid(
        az_values,
        el_values,
        fr_values,
        pol_values,
        rcs_power=power,
        rcs_phase=phase,
        rcs_domain="power_phase",
        source_path=path,
    )


def _load_dataset_from_dropped_text(path: str) -> tuple["RcsGrid", str]:
    """Load dropped delimited files, including theta/phi text variants."""
    lower = path.lower()
    attempts = []
    if lower.endswith(".out"):
        attempts = [
            ("OUT", lambda: RcsGrid.load_out(path)),
        ]
    elif lower.endswith(".txt"):
        attempts = [
            ("theta/phi TXT", lambda: RcsGrid.load_theta_phi_txt(path)),
            ("delimited table", lambda: _load_dataset_csv(path)),
        ]
    elif lower.endswith(".csv"):
        attempts = [
            ("delimited table", lambda: _load_dataset_csv(path)),
            ("theta/phi CSV", lambda: RcsGrid.load_theta_phi_csv(path)),
        ]
    else:
        attempts = [("delimited table", lambda: _load_dataset_csv(path))]

    errors: list[str] = []
    for label, loader in attempts:
        try:
            dataset = loader()
            history = str(getattr(dataset, "history", "") or "").strip()
            if not history:
                history = f"Imported delimited text: {path}"
            return dataset, history
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise ValueError("; ".join(errors))


class TimeGateDialog(QDialog):
    """Parameters for time-domain gating of frequency-domain RCS data."""

    def __init__(self, dataset: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Time Gate")
        c = 2.998e8
        freqs = np.asarray(dataset.frequencies, dtype=float)
        units = dataset.units or {}
        freq_unit = str(units.get("frequency", "")).lower()
        if freq_unit == "hz":
            freq_hz = freqs
        elif freq_unit == "mhz":
            freq_hz = freqs * 1e6
        else:
            freq_hz = freqs * 1e9
        n = len(freq_hz)
        if n > 1:
            bw = freq_hz[-1] - freq_hz[0]
            df = bw / (n - 1)
            range_res = c / (2.0 * bw) if bw > 0 else float("inf")
            range_per_bin = c / (2.0 * n * df) if df > 0 else float("inf")
            max_range = range_per_bin * (n // 2)
        else:
            range_res = range_per_bin = max_range = float("inf")

        layout = QVBoxLayout(self)
        info = QLabel(
            f"Frequency points: {n}\n"
            f"Range resolution: {range_res:.4g} m   |   bin width: {range_per_bin:.4g} m\n"
            f"Max physical range: {max_range:.4g} m"
        )
        layout.addWidget(info)
        grid = QGridLayout()
        grid.addWidget(QLabel("Gate start (m):"), 0, 0)
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 1e6)
        self.spin_start.setDecimals(3)
        self.spin_start.setValue(0.0)
        grid.addWidget(self.spin_start, 0, 1)
        grid.addWidget(QLabel("Gate end (m):"), 1, 0)
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0.0, 1e6)
        self.spin_end.setDecimals(3)
        default_end = min(max_range * 0.5, 20.0) if max_range < float("inf") else 20.0
        self.spin_end.setValue(default_end)
        grid.addWidget(self.spin_end, 1, 1)
        grid.addWidget(QLabel("Window:"), 2, 0)
        self.combo = QComboBox()
        self.combo.addItems(["hanning", "hamming", "blackman", "blackmanharris", "boxcar"])
        grid.addWidget(self.combo, 2, 1)
        layout.addLayout(grid)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_params(self) -> tuple[float, float, str]:
        return self.spin_start.value(), self.spin_end.value(), self.combo.currentText()


def _apply_time_gate(
    dataset: RcsGrid, gate_start_m: float, gate_end_m: float, window_type: str
) -> RcsGrid:
    from scipy.signal import get_window

    c = 2.998e8
    freqs = np.asarray(dataset.frequencies, dtype=float)
    units = dataset.units or {}
    freq_unit = str(units.get("frequency", "")).lower()
    if freq_unit == "hz":
        freq_hz = freqs
    elif freq_unit == "mhz":
        freq_hz = freqs * 1e6
    else:
        freq_hz = freqs * 1e9

    n = len(freq_hz)
    if n < 2:
        raise ValueError("Need at least 2 frequency points for time gating.")
    bw = freq_hz[-1] - freq_hz[0]
    if bw <= 0:
        raise ValueError("Frequencies must be strictly increasing.")
    df = bw / (n - 1)

    # Range axis: R[k] = c·k / (2·N·Δf)
    range_bins = (c / (2.0 * n * df)) * np.arange(n)

    # Build gate mask (only physical first n//2 bins are gated; rest zeroed)
    gate_mask = np.zeros(n, dtype=float)
    phys = np.arange(n // 2)
    gate_idx = phys[(range_bins[phys] >= gate_start_m) & (range_bins[phys] <= gate_end_m)]

    if gate_idx.size == 0:
        raise ValueError(
            f"No range bins in [{gate_start_m:.3f}, {gate_end_m:.3f}] m. "
            f"Max physical range ≈ {range_bins[n // 2 - 1]:.3f} m."
        )

    if window_type == "boxcar" or gate_idx.size == 1:
        gate_mask[gate_idx] = 1.0
    else:
        gate_mask[gate_idx] = get_window(window_type, gate_idx.size)

    rcs_time = np.fft.ifft(dataset.rcs, axis=2)
    rcs_gated = np.fft.fft(
        rcs_time * gate_mask[np.newaxis, np.newaxis, :, np.newaxis], axis=2
    )
    return RcsGrid(
        dataset.azimuths, dataset.elevations, dataset.frequencies,
        dataset.polarizations, rcs_gated,
        rcs_power=np.abs(rcs_gated) ** 2,
        rcs_domain="complex_amplitude",
        units=dataset.units,
    )


def _is_supported_dataset_path(path: str) -> bool:
    lower = str(path).lower()
    return (
        lower.endswith(".grim")
        or lower.endswith(".csv")
        or lower.endswith(".txt")
        or lower.endswith(".out")
    )


def _recommended_loader_workers(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    if cpu_total <= 2:
        target = cpu_total
    else:
        target = cpu_total - 1
    return max(1, min(int(task_count), int(target)))


def _load_dataset_path_task(task: tuple[int, str]) -> dict[str, object]:
    index, path = task
    file_name = os.path.basename(path)
    dataset_name = os.path.splitext(file_name)[0]
    lower = path.lower()
    try:
        if lower.endswith(".grim"):
            dataset = RcsGrid.load(path)
            history = path
        elif (
            lower.endswith(".csv")
            or lower.endswith(".txt")
            or lower.endswith(".out")
        ):
            dataset, history = _load_dataset_from_dropped_text(path)
        else:
            return {
                "status": "ignored",
                "index": index,
                "path": path,
                "file_name": file_name,
                "error": "Unsupported file extension",
            }
    except Exception as exc:
        return {
            "status": "error",
            "index": index,
            "path": path,
            "file_name": file_name,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "index": index,
        "path": path,
        "file_name": file_name,
        "name": dataset_name,
        "history": history,
        "dataset": dataset,
    }


def _join_many_with_progress(
    grids: list[RcsGrid],
    *,
    tol: float = 1e-6,
    progress_cb=None,
) -> RcsGrid:
    checked = RcsGrid._ensure_grids(grids)
    total = len(checked)
    if total == 1:
        grid = checked[0]
        if progress_cb is not None:
            progress_cb(1, 1)
        return grid._new_grid(
            np.array(grid.azimuths, copy=True),
            np.array(grid.elevations, copy=True),
            np.array(grid.frequencies, copy=True),
            np.array(grid.polarizations, copy=True),
            rcs_power=np.array(grid.rcs_power, copy=True),
            rcs_phase=np.array(grid.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    az_union = RcsGrid._axis_union([grid.azimuths for grid in checked], tol=tol)
    el_union = RcsGrid._axis_union([grid.elevations for grid in checked], tol=tol)
    f_union = RcsGrid._axis_union([grid.frequencies for grid in checked], tol=tol)
    p_union = RcsGrid._axis_union([grid.polarizations for grid in checked], tol=0.0)

    shape = (len(az_union), len(el_union), len(f_union), len(p_union))
    joined_power = np.full(shape, np.nan, dtype=np.float32)
    joined_phase = np.full(shape, np.nan, dtype=np.float32)

    for idx, grid in enumerate(checked, start=1):
        az_idx = RcsGrid._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
        el_idx = RcsGrid._indices_for_axis_values(el_union, grid.elevations, tol=tol)
        f_idx = RcsGrid._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
        p_idx = RcsGrid._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
        if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
            raise ValueError("failed to align a dataset during join")
        joined_power[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_power
        joined_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_phase
        if progress_cb is not None:
            progress_cb(idx, total)

    last = checked[-1]
    return RcsGrid(
        az_union,
        el_union,
        f_union,
        p_union,
        rcs_power=joined_power,
        rcs_phase=joined_phase,
        rcs_domain="power_phase",
        source_path=last.source_path,
        history=last.history,
        units=dict(last.units),
    )


class _DatasetLoadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, tasks: list[tuple[int, str]], ignored_count: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._tasks = list(tasks)
        self._ignored_count = int(ignored_count)

    def run(self) -> None:
        total = len(self._tasks)
        loaded: list[dict[str, object]] = []
        failed: list[str] = []
        used_multiprocessing = False
        fallback_reason: str | None = None

        def _consume(result: dict[str, object], done_count: int) -> None:
            status = str(result.get("status", "error"))
            file_name = str(result.get("file_name", "dataset"))
            if status == "ok":
                loaded.append(result)
                self.progress.emit(done_count, total, f"Loaded {file_name}")
                return
            error_text = str(result.get("error", "Unknown error"))
            failed.append(f"{file_name} ({error_text})")
            self.progress.emit(done_count, total, f"Failed {file_name}")

        if total == 0:
            self.finished.emit(
                {
                    "loaded": loaded,
                    "failed": failed,
                    "ignored": self._ignored_count,
                    "used_multiprocessing": used_multiprocessing,
                    "fallback_reason": fallback_reason,
                    "total_supported": total,
                }
            )
            return

        if total == 1:
            _consume(_load_dataset_path_task(self._tasks[0]), 1)
        else:
            worker_count = _recommended_loader_workers(total)
            try:
                with ProcessPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(_load_dataset_path_task, task): task
                        for task in self._tasks
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        result = future.result()
                        done_count += 1
                        _consume(result, done_count)
                used_multiprocessing = True
            except Exception as exc:
                fallback_reason = str(exc)
                loaded.clear()
                failed.clear()
                for done_count, task in enumerate(self._tasks, start=1):
                    _consume(_load_dataset_path_task(task), done_count)

        self.finished.emit(
            {
                "loaded": loaded,
                "failed": failed,
                "ignored": self._ignored_count,
                "used_multiprocessing": used_multiprocessing,
                "fallback_reason": fallback_reason,
                "total_supported": total,
            }
        )


class _JoinDatasetsWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, grids: list[RcsGrid], tol: float = 1e-6, parent=None) -> None:
        super().__init__(parent)
        self._grids = list(grids)
        self._tol = float(tol)

    def run(self) -> None:
        total = max(1, len(self._grids))
        try:
            def _emit_progress(done_count: int, total_count: int) -> None:
                self.progress.emit(done_count, total_count, "Joining datasets")

            merged = _join_many_with_progress(self._grids, tol=self._tol, progress_cb=_emit_progress)
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.finished.emit({"ok": True, "merged": merged, "total": total})


class DatasetOpsMixin:
    def _ensure_background_worker_state(self) -> None:
        if hasattr(self, "_background_worker_thread"):
            return
        self._background_worker_thread: QThread | None = None
        self._background_worker: QObject | None = None
        self._background_worker_name = ""
        self._pending_join_names: list[str] | None = None

    def _background_job_active(self) -> bool:
        self._ensure_background_worker_state()
        thread = self._background_worker_thread
        return isinstance(thread, QThread) and thread.isRunning()

    def _try_start_background_job(self, job_name: str, worker: QObject) -> bool:
        self._ensure_background_worker_state()
        if self._background_job_active():
            active_name = self._background_worker_name or "Another background job"
            self.status.showMessage(f"{active_name} is still running. Please wait.")
            return False

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_background_thread_finished)

        self._background_worker_thread = thread
        self._background_worker = worker
        self._background_worker_name = job_name
        thread.start()
        return True

    def _on_background_thread_finished(self) -> None:
        self._background_worker_thread = None
        self._background_worker = None
        self._background_worker_name = ""

    def _on_load_worker_progress(self, done_count: int, total_count: int, detail: str) -> None:
        detail_text = str(detail).strip()
        if detail_text:
            self.status.showMessage(
                f"Loading datasets... {done_count}/{total_count} ({detail_text})"
            )
            return
        self.status.showMessage(f"Loading datasets... {done_count}/{total_count}")

    def _on_load_worker_finished(self, summary: dict[str, object]) -> None:
        loaded_entries_raw = summary.get("loaded", [])
        failed_entries_raw = summary.get("failed", [])
        ignored = int(summary.get("ignored", 0) or 0)
        fallback_reason = summary.get("fallback_reason")
        used_multiprocessing = bool(summary.get("used_multiprocessing", False))
        total_supported = int(summary.get("total_supported", 0) or 0)

        loaded_entries = [entry for entry in loaded_entries_raw if isinstance(entry, dict)]
        loaded_entries.sort(key=lambda item: int(item.get("index", 0)))
        failed = [str(item) for item in failed_entries_raw]

        loaded = 0
        for entry in loaded_entries:
            dataset = entry.get("dataset")
            if not isinstance(dataset, RcsGrid):
                file_name = str(entry.get("file_name", "dataset"))
                failed.append(f"{file_name} (worker returned invalid dataset)")
                continue
            name = str(entry.get("name", "dataset"))
            history = str(entry.get("history", ""))
            file_name = str(entry.get("file_name", ""))
            self._add_dataset_row(dataset, name, history, file_name=file_name)
            loaded += 1

        if failed:
            msg = f"Loaded {loaded} dataset(s)." if loaded else "No datasets loaded."
            msg += f" Failed: {', '.join(failed)}"
        elif loaded:
            msg = f"Loaded {loaded} dataset(s)."
        else:
            msg = "No datasets loaded."

        if ignored:
            msg += f" Ignored {ignored} unsupported file(s)."
        if fallback_reason:
            msg += " Multiprocessing unavailable; used single-worker fallback."
        elif used_multiprocessing and total_supported > 1:
            msg += " Loaded in parallel."
        self.status.showMessage(msg)

    def _on_join_worker_progress(self, done_count: int, total_count: int, _: str) -> None:
        self.status.showMessage(f"Joining datasets... {done_count}/{total_count}")

    def _on_join_worker_finished(self, payload: dict[str, object]) -> None:
        names = self._pending_join_names or []
        self._pending_join_names = None

        ok = bool(payload.get("ok", False))
        if not ok:
            self.status.showMessage(str(payload.get("error", "Join failed.")))
            return

        merged = payload.get("merged")
        if not isinstance(merged, RcsGrid):
            self.status.showMessage("Join failed: worker produced invalid output.")
            return

        if not names:
            names = ["Dataset"]
        new_name = " | ".join(names)
        history = f"Join (last selected wins overlap): {new_name}"
        self._add_dataset_row(merged, f"Join[{new_name}]", history, file_name="")
        self.status.showMessage(f"Join created. Overlap winner: {names[-1]}.")

    def _handle_files_dropped(self, paths: list[str]) -> None:
        tasks: list[tuple[int, str]] = []
        ignored = 0
        for index, raw_path in enumerate(paths):
            path = str(raw_path)
            if _is_supported_dataset_path(path):
                tasks.append((index, path))
            else:
                ignored += 1

        if not tasks:
            if ignored:
                self.status.showMessage(
                    "No supported dropped files. Supported: .grim, .csv, .txt, .out"
                )
            return

        worker = _DatasetLoadWorker(tasks, ignored_count=ignored)
        worker.progress.connect(self._on_load_worker_progress)
        worker.finished.connect(self._on_load_worker_finished)
        if not self._try_start_background_job("Dataset loading", worker):
            return
        self.status.showMessage(f"Loading datasets... 0/{len(tasks)}")

    def _add_dataset_row(self, dataset: RcsGrid, name: str, history: str, file_name: str | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, dataset)
        file_text = file_name or ""
        file_item = QTableWidgetItem(file_text)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        history_item = QTableWidgetItem(history)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, file_item)
        self.table.setItem(row, 2, history_item)

    def _on_dataset_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        self._update_dataset_selection_order([idx.row() for idx in selected])
        if not selected:
            self.active_dataset = None
            self._clear_param_lists()
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        dataset = item.data(Qt.UserRole) if item else None
        if not isinstance(dataset, RcsGrid):
            self.active_dataset = None
            self._clear_param_lists()
            return
        self.active_dataset = dataset
        self._populate_params(dataset)

    def _update_dataset_selection_order(self, selected_rows: list[int]) -> None:
        selected_set = set(selected_rows)
        previous_order = getattr(self, "_dataset_selection_order", [])
        order = [row for row in previous_order if row in selected_set]
        current_row = self.table.currentRow()

        for row in selected_rows:
            if row not in order:
                order.append(row)

        # Use the active row as the most-recent selection.
        if current_row in selected_set and current_row in order:
            order = [row for row in order if row != current_row] + [current_row]

        self._dataset_selection_order = order

    def _populate_params(self, dataset: RcsGrid) -> None:
        self._fill_list(self.list_pol, dataset.polarizations)
        self._fill_list(self.list_freq, dataset.frequencies)
        self._fill_list(self.list_elev, dataset.elevations)
        self._fill_list(self.list_az, dataset.azimuths)
        self._apply_default_param_selection()

    @staticmethod
    def _select_first_item(widget: QListWidget) -> None:
        if widget.count() <= 0:
            return
        widget.clearSelection()
        first = widget.item(0)
        if first is None:
            return
        first.setSelected(True)
        widget.setCurrentItem(first)

    def _apply_default_param_selection(self) -> None:
        widgets = (self.list_pol, self.list_freq, self.list_elev, self.list_az)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._select_first_item(self.list_pol)
            self._select_first_item(self.list_freq)
            self._select_first_item(self.list_elev)
            if self.list_az.count() > 0:
                self.list_az.selectAll()
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        # Refresh availability masks from selected polarization and trigger one autoplot update.
        self._on_polarization_selection_changed()

    def _fill_list(self, widget: QListWidget, values, indices=None) -> None:
        widget.blockSignals(True)
        widget.clear()
        if indices is None:
            indices = list(range(len(values)))
        else:
            indices = [int(idx) for idx in indices]
        if widget is getattr(self, "list_pol", None):
            indices = _sorted_polarization_indices(values, indices)
        for idx in indices:
            value = values[idx]
            item = QListWidgetItem(str(value))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, value)
            item.setData(Qt.UserRole + 1, int(idx))
            widget.addItem(item)
        widget.blockSignals(False)

    def _clear_param_lists(self) -> None:
        for widget in (self.list_pol, self.list_freq, self.list_elev, self.list_az):
            widget.clear()

    def _on_param_item_changed(self, item: QListWidgetItem, axis_name: str, widget: QListWidget) -> None:
        if self.active_dataset is None:
            return
        axis_arr = self.active_dataset.get_axis(axis_name)
        idx = item.data(Qt.UserRole + 1)
        if idx is None:
            return
        if idx < 0 or idx >= len(axis_arr):
            return
        old_value = axis_arr[idx]
        new_text = item.text()
        if axis_name == "polarization":
            new_value = new_text
        else:
            try:
                new_value = float(new_text)
            except ValueError:
                widget.blockSignals(True)
                item.setText(str(old_value))
                widget.blockSignals(False)
                return
        axis_arr[idx] = new_value
        item.setData(Qt.UserRole, new_value)

    def _selected_indices(self, widget: QListWidget) -> set[int]:
        indices = set()
        for item in widget.selectedItems():
            idx = item.data(Qt.UserRole + 1)
            if idx is not None:
                indices.add(int(idx))
        return indices

    def _selected_values(self, widget: QListWidget) -> list:
        values = []
        for item in widget.selectedItems():
            values.append(item.data(Qt.UserRole))
        return values

    def _indices_for_values(self, axis_arr, values, tol=1e-6) -> list[int] | None:
        axis_arr = np.asarray(axis_arr)
        indices: list[int] = []
        is_numeric_axis = np.issubdtype(axis_arr.dtype, np.number)
        for value in values:
            if is_numeric_axis and isinstance(value, (int, float, np.floating, np.integer)):
                matches = np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
            else:
                matches = np.where(axis_arr == value)[0]
            if matches.size == 0:
                return None
            indices.append(int(matches[0]))
        return indices

    def _selected_datasets(self) -> list[tuple[str, RcsGrid]]:
        datasets: list[tuple[str, RcsGrid]] = []
        selected = self.table.selectionModel().selectedRows()
        for model_index in selected:
            row = model_index.row()
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if isinstance(dataset, RcsGrid):
                datasets.append((item.text(), dataset))
        if not datasets and isinstance(self.active_dataset, RcsGrid):
            datasets.append(("Dataset", self.active_dataset))
        return datasets

    def _selected_datasets_ordered(
        self,
        *,
        use_selection_order: bool = False,
        empty_message: str = "Select two or more datasets to combine.",
    ) -> list[tuple[str, RcsGrid]] | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage(empty_message)
            return None

        selected_rows = [idx.row() for idx in selected]
        if use_selection_order:
            ordered_rows = [
                row for row in getattr(self, "_dataset_selection_order", []) if row in selected_rows
            ]
            for row in selected_rows:
                if row not in ordered_rows:
                    ordered_rows.append(row)
            selected_rows = ordered_rows
        else:
            selected_rows = sorted(selected_rows)

        datasets: list[tuple[str, RcsGrid]] = []
        for row in selected_rows:
            item = self.table.item(row, 0)
            if item is None:
                return None
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return None
            datasets.append((item.text(), dataset))
        return datasets

    def _combine_datasets_add(
        self,
        op_label: str,
        op_symbol: str,
        func_add: str,
        func_add_many: str,
    ) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        base = datasets[0][1]
        try:
            if len(datasets) == 2:
                result = getattr(base, func_add)(datasets[1][1])
            else:
                others = [ds for _, ds in datasets[1:]]
                result = getattr(base, func_add_many)(*others)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _combine_datasets_sub(self, op_label: str, op_symbol: str, func_sub: str) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        result = datasets[0][1]
        try:
            for _, ds in datasets[1:]:
                result = getattr(result, func_sub)(ds)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _coherent_add_selected(self) -> None:
        self._combine_datasets_add("Coherent +", "+", "coherent_add", "coherent_add_many")

    def _coherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Coherent -", "-", "coherent_subtract")

    def _incoherent_add_selected(self) -> None:
        self._combine_datasets_add("Incoherent +", "+", "incoherent_add", "incoherent_add_many")

    def _incoherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Incoherent -", "-", "incoherent_subtract")

    def _join_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to join.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to join.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        worker = _JoinDatasetsWorker(grids, tol=1e-6)
        worker.progress.connect(self._on_join_worker_progress)
        worker.finished.connect(self._on_join_worker_finished)
        if not self._try_start_background_job("Dataset join", worker):
            return
        self._pending_join_names = names
        self.status.showMessage(f"Joining datasets... 0/{len(grids)}")

    def _overlap_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets for overlap.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets for overlap.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        try:
            overlap_grids = RcsGrid.overlap_many(*grids, tol=1e-6)
            produced = 0
            for (name, _), overlap_grid in zip(datasets, overlap_grids):
                history = f"Overlap with [{', '.join(names)}]: {name}"
                self._add_dataset_row(overlap_grid, f"{name} [Overlap]", history, file_name="")
                produced += 1
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        if produced == 0:
            self.status.showMessage("No overlap outputs were created.")
            return
        self.status.showMessage(f"Overlap created {produced} dataset(s).")

    def _prompt_choice(self, title: str, label: str, choices: list[str], default_idx: int = 0) -> str | None:
        value, ok = QInputDialog.getItem(self, title, label, choices, default_idx, False)
        if not ok:
            return None
        return str(value)

    def _axis_crop_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to crop.",
        )
        if datasets is None:
            return

        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        dlg = AxisCropDialog(
            ref,
            n_datasets=len(datasets),
            presel_az=self._selected_values(self.list_az) or None,
            presel_el=self._selected_values(self.list_elev) or None,
            presel_freq=self._selected_values(self.list_freq) or None,
            presel_pol=self._selected_values(self.list_pol) or None,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        crop_params = dlg.get_crop_params()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                cropped = dataset.axis_crop(**crop_params)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            self._add_dataset_row(cropped, f"{name} [Crop]", f"Axis Crop: {name}", file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Axis Crop created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Axis Crop created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Axis Crop created {produced} dataset(s).")

    def _slice_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to slice.",
        )
        if datasets is None:
            return

        sel_az = self._selected_values(self.list_az)
        sel_el = self._selected_values(self.list_elev)
        sel_freq = self._selected_values(self.list_freq)
        sel_pol = self._selected_values(self.list_pol)

        if not (sel_az or sel_el or sel_freq or sel_pol):
            self.status.showMessage(
                "Select parameter values (azimuth/elevation/frequency/polarization) to slice."
            )
            return

        crop_params = {
            "azimuths": sel_az or None,
            "elevations": sel_el or None,
            "frequencies": sel_freq or None,
            "polarizations": sel_pol or None,
        }

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                sliced = dataset.axis_crop(**crop_params)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = (
                "Slice (selected params): "
                f"{name} | az={len(sliced.azimuths)}, el={len(sliced.elevations)}, "
                f"freq={len(sliced.frequencies)}, pol={len(sliced.polarizations)}"
            )
            self._add_dataset_row(sliced, f"{name} [Slice]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Slice created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Slice created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Slice created {produced} dataset(s).")

    def _medianize_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to medianize.",
        )
        if datasets is None:
            return

        window_deg, ok = QInputDialog.getDouble(
            self,
            "Medianize",
            "Window (degrees):",
            1.0,
            1e-6,
            1e9,
            6,
        )
        if not ok:
            return

        slide_deg, ok = QInputDialog.getDouble(
            self,
            "Medianize",
            "Slide (degrees):",
            1.0,
            1e-6,
            1e9,
            6,
        )
        if not ok:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = dataset.medianize_azimuth(window_deg, slide_deg)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = (
                f"Medianize azimuth (window={window_deg:.6g} deg, "
                f"slide={slide_deg:.6g} deg): {name}"
            )
            out_name = f"{name} [Median {window_deg:.6g}/{slide_deg:.6g}deg]"
            self._add_dataset_row(result, out_name, history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Medianize created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Medianize created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Medianize created {produced} dataset(s).")

    def _statistics_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets for statistics.",
        )
        if datasets is None:
            return

        dlg = StatisticsDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        statistic, percentile, domain, axes = dlg.get_params()
        if not axes:
            self.status.showMessage("Select at least one axis for statistics reduction.")
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                stat_grid = dataset.statistics_dataset(
                    statistic=statistic,
                    axes=axes,
                    domain=domain,
                    percentile=percentile,
                    broadcast_reduced=True,
                )
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue

            if statistic == "percentile":
                stat_label = f"p{percentile:g}"
            else:
                stat_label = statistic
            history = f"Statistics ({stat_label}, {domain}, axes={axes}): {name}"
            self._add_dataset_row(stat_grid, f"{name} [{stat_label}]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Statistics created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Statistics created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Statistics created {produced} dataset(s).")

    def _difference_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets for difference.",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Select exactly 2 datasets for difference.")
            return

        mode = self._prompt_choice(
            "Difference",
            "Mode:",
            ["coherent", "incoherent", "db"],
            default_idx=0,
        )
        if mode is None:
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        try:
            result = grids[0].difference(grids[1], mode=mode)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f"Diff[{mode}] " + " - ".join(names)
        history = f"Difference ({mode}): {' - '.join(names)}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"Difference created: {new_name}")

    def _delete_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to delete.")
            return
        rows = sorted((idx.row() for idx in selected), reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self.active_dataset = None
        self._clear_param_lists()
        self.status.showMessage(f"Deleted {len(rows)} dataset(s).")

    def _save_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to save.")
            return

        rows = sorted(idx.row() for idx in selected)

        if len(rows) == 1:
            # Single dataset — let the user pick the exact file path.
            row = rows[0]
            item = self.table.item(row, 0)
            if item is None:
                return
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return
            name = item.text().strip() or "dataset"
            file_item = self.table.item(row, 1)
            prev_file = file_item.text() if file_item else ""
            prev_stem = os.path.splitext(prev_file)[0] if prev_file else ""
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Dataset", f"{name}.grim", "GRIM Files (*.grim)"
            )
            if not path:
                return
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
            new_stem = os.path.splitext(file_name)[0]
            if prev_stem and item.text().strip() == prev_stem:
                item.setText(new_stem)
            elif not item.text().strip():
                item.setText(new_stem)
            self.status.showMessage("Save completed.")
        else:
            # Multiple datasets — pick a folder once, save each using its table name.
            directory = QFileDialog.getExistingDirectory(self, "Save Selected Datasets")
            if not directory:
                return
            saved = 0
            for row in rows:
                item = self.table.item(row, 0)
                if item is None:
                    continue
                dataset = item.data(Qt.UserRole)
                if not isinstance(dataset, RcsGrid):
                    continue
                name = item.text().strip() or f"dataset_{row + 1}"
                path = os.path.join(directory, f"{name}.grim")
                saved_path = dataset.save(path)
                file_name = os.path.basename(saved_path)
                file_item = self.table.item(row, 1)
                if file_item is None:
                    file_item = QTableWidgetItem(file_name)
                    file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 1, file_item)
                else:
                    file_item.setText(file_name)
                history_item = self.table.item(row, 2)
                if history_item is None:
                    history_item = QTableWidgetItem(saved_path)
                    self.table.setItem(row, 2, history_item)
                else:
                    history_item.setText(saved_path)
                saved += 1
            self.status.showMessage(f"Saved {saved} dataset(s) to {directory}.")

    def _save_all_datasets(self) -> None:
        if self.table.rowCount() == 0:
            self.status.showMessage("No datasets to save.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Save All Datasets")
        if not directory:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                continue
            name = item.text().strip() or f"dataset_{row + 1}"
            filename = f"{name}.grim"
            path = os.path.join(directory, filename)
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            file_item = self.table.item(row, 1)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
        self.status.showMessage("Save all completed.")

    def _export_plot(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        root, ext = os.path.splitext(path)
        if not ext:
            if "PDF" in selected_filter:
                path = f"{path}.pdf"
            else:
                path = f"{path}.png"
        self.plot_figure.savefig(path, dpi=200, bbox_inches="tight")
        self.status.showMessage(f"Plot exported: {os.path.basename(path)}")

    def _on_plot_context_menu(self, pos) -> None:
        menu = QMenu(self)
        action_copy = menu.addAction("Copy Plot")
        pbp_menu = menu.addMenu("PBP Fill Mode")
        action_pbp_gray = pbp_menu.addAction("Gray")
        action_pbp_gray.setCheckable(True)
        action_pbp_gray.setChecked(self.pbp_fill_mode == "gray")
        action_pbp_rcs = pbp_menu.addAction("Heatmap (RCS Value)")
        action_pbp_rcs.setCheckable(True)
        action_pbp_rcs.setChecked(self.pbp_fill_mode == "heatmap_rcs")
        action_pbp_density = pbp_menu.addAction("Heatmap (Overlap Density)")
        action_pbp_density.setCheckable(True)
        action_pbp_density.setChecked(self.pbp_fill_mode == "heatmap_density")
        action = menu.exec(self.plot_canvas.mapToGlobal(pos))
        if action == action_copy:
            pixmap = self.plot_canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.status.showMessage("Plot copied to clipboard.")
        elif action in (action_pbp_gray, action_pbp_rcs, action_pbp_density):
            if action == action_pbp_gray:
                self.pbp_fill_mode = "gray"
            elif action == action_pbp_rcs:
                self.pbp_fill_mode = "heatmap_rcs"
            else:
                self.pbp_fill_mode = "heatmap_density"
            if self.last_plot_mode == "azimuth_rect":
                self._plot_azimuth_rect()
            elif self.last_plot_mode == "azimuth_polar":
                self._plot_azimuth_polar()
            elif self.last_plot_mode == "frequency":
                self._plot_frequency()
            elif self.last_plot_mode == "isar_image":
                self._plot_isar_image()

    def _on_dataset_header_double_clicked(self, section: int) -> None:
        if section != 0:
            return
        self.table.selectAll()

    def _on_dataset_context_menu(self, pos) -> None:
        if not self.table.selectionModel().selectedRows():
            index = self.table.indexAt(pos)
            if index.isValid():
                self.table.selectRow(index.row())
            else:
                return
        menu = QMenu(self)
        action_save = menu.addAction("Save")
        action_delete = menu.addAction("Delete")
        menu.addSeparator()
        action_color = menu.addAction("Text Color…")
        action_reset_color = menu.addAction("Reset Text Color")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == action_save:
            self._save_selected_datasets()
        elif action == action_delete:
            self._delete_selected_datasets()
        elif action == action_color:
            self._set_dataset_text_color()
        elif action == action_reset_color:
            self._reset_dataset_text_color()

    def _set_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        initial = self.table.item(rows[0], 0)
        initial_color = initial.foreground().color() if initial else QColor()
        color = QColorDialog.getColor(initial_color, self, "Choose Text Color")
        if not color.isValid():
            return
        brush = QBrush(color)
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(brush)

    def _reset_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(QBrush())

    def _align_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to align (first = reference).",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to align (first = reference).")
            return

        ref_name, ref_grid = datasets[0]
        others = datasets[1:]
        dlg = AlignDialog(ref_name, len(others), parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        mode = dlg.get_mode()
        produced = 0
        skipped: list[str] = []
        for name, dataset in others:
            try:
                aligned = dataset.align_to(ref_grid, mode=mode)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Align ({mode}) to {ref_name}: {name}"
            self._add_dataset_row(aligned, f"{name} [Aligned]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Align created 0 datasets.")
            return
        msg = f"Align created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _mirror_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to mirror.",
        )
        if datasets is None:
            return

        default_about = 0.0
        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        if isinstance(ref, RcsGrid) and len(ref.azimuths) > 0:
            az_vals = np.asarray(ref.azimuths, dtype=float)
            finite = az_vals[np.isfinite(az_vals)]
            if finite.size > 0:
                default_about = float(np.mean(finite))

        about, ok = QInputDialog.getDouble(
            self,
            "Mirror Dataset",
            "Mirror about azimuth (degrees):",
            default_about,
            -1e9,
            1e9,
            6,
        )
        if not ok:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                mirrored = dataset.mirror_about_azimuth(about)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Mirror about az={about:.6g} deg: {name}"
            self._add_dataset_row(
                mirrored,
                f"{name} [Mirror {about:.6g}°]",
                history,
                file_name="",
            )
            produced += 1

        if produced == 0:
            self.status.showMessage("Mirror created 0 datasets.")
            return
        msg = f"Mirror created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _azimuth_shift_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to azimuth-shift.",
        )
        if datasets is None:
            return

        delta, ok = QInputDialog.getDouble(
            self,
            "Azimuth Shift",
            "Azimuth offset (degrees):",
            0.0,
            -1e9,
            1e9,
            6,
        )
        if not ok:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                shifted = dataset.shift_azimuth(delta)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Azimuth shift ({delta:+.6g} deg): {name}"
            self._add_dataset_row(
                shifted,
                f"{name} [AzShift {delta:+.6g}°]",
                history,
                file_name="",
            )
            produced += 1

        if produced == 0:
            self.status.showMessage("Azimuth Shift created 0 datasets.")
            return
        msg = f"Azimuth Shift created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _elevation_to_azimuth_360_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert elevation pair into 360 azimuth.",
        )
        if datasets is None:
            return

        selected_el_values = self._selected_values(self.list_elev)
        selected_pair: tuple[float, float] | None = None
        if len(selected_el_values) == 2:
            try:
                pair = tuple(sorted(float(v) for v in selected_el_values))
                selected_pair = (pair[0], pair[1])
            except (TypeError, ValueError):
                selected_pair = None

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if selected_pair is None:
                    result = dataset.combine_elevation_pair_to_azimuth_360(azimuth_shift_deg=180.0)
                    pair_text = "min/max elevation"
                else:
                    result = dataset.combine_elevation_pair_to_azimuth_360(
                        selected_pair[0],
                        selected_pair[1],
                        azimuth_shift_deg=180.0,
                    )
                    pair_text = f"{selected_pair[0]:.6g}/{selected_pair[1]:.6g} deg"
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            history = f"El->Az360 (shift +180 deg, pair={pair_text}): {name}"
            self._add_dataset_row(
                result,
                f"{name} [El->Az360]",
                history,
                file_name="",
            )
            produced += 1

        if produced == 0:
            self.status.showMessage("El->Az360 created 0 datasets.")
            return
        msg = f"El->Az360 created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _scale_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to scale.",
        )
        if datasets is None:
            return

        dlg = ScaleDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        factor = dlg.get_factor()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                power_scale = float(factor)
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale = np.sqrt(max(power_scale, 0.0))
                    result_rcs = dataset.rcs * amp_scale
                else:
                    result_rcs = dataset.rcs * power_scale
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power * power_scale,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Scale (×{factor:.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Scaled]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Scale created 0 datasets.")
            return
        msg = f"Scale created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _offset_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to offset.",
        )
        if datasets is None:
            return

        value, ok = QInputDialog.getDouble(
            self, "Offset", "Offset (dB) — shifts all displayed values by this amount:",
            0.0, -300.0, 300.0, 4,
        )
        if not ok:
            return

        linear_scale = 10.0 ** (value / 10.0)
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if dataset.rcs_domain == "complex_amplitude":
                    result_rcs = dataset.rcs * np.sqrt(linear_scale)
                else:
                    result_rcs = dataset.rcs * linear_scale
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power * linear_scale,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Offset ({value:+.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Offset {value:+.6g}]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Offset created 0 datasets.")
            return
        msg = f"Offset created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _normalize_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to normalize.",
        )
        if datasets is None:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                peak = float(np.nanmax(dataset.rcs_power))
                if peak == 0.0:
                    skipped.append(f"{name} (all-zero RCS)")
                    continue
                if dataset.rcs_domain == "complex_amplitude":
                    result_rcs = dataset.rcs / np.sqrt(peak)
                else:
                    result_rcs = dataset.rcs / peak
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power / peak,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Normalize (peak={peak:.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Norm]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Normalize created 0 datasets.")
            return
        msg = f"Normalize created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _phase_shift_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to phase-shift.",
        )
        if datasets is None:
            return

        phase_deg, ok = QInputDialog.getDouble(
            self, "Phase Shift", "Phase shift (degrees):", 0.0, -360.0, 360.0, 4
        )
        if not ok:
            return

        phasor = np.exp(1j * np.deg2rad(phase_deg))
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _dataset_with_rcs(
                    dataset,
                    dataset.rcs * phasor,
                    rcs_power=dataset.rcs_power,
                    rcs_domain="complex_amplitude",
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Phase Shift ({phase_deg:+.4g} deg): {name}"
            self._add_dataset_row(result, f"{name} [Phase {phase_deg:+.4g}°]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Phase Shift created 0 datasets.")
            return
        msg = f"Phase Shift created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _resample_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to resample.",
        )
        if datasets is None:
            return

        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        dlg = ResampleDialog(ref, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        n_az, n_el, n_freq = dlg.get_target_counts()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _resample_grid(dataset, n_az, n_el, n_freq)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Resample ({n_az}×{n_el}×{n_freq}): {name}"
            self._add_dataset_row(result, f"{name} [Resampled]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Resample created 0 datasets.")
            return
        msg = f"Resample created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _duplicate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to duplicate.",
        )
        if datasets is None:
            return

        for name, dataset in datasets:
            dup = RcsGrid(
                dataset.azimuths.copy(),
                dataset.elevations.copy(),
                dataset.frequencies.copy(),
                list(dataset.polarizations),
                dataset.rcs.copy(),
                rcs_power=dataset.rcs_power.copy(),
                rcs_domain=dataset.rcs_domain,
            )
            self._add_dataset_row(dup, f"{name} [Copy]", f"Duplicate of: {name}", file_name="")
        self.status.showMessage(f"Duplicated {len(datasets)} dataset(s).")

    def _export_csv_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to export.",
        )
        if datasets is None:
            return

        dlg = ExportCsvDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        scale, include_phase = dlg.get_options()
        produced = 0
        for name, dataset in datasets:
            safe_name = name.replace("/", "_").replace("\\", "_")
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {name}",
                f"{safe_name}.csv",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not path:
                continue
            if not path.lower().endswith(".csv"):
                path = f"{path}.csv"
            _write_dataset_csv(dataset, path, scale=scale, sep=",", include_phase=include_phase)
            produced += 1

        if produced:
            self.status.showMessage(f"Exported {produced} dataset(s) to CSV.")
        else:
            self.status.showMessage("Export cancelled.")

    def _reselect_indices(self, widget: QListWidget, indices: set[int]) -> None:
        if not indices:
            return
        widget.blockSignals(True)
        for row in range(widget.count()):
            item = widget.item(row)
            idx = item.data(Qt.UserRole + 1)
            if idx in indices:
                item.setSelected(True)
        widget.blockSignals(False)

    # ── RCS-specific processing ───────────────────────────────────────────────

    def _coherent_div_selected(self) -> None:
        """Divide numerator dataset by denominator (complex, element-wise)."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select exactly 2 datasets (numerator first, then denominator).",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Coherent ÷: select exactly 2 datasets.")
            return
        name_a, ds_a = datasets[0]
        name_b, ds_b = datasets[1]

        if ds_a.rcs.shape != ds_b.rcs.shape:
            self.status.showMessage(
                f"Coherent ÷: shape mismatch {ds_a.rcs.shape} vs {ds_b.rcs.shape}."
            )
            return

        denom = ds_b.rcs.copy()
        denom[denom == 0] = 1e-30 + 0j
        result_rcs = ds_a.rcs / denom
        result = RcsGrid(
            ds_a.azimuths, ds_a.elevations, ds_a.frequencies,
            ds_a.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ds_a.units,
        )
        out_name = f"{name_a} ÷ {name_b}"
        self._add_dataset_row(result, out_name, f"Coherent ÷: {name_a} / {name_b}", file_name="")
        self.status.showMessage(f"Coherent ÷ produced: {out_name}")

    def _time_gate_selected(self) -> None:
        """Apply time-domain gating via IFFT → window → FFT."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to time gate.",
        )
        if datasets is None:
            return

        # Use first selected dataset to configure the dialog
        _, ds_first = datasets[0]
        dlg = TimeGateDialog(ds_first, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        gate_start, gate_end, window_type = dlg.get_params()

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _apply_time_gate(dataset, gate_start, gate_end, window_type)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            out_name = f"{name} [Gate {gate_start:.1f}-{gate_end:.1f}m {window_type}]"
            self._add_dataset_row(result, out_name,
                f"Time Gate: {name}  {gate_start:.3f}–{gate_end:.3f}m  {window_type}", file_name="")
            produced += 1

        msg = f"Time Gate produced {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)
