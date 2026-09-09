"""Shared model parameters and reference geometry (N, mm, s)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import cached_property
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class FEMConfig:
    """All dimensional quantities use mm, s, N, and N/mm^2."""

    # Geometry and laboratory setup
    length_mm: float = 30.0
    height_mm: float = 5.0
    width_mm: float = 20.0
    inclination_deg: float = 60.0
    drop_s_mm: float = 7.0

    # Blade A geometry from the supplied blade drawing
    blade_thickness_mm: float = 0.245
    blade_tip_angle_deg: float = 23.0

    # Prescribed motion
    speed_mm_s: float = 0.2
    max_indentation_mm: float = 3.0
    hold_s: float = 0.1
    pullback_speed_mm_s: float = 0.2
    approach_steps: int = 61
    hold_steps: int = 11
    pullback_steps: int = 61

    # Mesh and nonlinear solve
    mesh_size_mm: float = 0.45
    contact_surface_penalty_n_mm3: float = 250.0
    contact_smoothing_mm: float = 0.002
    friction_coefficient: float = 0.35
    friction_velocity_regularization_mm_s: float = 0.002
    adhesive_shear_strength_n_mm2: float = 0.0
    max_newton_iterations: int = 35
    max_newton_residual_evaluations: int = 120
    # The residual tolerance is an absolute force-balance tolerance.  The
    # default is 0.001 N. Optimizer termination alone is never equilibrium.
    newton_tolerance_n: float = 1.0e-3
    max_line_search_iterations: int = 14
    max_damage_iterations: int = 8
    max_contact_iterations: int = 60
    contact_multiplier_tolerance_n_mm2: float = 1.0e-3
    contact_penetration_tolerance_mm: float = 1.0e-4
    contact_multiplier_relaxation: float = 0.5
    damage_tolerance: float = 1.0e-6
    min_adaptive_increment_mm: float = 0.005
    max_adaptive_substeps: int = 256
    progress: bool = False

    # CB245NE starting material data.  G and tensile strength are published
    # catalogue values; Gc is a neoprene cutting prior, not a CB245NE test.
    shear_modulus_n_mm2: float = 0.66
    poisson_ratio: float = 0.49
    tensile_strength_n_mm2: float = 4.4
    fracture_energy_n_mm: float = 2.15
    cohesive_opening_mm: float = 0.18
    cohesive_mode_mix: float = 1.0
    cohesive_viscosity_s: float = 0.05
    cohesive_compression_penalty_n_mm3: float = 100.0

    @classmethod
    def from_json(cls, path: str | Path) -> "FEMConfig":
        # ``utf-8-sig`` accepts ordinary UTF-8 and the BOM emitted by
        # Windows PowerShell's UTF-8 writer, keeping JSON runs reproducible
        # across the two common Windows toolchains.
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            data = json.load(stream)
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def bulk_modulus_n_mm2(self) -> float:
        return 2.0 * self.shear_modulus_n_mm2 * (1.0 + self.poisson_ratio) / (
            3.0 * (1.0 - 2.0 * self.poisson_ratio)
        )

    @property
    def cohesive_strength_n_mm2(self) -> float:
        # Smith-Ferrante relation used in the recent cohesive cutting model:
        # Gc = e * sigma_c * delta_c.
        return self.fracture_energy_n_mm / (
            math.e * self.cohesive_opening_mm
        )

    def validate(self) -> None:
        positive = {
            "length_mm": self.length_mm,
            "height_mm": self.height_mm,
            "width_mm": self.width_mm,
            "mesh_size_mm": self.mesh_size_mm,
            "blade_thickness_mm": self.blade_thickness_mm,
            "speed_mm_s": self.speed_mm_s,
            "max_indentation_mm": self.max_indentation_mm,
            "pullback_speed_mm_s": self.pullback_speed_mm_s,
            "contact_surface_penalty_n_mm3": self.contact_surface_penalty_n_mm3,
            "contact_smoothing_mm": self.contact_smoothing_mm,
            "friction_velocity_regularization_mm_s": self.friction_velocity_regularization_mm_s,
            "shear_modulus_n_mm2": self.shear_modulus_n_mm2,
            "tensile_strength_n_mm2": self.tensile_strength_n_mm2,
            "fracture_energy_n_mm": self.fracture_energy_n_mm,
            "cohesive_opening_mm": self.cohesive_opening_mm,
            "cohesive_compression_penalty_n_mm3": self.cohesive_compression_penalty_n_mm3,
            "min_adaptive_increment_mm": self.min_adaptive_increment_mm,
            "contact_multiplier_tolerance_n_mm2": self.contact_multiplier_tolerance_n_mm2,
            "contact_penetration_tolerance_mm": self.contact_penetration_tolerance_mm,
            "newton_tolerance_n": self.newton_tolerance_n,
            "damage_tolerance": self.damage_tolerance,
            "cohesive_mode_mix": self.cohesive_mode_mix,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("hold_s", "cohesive_viscosity_s", "adhesive_shear_strength_n_mm2", "friction_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.blade_tip_angle_deg < 180.0:
            raise ValueError("blade_tip_angle_deg must lie in (0, 180)")
        if not 0.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be between 0 and 0.5")
        if not 0.0 <= self.inclination_deg < 90.0:
            raise ValueError("inclination_deg must be in [0, 90) degrees")
        if not 0.0 <= self.friction_coefficient:
            raise ValueError("friction_coefficient must be non-negative")
        if not 0.0 < self.contact_multiplier_relaxation <= 1.0:
            raise ValueError("contact_multiplier_relaxation must lie in (0, 1]")
        if not 0.0 < self.drop_s_mm < self.length_mm:
            raise ValueError("drop_s_mm must lie inside the specimen length")
        if self.max_indentation_mm > self.height_mm / math.cos(
            math.radians(self.inclination_deg)
        ):
            raise ValueError("max_indentation_mm is outside the configured specimen envelope")
        for name in (
            "approach_steps",
            "hold_steps",
            "pullback_steps",
            "max_newton_iterations",
            "max_newton_residual_evaluations",
            "max_line_search_iterations",
            "max_damage_iterations",
            "max_contact_iterations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if type(self.max_adaptive_substeps) is not int or self.max_adaptive_substeps < 1:
            raise ValueError("max_adaptive_substeps must be positive")


@dataclass
class FEMMesh:
    reference_xy: np.ndarray
    local_sn: np.ndarray
    triangles: np.ndarray
    seam_pairs: np.ndarray
    seam_segment_lengths_mm: np.ndarray
    bottom_nodes: np.ndarray
    seam_nodes: np.ndarray
    side_of_node: np.ndarray

    @cached_property
    def bulk_data(self):
        x = self.reference_xy[self.triangles]
        dm = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=2)
        areas = np.linalg.det(dm) / 2.0
        gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]]) @ np.linalg.inv(dm)
        dofs = (2 * self.triangles[:, :, None] + np.arange(2)).reshape(-1, 6)
        return areas, gradients, dofs

    @property
    def node_count(self) -> int:
        return int(self.reference_xy.shape[0])

    @property
    def dof_count(self) -> int:
        return 2 * self.node_count


def _rotation_vectors(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the inclined longitudinal and outward-normal lab vectors."""

    theta = math.radians(angle_deg)
    e_s = np.array([math.cos(theta), -math.sin(theta)], dtype=float)
    e_n = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    return e_s, e_n


def local_to_lab(sn: np.ndarray, config: FEMConfig) -> np.ndarray:
    e_s, e_n = _rotation_vectors(config.inclination_deg)
    return np.asarray(sn)[:, [0]] * e_s + np.asarray(sn)[:, [1]] * e_n
