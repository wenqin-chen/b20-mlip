"""Shared fixtures (CONTRACTS.md section 7): offline session, tiny B20 frames, isolated settings.

``tiny_b20.extxyz`` is generated in-test: 12 synthetic 8-atom P2_1 3 (B20, space group 198)
frames — 4 compounds x 3 config types with random energies/forces/stress and group ids,
``label_source="none"`` — plus 3 frames tagged ``energy_scale="qe"`` with
``config_energy_weight=0`` to exercise the weight round trip (15 frames in total).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup import crystal

from b20mlip.config import Settings
from b20mlip.io import frame_from_atoms, write_frames
from b20mlip.models import Frame

# compound -> (a [A], u_TM, u_X, TM, X, magnetic); approximate experimental B20 parameters
B20_PARAMS: dict[str, tuple[float, float, float, str, str, bool]] = {
    "FeSi": (4.48, 0.137, 0.842, "Fe", "Si", False),
    "CoSi": (4.44, 0.144, 0.843, "Co", "Si", False),
    "MnSi": (4.56, 0.138, 0.846, "Mn", "Si", True),
    "FeGe": (4.70, 0.135, 0.842, "Fe", "Ge", True),
}
CONFIG_TYPES = ("relax", "strain", "rattle")
QE_TAGGED = ("FeSi", "CoSi", "MnSi")


def _blocked(*args: object, **kwargs: object) -> None:
    raise RuntimeError("network access is disabled for the whole test session")


@pytest.fixture(scope="session", autouse=True)
def block_network() -> Iterator[None]:
    """No test may open a socket (CONTRACTS.md section 7)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(socket, "create_connection", _blocked)
    mp.setattr(socket.socket, "connect", _blocked)
    mp.setattr(socket.socket, "connect_ex", _blocked)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def clean_b20_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any B20_* variables from the developer's shell so config tests are deterministic,
    and pin the CLI's help rendering: on GitHub Actions Rich emits ANSI colour codes and wraps
    help text at a narrow width, which broke the ``"--option" in result.output`` assertions
    (CI run 35630049503). NO_COLOR + a wide COLUMNS make CliRunner output plain everywhere."""
    for key in list(os.environ):
        if key.upper().startswith("B20_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    # the executors let an inherited OMP_NUM_THREADS win over compute.threads (setdefault);
    # CI exports OMP_NUM_THREADS=2, so tests must start from a clean thread environment
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)


def b20_atoms(compound: str, config_type: str, rng: np.random.Generator) -> Atoms:
    a, u, v, tm, x, _ = B20_PARAMS[compound]
    atoms = crystal(
        [tm, x],
        basis=[(u, u, u), (v, v, v)],
        spacegroup=198,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )
    atoms.info.clear()  # drop the Spacegroup object / unit_cell tag that crystal() attaches
    if config_type == "strain":
        atoms.set_cell(atoms.cell[:] * 1.03, scale_atoms=True)
    elif config_type == "rattle":
        atoms.positions = atoms.positions + rng.normal(0.0, 0.05, atoms.positions.shape)
    return atoms


def _label(atoms: Atoms, rng: np.random.Generator, magnetic: bool) -> None:
    n = len(atoms)
    results = {
        "energy": float(-6.0 * n + rng.normal(0.0, 0.1)),
        "forces": rng.normal(0.0, 0.5, (n, 3)),
        "stress": rng.normal(0.0, 0.01, 6),  # Voigt 6, eV/A^3
    }
    if magnetic:
        results["magmoms"] = rng.normal(1.0, 0.1, n)
        atoms.info["total_magnetization"] = float(np.sum(results["magmoms"]))
    atoms.calc = SinglePointCalculator(atoms, **results)


def make_tiny_frames(seed: int = 0) -> list[Frame]:
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    for compound, (_, _, _, _, _, magnetic) in B20_PARAMS.items():
        parent = f"{compound}-p0"
        for config_type in CONFIG_TYPES:
            atoms = b20_atoms(compound, config_type, rng)
            if config_type == "rattle":
                atoms.info["temperature_K"] = 300.0
            atoms.info["lineage"] = parent
            _label(atoms, rng, magnetic)
            frames.append(
                frame_from_atoms(
                    atoms,
                    group_id=f"{compound}/{config_type}/{parent}",
                    compound=compound,
                    config_type=config_type,  # type: ignore[arg-type]
                    parent_id=parent,
                )
            )
    for compound in QE_TAGGED:
        atoms = b20_atoms(compound, "rattle", rng)  # distinct geometry -> distinct frame_id
        atoms.info["qe_unit"] = f"u-{compound}"
        _label(atoms, rng, B20_PARAMS[compound][5])
        frame = frame_from_atoms(
            atoms,
            group_id=f"{compound}/noise_floor/{compound}-p0",
            compound=compound,
            config_type="noise_floor",
            parent_id=f"{compound}-p0",
            label_source="qe",
            energy_scale="qe",
        )
        frames.append(frame.model_copy(update={"weights": {"energy": 0.0}}))
    return frames


@pytest.fixture(scope="session")
def tiny_frames() -> list[Frame]:
    return make_tiny_frames()


@pytest.fixture(scope="session")
def tiny_b20_path(tmp_path_factory: pytest.TempPathFactory, tiny_frames: list[Frame]) -> Path:
    path = tmp_path_factory.mktemp("fixtures") / "tiny_b20.extxyz"
    write_frames(tiny_frames, path)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with every writable path inside ``tmp_path`` (never the repo's runs/)."""
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "runs_dir": str(tmp_path / "runs"),
                "models_dir": str(tmp_path / "models"),
                "dft_dir": str(tmp_path / "dft"),
                "reports_dir": str(tmp_path / "reports"),
            }
        }
    )


@pytest.fixture
def repo() -> Path:
    return Path(__file__).resolve().parents[1]


# --- tiny MACE model (CONTRACTS.md section 7), shared by the train / phonons / evaluate / md tiers

TINY_MACE_NAME = "tiny_b20"
TINY_MACE_HIDDEN_IRREPS = "8x0e"
TINY_MACE_R_MAX = 4.0
TINY_MACE_EXTRA_ARGS: tuple[str, ...] = (
    "--num_interactions", "1", "--max_ell", "1", "--correlation", "2", "--num_radial_basis", "4",
)  # fmt: skip


@dataclass(frozen=True)
class TinyMace:
    """A MACE model trained in-test on the tiny B20 fixture (1 epoch, 8x0e, r_max 4 A).

    ``model_path`` is the ``.model`` MACE saved, ``lammps_path`` its ``mace_create_lammps_model``
    export (``<model>-lammps.pt``). The elements are Si, Mn, Fe, Co, Ge (every fixture compound),
    so any B20 fixture cell can be evaluated with ``calculator()``. Random labels: the model is
    numerically valid but physically meaningless.
    """

    model_path: Path
    lammps_path: Path
    sha256: str
    lammps_sha256: str
    work_dir: Path
    argv: tuple[str, ...]
    seconds: float
    elements: tuple[int, ...] = (14, 25, 26, 27, 32)

    def calculator(self) -> Any:
        """A fresh ``mace.calculators.MACECalculator`` for this model (CPU, float64)."""
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=str(self.model_path), device="cpu", default_dtype="float64"
        )


def tiny_mace_settings(root: Path) -> Settings:
    """Settings that make ``train --variant scratch`` produce the tiny CI architecture."""
    return Settings.model_validate(
        {
            "paths": {
                "data_dir": str(root / "data"),
                "runs_dir": str(root / "runs"),
                "models_dir": str(root / "models"),
                "dft_dir": str(root / "dft"),
                "reports_dir": str(root / "reports"),
            },
            "train": {
                "epochs": 1,
                "batch_size": 4,
                "scratch": {"hidden_irreps": TINY_MACE_HIDDEN_IRREPS, "r_max": TINY_MACE_R_MAX},
                "extra_args": list(TINY_MACE_EXTRA_ARGS),
            },
        }
    )


@pytest.fixture(scope="session")
def tiny_mace(tmp_path_factory: pytest.TempPathFactory, tiny_frames: list[Frame]) -> TinyMace:
    """Train the tiny model once per session with ``mace_run_train`` (< 30 s) and export it."""
    from b20mlip.provenance import sha256_file
    from b20mlip.train import export, finetune

    root = tmp_path_factory.mktemp("tiny_mace")
    cfg = tiny_mace_settings(root)
    valid = [f for f in tiny_frames if f.config_type == "strain"]  # one per compound, labelled
    train = [f for f in tiny_frames if f.config_type != "strain"]  # incl. the qe weight-0 frames
    files = finetune.write_split_files(
        {"train": train, "valid": valid, "test": valid}, root / "data"
    )
    argv = finetune.build_args(
        cfg, "scratch", files, 0, root,
        energy_scale=finetune.frames_energy_scale(tiny_frames), name=TINY_MACE_NAME,
    )  # fmt: skip
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    log = root / "mace_stdout.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            finetune.train_command(argv), cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        pytest.fail(f"tiny mace_run_train failed ({proc.returncode}):\n{log.read_text()[-3000:]}")
    model = finetune.locate_model(root, TINY_MACE_NAME)
    lammps = Path(export.to_lammps(model, "Default"))
    seconds = time.perf_counter() - t0
    return TinyMace(
        model_path=model,
        lammps_path=lammps,
        sha256=sha256_file(model),
        lammps_sha256=sha256_file(lammps),
        work_dir=root,
        argv=tuple(argv),
        seconds=seconds,
    )
