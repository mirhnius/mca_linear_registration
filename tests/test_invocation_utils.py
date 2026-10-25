import gzip
import json
from pathlib import Path

import pytest

from lnrgst_mca import invocation_utils as invo

TEMPLATE = "tpl"
SCAN = "sub-{id}_ses-BL_acq-sag3D_run-01_T1w.nii.gz"


def collect(root: Path):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.fixture
def bids_dir(tmp_path):
    root = tmp_path / "bids"
    for sub_id in ("001", "002"):
        anat = root / f"sub-{sub_id}" / "ses-BL" / "anat"
        anat.mkdir(parents=True)
        (anat / SCAN.format(id=sub_id)).write_bytes(b"")
    return root


@pytest.fixture
def subjects_map(bids_dir):
    return invo.create_subject_map(bids_dir)


class TestSubjectMap:
    def test_scan_fields(self):
        fields = invo.scan_filed_dict(Path("/data") / SCAN.format(id="001"))
        assert fields == {
            "input_path": str(Path("/data") / SCAN.format(id="001")),
            "subject": "sub-001",
            "session": "ses-BL",
            "run": "run-01",
            "modality": "T1w.nii.gz",
        }

    def test_create_subject_map(self, subjects_map):
        assert sorted(subjects_map) == ["sub-001", "sub-002"]
        assert list(subjects_map["sub-001"]) == ["ses-BL"]

    def test_sub_dirs_filter(self, bids_dir):
        subjects_map = invo.create_subject_map(bids_dir, sub_dirs=["sub-002"])
        assert list(subjects_map) == ["sub-002"]

    def test_sub_dirs_absolute(self, bids_dir):
        # subject_selector.py writes absolute subject directories, one per line
        subjects_map = invo.create_subject_map(bids_dir, sub_dirs=[str(bids_dir / "sub-001")])
        assert list(subjects_map) == ["sub-001"]

    def test_updating_subject_map_does_not_mutate(self, subjects_map, tmp_path):
        original = subjects_map["sub-001"]["ses-BL"]["input_path"]
        updated = invo.updating_subject_map(subjects_map, tmp_path / "pre")
        assert updated["sub-001"]["ses-BL"]["input_path"] == str(tmp_path / "pre" / "sub-001_ses-BL.nii.gz")
        assert subjects_map["sub-001"]["ses-BL"]["input_path"] == original


class TestRegistrationInvocations:
    def test_flirt_ieee(self, subjects_map, tmp_path):
        step = invo.FLIRT_IEEE_registration(subjects_map, tmp_path / "out", tmp_path / "inv", ref="/ref.nii.gz", template_name=TEMPLATE)
        step.create_invocations()

        inv_file = tmp_path / "inv" / "flirt" / TEMPLATE / "anat-12dofs" / "ieee" / "sub-001_ses-BL_invocation.json"
        out_dir = tmp_path / "out" / "flirt" / TEMPLATE / "anat-12dofs" / "ieee"
        assert json.loads(inv_file.read_text()) == {
            "in_file": subjects_map["sub-001"]["ses-BL"]["input_path"],
            "out_filename": str(out_dir / "sub-001_ses-BL.nii.gz"),
            "reference": "/ref.nii.gz",
            "out_mat_filename": str(out_dir / "sub-001_ses-BL.mat"),
            "dof": 12,
            "cost": "corratio",
        }

    @pytest.mark.parametrize(
        "cls, tool",
        [(invo.FLIRT_MCA_registration, "flirt"), (invo.ANTS_MCA_registration, "ants"), (invo.SPM_MCA_registration, "spm")],
    )
    def test_mca_layout(self, subjects_map, tmp_path, cls, tool):
        step = cls(subjects_map, tmp_path / "out", tmp_path / "inv", ref="/ref.nii", n_mca=3, template_name=TEMPLATE)
        step.create_invocations()

        assert collect(tmp_path / "inv") == [
            f"{tool}/{TEMPLATE}/anat-12dofs/mca/{i}/sub-{s}_ses-BL_invocation.json" for i in (1, 2, 3) for s in ("001", "002")
        ]
        # the step's base dirs are restored after the per-iteration loop
        assert step.invocation_dir == tmp_path / "inv" / tool / TEMPLATE / "anat-12dofs" / "mca"

    def test_ants_ieee_fields(self, subjects_map, tmp_path):
        step = invo.ANTS_IEEE_registration(subjects_map, tmp_path / "out", tmp_path / "inv", ref="/ref.nii.gz", template_name=TEMPLATE)
        invocation = step.create_single_subject_invocation(subjects_map["sub-001"]["ses-BL"])
        assert invocation["output_prefix"] == str(tmp_path / "out" / "ants" / TEMPLATE / "anat-12dofs" / "ieee" / "sub-001_ses-BL")
        assert invocation["transform_type"] == "a"

    def test_spm_ieee_fields(self, subjects_map, tmp_path):
        step = invo.SPM_IEEE_registration(subjects_map, tmp_path / "out", tmp_path / "inv", ref="/ref.nii", template_name=TEMPLATE)
        invocation = step.create_single_subject_invocation(subjects_map["sub-001"]["ses-BL"])
        prefix = str(tmp_path / "out" / "spm" / TEMPLATE / "anat-12dofs" / "ieee" / "sub-001_ses-BL")
        assert invocation["output_img"] == prefix + ".nii"
        assert invocation["output_mat"] == prefix + ".mat"


class TestDryRun:
    def test_create_invocations_writes_nothing(self, subjects_map, tmp_path, capsys):
        invo.FLIRT_MCA_registration(subjects_map, tmp_path / "out", tmp_path / "inv", n_mca=2, template_name=TEMPLATE).create_invocations(
            dry_run=True
        )
        assert not (tmp_path / "out").exists() and not (tmp_path / "inv").exists()
        assert "[DRY-RUN]" in capsys.readouterr().out

    def test_handle_preprocessing_writes_nothing(self, subjects_map, tmp_path):
        after = invo.updating_subject_map(subjects_map, tmp_path / "out" / "preprocess")
        invo.handle_preprocessing(subjects_map, after, tmp_path / "out", tmp_path / "inv", dry_run=True)
        assert not (tmp_path / "out").exists() and not (tmp_path / "inv").exists()

    def test_prepare_preprocess_directories(self, tmp_path):
        pre, unzipped = invo.prepare_preprocess_directories(tmp_path / "out", dry_run=True)
        assert (pre, unzipped) == (tmp_path / "out" / "preprocess", tmp_path / "out" / "preprocess_unzipped")
        assert not (tmp_path / "out").exists()

        invo.prepare_preprocess_directories(tmp_path / "out")
        assert pre.is_dir() and unzipped.is_dir()

    def test_handle_unzipping(self, tmp_path):
        src = tmp_path / "pre"
        src.mkdir()
        with gzip.open(src / "sub-001_ses-BL.nii.gz", "wb") as f:
            f.write(b"data")

        invo.handle_unzipping(src, tmp_path / "unzipped", dry_run=True)
        assert not (tmp_path / "unzipped").exists()

        invo.handle_unzipping(src, tmp_path / "unzipped")
        assert (tmp_path / "unzipped" / "sub-001_ses-BL.nii").read_bytes() == b"data"


class TestHandleReference:
    @pytest.fixture
    def gz_ref(self, tmp_path):
        path = tmp_path / "tpl.nii.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"template")
        return path

    def test_unzips_gz_and_returns_path(self, gz_ref):
        ref = invo.handle_reference(gz_ref)
        assert isinstance(ref, Path) and ref == gz_ref.with_suffix("")
        assert ref.read_bytes() == b"template"

    def test_dry_run_does_not_unzip(self, gz_ref):
        ref = invo.handle_reference(str(gz_ref), dry_run=True)
        assert ref == gz_ref.with_suffix("") and not ref.exists()

    def test_nii_returns_path(self, tmp_path):
        ref = invo.handle_reference(str(tmp_path / "tpl.nii"))
        assert isinstance(ref, Path) and ref == tmp_path / "tpl.nii"

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError):
            invo.handle_reference(tmp_path / "tpl.mgz")
