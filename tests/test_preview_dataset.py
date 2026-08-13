import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "preview_dataset.py"
SPEC = importlib.util.spec_from_file_location("preview_dataset", SCRIPT)
assert SPEC and SPEC.loader
preview_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview_dataset)

collect_images = preview_dataset.collect_images
create_preview = preview_dataset.create_preview


def test_collect_images_groups_supported_files(tmp_path):
    data = tmp_path / "data"
    (data / "paper").mkdir(parents=True)
    (data / "trash").mkdir()
    Image.new("RGB", (8, 8), "white").save(data / "paper" / "one.jpg")
    Image.new("RGB", (8, 8), "black").save(data / "trash" / "one.png")
    (data / "paper" / "notes.txt").write_text("ignore")

    grouped = collect_images(data)

    assert [path.name for path in grouped["paper"]] == ["one.jpg"]
    assert [path.name for path in grouped["trash"]] == ["one.png"]


def test_create_preview_writes_contact_sheet_and_counts(tmp_path):
    data = tmp_path / "data"
    (data / "paper").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(data / "paper" / "one.jpg")
    output, counts = create_preview(data, tmp_path / "preview" / "sheet.png", samples_per_class=1)

    assert output.exists()
    assert counts.read_text() == "class,count\npaper,1\n"
    assert Image.open(output).size[0] > 0


@pytest.mark.parametrize("kwargs", [{"samples_per_class": 0}, {"tile_size": 16}])
def test_create_preview_rejects_invalid_layout_options(tmp_path, kwargs):
    data = tmp_path / "data"
    (data / "paper").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(data / "paper" / "one.jpg")

    with pytest.raises(ValueError):
        create_preview(data, tmp_path / "preview.png", **kwargs)
