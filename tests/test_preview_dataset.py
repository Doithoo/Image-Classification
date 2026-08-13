import pytest
from PIL import Image
from scripts.preview_dataset import collect_images, create_preview


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
