import pathlib


def test_merfish_main():
    test_dir = pathlib.Path(__file__).parent.absolute()
    test_data = test_dir.joinpath("data/ten_rows_of_moffitt.csv")
    assert test_data.is_file()
