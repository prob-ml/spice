import pathlib

from spatial.monet_merfish_ae import merfish_main


def test_merfish_main():
    test_dir = pathlib.Path(__file__).parent.absolute()
    merfish_main(test_dir + "data/ten_rows_of_moffitt.csv")
