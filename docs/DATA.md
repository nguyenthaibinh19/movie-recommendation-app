# Dataset and asset notes

The data source is [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/), maintained by GroupLens Research. Refer to the [original dataset README and usage terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt).

Raw dataset files are excluded from the current source tree. Run `python scripts/download_dataset.py` to obtain them directly from the official source for local research use. The script preserves existing files and retrieves the official README with the archive. Earlier repository history contains raw dataset copies; removing them from the current tree does not remove those historical copies. No Git history rewrite is performed by this update.

Dataset use and redistribution remain subject to GroupLens terms. Attribution:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems. DOI: 10.1145/2827872.

No GroupLens endorsement is implied. Checkpoint files and other third-party material retain any applicable terms; this project does not relicense the dataset.

Movie posters, backdrops and descriptions are requested from TMDB at runtime, not copied into the repository. TMDB receives the lookup title/year; demographic inputs and session IDs are sent only to the configured recommendation API. Official TMDB attribution appears in the footer when its API is configured. No API key is included in tracked frontend source. Keys that appeared in older Git history are not removed by editing the current file.
