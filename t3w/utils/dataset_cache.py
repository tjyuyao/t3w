from pathlib import Path
import mgzip
import pickle


class CachedDataset:

    def __init__(self, raw_dataset, cache_dir, compresslevel=1):
        self.raw_dataset = raw_dataset
        self.cache_dir = Path(cache_dir)
        self.compresslevel = compresslevel
        Path(self.cache_dir).mkdir(exist_ok=True)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        cached_datum = self.cache_dir / (str(index) + ".gz")
        if cached_datum.exists():
            with mgzip.open(str(cached_datum)) as fp:
                datum = pickle.load(fp)
        else:
            datum = self.raw_dataset[index]
            with mgzip.open(str(cached_datum), "wb", compresslevel=self.compresslevel) as fp:
                pickle.dump(datum, fp)
        return datum

    @property
    def datum_type(self):
        return self.raw_dataset.datum_type
