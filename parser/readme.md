### A tool for parsing books from GEOKNIGA


Environment creation:

```commandline
conda create --name parser-env python=3.11
conda activate parser-env
```

Install dependancies:

```commandline
cd parser
pip install -e .
```

Go do job

Get help
```commandline
parser --help
```

Output:

* `source` **TEXT** Source to parse `[default: None]` [required]
* `page_limit` **INTEGER**  Number of pages to parse `[default: None]` [required]
* `file_limit` **INTEGER**  Limit of pdfs to download `[default: None]` [required]


Run
```
parser geokniga 1 5
```
which means that we need to parse first page of query and first 5 available files


Results are stored in `results` directory

* `geosearch.json` – list of all books that were found
* `geokniga_20250909_172312` - subdirectory with `downloads` and logs

