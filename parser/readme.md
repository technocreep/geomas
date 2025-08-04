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

Log examples

```logs
2025-08-04 18:38:28 - PARSER - Parser initiated for geokniga with 1 pages and 5 files
2025-08-04 18:38:28 - PARSER - Process started
2025-08-04 18:38:28 - PARSER - Searching for books on 1 page...
2025-08-04 18:38:29 - PARSER - Found 10 book links
2025-08-04 18:38:31 - PARSER - List of books is saved at:
2025-08-04 18:38:31 - PARSER - results/geokniga_20250804_183828/geosearch.json
2025-08-04 18:38:31 - PARSER - Loading 5/10 books
2025-08-04 18:38:34 - PARSER - [✔] Downloaded: results/geokniga_20250804_183828/downloads/geokniga-analizproizvodstvenno-hozyaystvennoydeyatelnostigeologicheskihorganizaciy.pdf
2025-08-04 18:39:09 - PARSER - [✔] Downloaded: results/geokniga_20250804_183828/downloads/geokniga-aerogammaspektrometricheskiy-metod-poiskov-rudnyh-mestorozhdeniy.pdf
2025-08-04 18:39:22 - PARSER - [✔] Downloaded: results/geokniga_20250804_183828/downloads/geokniga-atmogeohimicheskiemetodypoiskov.pdf
2025-08-04 18:39:26 - PARSER - [✔] Downloaded: results/geokniga_20250804_183828/downloads/geokniga-avtomatizaciya-processa-geologorazvedochnogo-bureniya.pdf
2025-08-04 18:40:06 - PARSER - [✔] Downloaded: results/geokniga_20250804_183828/downloads/geokniga-aerokosmicheskie-issledovaniya-na-regionalnom-etape_0.pdf
2025-08-04 18:40:06 - PARSER - Process finished
```
