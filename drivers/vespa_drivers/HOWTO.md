To run vanilla vespa:
    $ (vespa) starfit --all dirname_with_ini_scripts
    $ (vespa) calcfpp -n 20000 dirname_with_ini_scripts

To incorporate observational constraints (e.g., no HEBs, as in the TOI-837
calculation) both the FPP calculation, and also the quicklook summary plots:

    $ (vespa) starfit --all dirname_with_ini_scripts
    $ (vespa) calcfpp -n 20000 dirname_with_ini_scripts
    $ (vespa) python run_simple.py -n 20000 dirname_with_ini_scripts

