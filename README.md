### Starter pack for participating in CAIP 2019 local descriptor contest

This is python started pack depending on numpy and pytorch for particiation in [CAIP 2019 contest](http://cvg.dsi.unifi.it/cvg/index.php?id=caip-2019-contest)

Official starter pack is in matlab, so I have prepared python version.

How to use it:
    
1. Download the dataset and unpack, following the [official instructions](https://drive.google.com/drive/folders/1eX5CKvzcOhJUN8gwDogfquN0uskCXflD)

2. Put this repo into "code" directory

3. Run 
```bash
        python extract_hardnets.py
        python match_hardnets.py
 ```

4. Now you have ```output_data``` folder, which you may zip and send to organizers.

5. You might want to explore the data with [EDA.ipynb](EDA.ipynb) script and visually check a validity of your submission via [sanity_check_submission.ipynb](sanity_check_submission.ipynb) 
