## Getting the human genome data:

### Download hg19:

If on mac: run 
``` 
curl https://hgdownload.soe.ucsc.edu/goldenpath/hg19/bigZips/hg19.2bit -o hg19
```
If not on mac, try using wget instead of curl. 


### Get the human recombination data with :

``` 
curl ftp://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/recombRate.txt.gz -o hg19recombRate.txt.gz 
```

then unzip with:

```
gunzip hg19recombRate.gz
```
### Get the DSB Hotspot data:

click: 
https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE59836&format=file&file=GSE59836%5FPeak%5Fdata%5FSupplementary%5FFile%5F1%2Etxt%2Egz 

(or download GSE59836_Peak_data_Supplementary_File_1.txt.gz	1 from here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59836 ). 

# Getting the Rat Data

First, ya gotta download the rat genome, rn6.fa.gz. 

``` 
curl https://hgdownload.soe.ucsc.edu/goldenpath/rn6/bigZips/rn6.fa.gz -o rn6.fa.gz
```

```
gunzip rn6.fa.gz 
```

Download the rat dsb hotspot data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163474 Get: GSE163474_allMergedRatHotspots.finalTable.tab.gz	 

```
gunzip GSE163474_allMergedRatHotspots.finalTable.tab.gz
```

Then you need to get the recombination rate data for the rat genome. 

Download FileS2 from here https://figshare.com/articles/dataset/Supplement_for_Littrell_et_al_2018/6260888 

then 

```
gunzip FileS2.gz
```


# Getting the Mouse Data

Get the mouse genome from: https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/ 

```
curl ftp://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz --output mm10.fa.gz 
```
then 
``` 
gunzip mm10.fa.gz 
``` 

Download the recombination rate data from https://pmc.ncbi.nlm.nih.gov/articles/instance/3389972/bin/supp_112.141036_TableS1.csv

Download the mouse DSB hotspot data from https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE118913&format=file 



# Python time
create a virtual environment. easy way to do this is 
```
python -m venv env
```
Activate the environment with 
```
source env/bin/activate
```

Then install the requirements
``` 
pip install -r requirements.txt
```

# Processing Human Data
```
python process_hg19.py -i GSE59836_Peak_data_Supplementary_File_1.txt -g hg19
```


# Processing Rat Data
```
python process_da_rat.py  rn6.fa GSE163474_allMergedRatHotspots.finalTable.tab FileS2 rat_hotspots
```


# Processing Mouse Data
```
Start with 
```

# Running experiments

then 
```
python calculate_features.py
```
and then for training you can run
```
python rf_classifier.py -i <inputfile> -o <outputfile>
```
or 

``` 
python xg_classifier.py -i <inputfile> -o <outputfile>
```

the input file is going to end with something like "with_dinucleotides_and_motifs.tsv" e.g. rat_hotspots_1.2hot_with_dinucleotides_and_motifs.tsv. 

