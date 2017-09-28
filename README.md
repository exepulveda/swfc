# swfc
Spatial Weighted Fuzzy Clustering

folder structure:

data       contains 3 data sets:
           ds4.csv                is a sythetic 2D dataset for evaluation
           2d.csv                 is a 2D geometallurgical dataset
           synthetic_bm_small.csv is a full 3D sythetic geometallurgical dataset


#fortran modules
utils.f90      utilities, for example, wrapper to random number generator
distances.f90  different distances functions to support dimension distances 
               for continuous, categorical and targeted attributes
              
clustering.f90 fuzzy clustering functions supporting WFC


##How to build fortran extension
make


# Python modules
## Plotting
plotting.py          utilities for plotting using matplotlib

## Utilities
categorical_utils.py utilities for supporting categorical attributes
cluster_utils.py     utilities for clustering

## Graph Cut
graph_labeling.py    wrapper to pygco pytho extension which must be separatedly installed 

## GA Optimisation
clustering_ga.py     GA optimisation for SWFC method (uses deap Python package)

## Python scripts
METHOD               kmean, pca, wfc
DATASET              ds4, 2d, bm


boxplot_*.py            Generates boxplot charts
case_study_*.py         Loads the case study data
clustering_*_*.py       Performs clustering usign a METHOD to a DATASET
plot_*.py               Plot many charts, see doc in each script
spatial_correction_*.py Performs Spatial Correction (SWFC method)
