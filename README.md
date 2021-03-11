# DS-Capstone-Movielens
One of two projects for the edX / HarvardX Data Science Capstone course

There are four files related to this project:
1. DS-Capstone-Movielens.R
2. DS-Capstone-Movielens.Rmd
3. DS-Capstone-Movielens-Report.pdf
4. .gitignore

The R script and Markdown file may take ~ 30 minutes to run. Setting up the datasets, edx and vlaidation, are particularly time intensive. For this reason, the script stores these as .rds files and reads them into memory if they've been previously created. 

A number of libraries are used in the analysis. Both the R script and .Rmd file will check if they are installed and, if not, install them, before loading the library.

The output is formatted to .pdf.