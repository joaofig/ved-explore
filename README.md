# ved-explore
Exploration of the Vehicle Energy Dataset

This code repository uses a SQLite database to explore the
[Vehicle Energy Dataset (VED)](https://arxiv.org/abs/1905.02081).

Small datasets are cool. You can load them into memory and 
manipulate them at will, no sweat. Massive datasets are also 
cool. They have lots of data and the promise of exciting 
models and analyses. You gladly pay the price of the required 
cluster just to handle all that goodness. Medium-sized 
datasets are a pain. They are either small enough to fit your 
RAM but too cumbersome to handle or just a bit larger than 
your memory but not worthy of the cluster cost. How do you 
tackle such a scenario?

There are several solutions to handle medium-sized datasets, 
and my favorite is to use a local database. You can fit the 
data in local storage and just bring into memory what you 
need to handle. An old but proven solution.

My regular setup is the Jupyter notebook supported by the 
Python machine learning and data analysis ecosystem. A 
natural local database choice for such an environment is 
SQLite. The Python distribution comes packaged with an 
implementation of this database with a straightforward and 
intuitive API.

In this repository, I illustrate such use with a dataset of 
vehicle fuel and electric energy consumption, which is 
medium-size, as described above.

## Medium Articles
[The Medium-Sized Dataset](https://towardsdatascience.com/the-medium-sized-dataset-632cf0f15bb6)

[Geographic Clustering with HDBSCAN](https://towardsdatascience.com/geographic-clustering-with-hdbscan-ef8cb0ed6051)
