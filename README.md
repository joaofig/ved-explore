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

## Using the Code

Run all the notebooks in sequence, starting from one.

Before running notebook number 10, you must first build the 
tile database using the following command, issued from the 
project's root:

`python generate_densities.py`

Please note that this may take a long time to run, most likely
more than on hour.
To explore the custom tiles, start the tile-serving
API with the following command, also issued from the project's 
root:

`python tileapi.py`

Now you can run notebook number 10.

## Medium Articles

All the links provided below should be freely accessible.
If not, please open an issue.

[The Medium-Sized Dataset](https://towardsdatascience.com/the-medium-sized-dataset-632cf0f15bb6) - Covers notebooks from 1 to 4.

[Geographic Clustering with HDBSCAN](https://towardsdatascience.com/geographic-clustering-with-hdbscan-ef8cb0ed6051) - Covers notebooks 5 to 7.

[Clustering Moving Object Trajectories](https://towardsdatascience.com/clustering-moving-object-trajectories-216c372d37e2?source=friends_link&sk=4a7688795231f03f901c33cae2d2ce2d) - Covers notebook 9.

[Geospatial Indexing with Quadkeys](https://towardsdatascience.com/geospatial-indexing-with-quadkeys-d933dff01496?source=email-64bc009cedeb-1601470855647-layerCake.autoLayerCakeWriterNotification-------------------------e7509b67_3f86_4235_a64a_9eeeb9f544a9&sk=0de1a65d95817fd8abc841fa60f7a279) - 
Covers the alternative version of notebook 7.

[Displaying Geographic Information Using Custom Map Tiles](https://towardsdatascience.com/displaying-geographic-information-using-custom-map-tiles-c0e3344909a4?source=friends_link&sk=df31039fbdfeafad2554e8c99673135f) - Covers notebook 10, `generate_densities.py`, and `tileapi.py`.
