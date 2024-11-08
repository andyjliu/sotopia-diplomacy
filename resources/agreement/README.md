# Agreement metrics
This script computes pairwise agreement, random pairwise agreement, and Krippendorf's alpha.

## Usage
Given a Pandas DataFrame longDf
```python
In [1]: longDf.head()
Out[1]:
      id raterId  rating  ratingBinary
   0   0      r1       4           1.0
   1   1      r1       1           0.0
   2   2      r1       1           0.0
   3   3      r1       1           0.0
   4   4      r1       1           0.0
```
you can compute the scores as
```python
In [2]: scores = computeAlpha(longDf,"ratingBinary",groupCol="id")
In [3]: scores
Out[3]:
{'alpha': 0.2635467980295564,
 'ppa': 0.6399999999999999,
 'rnd_ppa ': 0.5111705685618729,
 'skew': 0.42,
 'n': 100}
```
:warning: **For non-binary ratings**: Normalize (min-max) the ratings in the data frame before using the script.
