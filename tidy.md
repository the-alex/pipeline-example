---
title: Tidy Data & Reshaping
author: Tom Augspurger
---

> Structuring datasets to facilitate analysis [(Wickham 2014)](http://www.jstatsoft.org/v59/i10/paper)

So, you've sat down to analyze a new dataset.
What do you do first?

In episode 11 of [Not So Standard Deviations](https://www.patreon.com/NSSDeviations?ty=h), Hilary and Roger discussed their typical approaches.
I'm with Hilary on this one, you should make sure your data is tidy.
Before you do any plots, filtering, transformations, summary statistics, regressions...
Without a tidy dataset, you'll be fighting your tools to get the result you need.
With a tidy dataset, it's relatively easy to do all of those.

Hadley Wickham kindly summarized tidiness as a dataset where

1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

And today we'll only concern ourselves with the first two.
As quoted at the top, this really is about facilitating analysis: going as quickly as possible from question to answer.


```python
%pylab inline

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10
sns.set(style='ticks', context='talk')
plt.plot(np.random.randn(10), np.random.randn(10), kind='bar')
plt.show()
```
