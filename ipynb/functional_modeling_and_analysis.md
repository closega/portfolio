---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.7.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Functional modeling and quantitative system analysis in python

This notebook is an appendix to the following article:

* G. Close, “Functional modeling and quantitative system analysis in python,” Towards Data Science, 31-Jan-2021. Available: https://towardsdatascience.com/functional-modeling-and-quantitative-system-analysis-in-python-22b90bf0b9b5. [Accessed: 31-Jan-2021]

The illustrative functional block diagram analyzed in the article is shown below.

```{code-cell} ipython3
from IPython.display import Image
Image("functional_bd_example.png", width=700)
```

## Code listing

The complete code below model each functional block with a corresponding Python function.
Each function is wrapped to comply with the coding pattern introduced in the article.
To simulate the block diagram, each function is called in turn mimicking the flow depicted in block diagram.
The analysis results are summarized in an interactive dashboard, reproducing the final figure of the article.

```{code-cell} ipython3
#----------------------
# Common packages
#----------------------

import numpy as np
import scipy
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
pn.extension()
from box import Box
import itertools

#----------------------
# Function wrapper
#----------------------

def apply_func_df(func):
  def wrapper(df, dd):
    Y = df.apply(func, dd=dd, axis=1, result_type='expand')
    return df.drop(columns=Y.columns, errors='ignore').join(Y)
  wrapper.__wrapped__ = func
  return wrapper

#----------------------
# Data dictionary
# stored in series (or Box)
#----------------------

dd=pd.Series({
  'offset_u' : 0.01, 
  'offset_v' : 0.0,
  'noise_std': 0.01,
})

#----------------------
# Functional blocks
#----------------------

@apply_func_df
def polar2cart(X, dd):
  return {
    'u': X.A * np.cos(X.theta_i) + dd.offset_u,
    'v': X.A * np.sin(X.theta_i) + dd.offset_v,
  }

# Already data frame compatible ==> No need to wrap
def add_noise(df, dd):
  df[['u', 'v']] += np.random.randn(len(df),2)*dd.noise_std
  return df

@apply_func_df
def calc_angle(X, dd):
  return {
    'theta_o': np.arctan2(X.v, X.u)
  }

@apply_func_df
def calc_angle_err(X, dd):
  e = X.theta_o - X.theta_i 
  # phase wrapping
  if e > np.pi:
    e-=2*np.pi
  elif e < -np.pi:
    e+=2*np.pi
  return {
    'theta_err': e
  }

# Already data frame compatible ==> No need to wrap
def convert_to_deg(df, dd):
  df[df.filter(like='theta_').columns]=np.degrees(df[df.filter(like='theta_').columns])
  return df
  
#----------------------
# Stimuli
#----------------------
df = pd.DataFrame(itertools.product(
    np.arange(0,2*np.pi,np.pi/100),
    [1,2]),
    columns=['theta_i', 'A'])

#----------------------
# Simulation: execute the pipeline
#----------------------
df=(df
 .pipe(polar2cart,     dd=dd)
 .pipe(add_noise,      dd=dd)
 .pipe(calc_angle,     dd=dd)
 .pipe(calc_angle_err, dd=dd)
 .pipe(convert_to_deg, dd=dd) )

#----------------------
# Analysis
#----------------------
g1=df.hvplot(groupby='A',  x='theta_i', y='theta_err', 
          xlabel='Input angle [°]', ylabel='Angle error [°]',
          ylim=[-2,2],   
          width=600, widget_location='bottom')
summary=pd.pivot_table(df, index='A', values=['theta_err'], aggfunc=lambda x: np.sqrt((x**2).mean()))

#----------------------
# Dashboard
#----------------------
hv.extension('bokeh')
panel=pn.Row(pn.Column('### Angle error curve',  g1),
       pn.Column('### Root mean square error',  
       pn.widgets.DataFrame(summary.applymap("{0:.2f}°,rms".format), width=200)))
panel.embed()
```
