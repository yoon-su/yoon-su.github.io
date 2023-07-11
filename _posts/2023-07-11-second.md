---
layout: post
title:  "테스트2 입니다!"
---



```python
# Baseline removal code using python BaselineRemoval module
# https://pypi.org/project/BaselineRemoval/
# Input: X-,Y1,Y2,Y3, ... multi-column csv file
# Output: Baseline subtracted csv file, fitting graph

from BaselineRemoval import BaselineRemoval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

    
    Bad key "text.kerning_factor" on line 4 in
    /home/yuk/anaconda3/envs/pytorch_17/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/_classic_test_patch.mplstyle.
    You probably need to get an updated matplotlibrc file from
    https://github.com/matplotlib/matplotlib/blob/v3.1.3/matplotlibrc.template
    or from the matplotlib source distribution



```python
csv_before = "XRD_before.CSV"
df_b = pd.read_csv(csv_before, names = ["Angle", "Intensity"], index_col = 'Angle')
#df = df.set_index(["Angle", "Intensity"])
print(df_b.head(10))
```

           Intensity
    Angle           
    10.00       1904
    10.01       1901
    10.02       1877
    10.03       1971
    10.04       1926
    10.05       1923
    10.06       1905
    10.07       1924
    10.08       1970
    10.09       1882



```python
csv_after = "XRD_after.CSV"
df_a = pd.read_csv(csv_after, names = ["Angle", "Intensity","Baseline"], index_col = 'Angle')
print(df_a.head(10))
```

           Intensity    Baseline
    Angle                       
    10.01   22.68297  1878.31703
    10.02   -2.13030  1879.13030
    10.03   91.05346  1879.94654
    10.04   45.23425  1880.76575
    10.05   41.41208  1881.58792
    10.06   22.58695  1882.41305
    10.07   40.75887  1883.24113
    10.08   85.92785  1884.07215
    10.09   -2.90612  1884.90612
    10.10   75.25699  1885.74301



```python
df_b.iloc[:,0:1].plot()
plt.title("TEST")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```


![png](output_3_0.png)



```python
# Baseline subtraction using ModPoly scheme
Mdf = pd.DataFrame(index = df_b.index, columns = df_b.columns)
print(Mdf.head())
```

          Intensity
    Angle          
    10.00       NaN
    10.01       NaN
    10.02       NaN
    10.03       NaN
    10.04       NaN



```python
# only needed for Modpoly and IModPoly algorithm
# 1차 직선식 baseline을 원하면 1, 2차 곡선 baseline은 2 이상
polynomial_degree=2

for Y_col in df_b: #looping for every columns in df
        input_array = df_b[Y_col].tolist() # as list
        # print(input_array)

        baseObj=BaselineRemoval(input_array)
        Modpoly_output=baseObj.ModPoly(polynomial_degree)
        Mdf[Y_col] = Modpoly_output
Mdf = Mdf.apply(lambda x: round(x, 2))
print(Mdf.head(10))
```

           Intensity
    Angle           
    10.00      59.58
    10.01      55.54
    10.02      30.50
    10.03     123.45
    10.04      77.41
    10.05      73.37
    10.06      54.33
    10.07      72.29
    10.08     117.25
    10.09      28.20



```python
Mdf.to_csv(csv_before.split('.')[0]+'_Mod_test'+'.csv')
```


```python
print(Mdf.iloc[:,0:1].head())

```

           Intensity
    Angle           
    10.00      59.58
    10.01      55.54
    10.02      30.50
    10.03     123.45
    10.04      77.41



```python
Mdf.iloc[:,0:1].plot()
plt.title("TEST")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```


![png](output_8_0.png)



```python
df_total = pd.concat([df_b,Mdf], axis = 1)
```


```python
df_total.iloc[:,0:2].plot()
plt.title("TEST")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```


![png](output_10_0.png)



```python
print(df_total.iloc[:,0:2].head())
```

           Intensity  Intensity
    Angle                      
    10.00       1904      59.58
    10.01       1901      55.54
    10.02       1877      30.50
    10.03       1971     123.45
    10.04       1926      77.41



```python
df_total2 = pd.concat([df_total,df_a.iloc[:,0:2]], axis = 1)
df_total2.head()
#df_total2.set_index(''['Angle','Intensity_raw','BG_python','BG_truth'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intensity</th>
      <th>Intensity</th>
      <th>Intensity</th>
      <th>Baseline</th>
    </tr>
    <tr>
      <th>Angle</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10.00</th>
      <td>1904</td>
      <td>59.58</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10.01</th>
      <td>1901</td>
      <td>55.54</td>
      <td>22.68297</td>
      <td>1878.31703</td>
    </tr>
    <tr>
      <th>10.02</th>
      <td>1877</td>
      <td>30.50</td>
      <td>-2.13030</td>
      <td>1879.13030</td>
    </tr>
    <tr>
      <th>10.03</th>
      <td>1971</td>
      <td>123.45</td>
      <td>91.05346</td>
      <td>1879.94654</td>
    </tr>
    <tr>
      <th>10.04</th>
      <td>1926</td>
      <td>77.41</td>
      <td>45.23425</td>
      <td>1880.76575</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_total2.iloc[:,0:3].plot()
plt.title("TEST")
plt.xlabel("angle")
plt.ylabel("Intensity")
plt.show()
```


![png](output_13_0.png)



```python
df_total2.columns = ['Intensity_raw','BG_python','BG_truth','Baseline']
```


```python
df_total2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intensity_raw</th>
      <th>BG_python</th>
      <th>BG_truth</th>
      <th>Baseline</th>
    </tr>
    <tr>
      <th>Angle</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10.00</th>
      <td>1904</td>
      <td>59.58</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10.01</th>
      <td>1901</td>
      <td>55.54</td>
      <td>22.68297</td>
      <td>1878.31703</td>
    </tr>
    <tr>
      <th>10.02</th>
      <td>1877</td>
      <td>30.50</td>
      <td>-2.13030</td>
      <td>1879.13030</td>
    </tr>
    <tr>
      <th>10.03</th>
      <td>1971</td>
      <td>123.45</td>
      <td>91.05346</td>
      <td>1879.94654</td>
    </tr>
    <tr>
      <th>10.04</th>
      <td>1926</td>
      <td>77.41</td>
      <td>45.23425</td>
      <td>1880.76575</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69.95</th>
      <td>3543</td>
      <td>547.10</td>
      <td>116.85275</td>
      <td>3426.14725</td>
    </tr>
    <tr>
      <th>69.96</th>
      <td>3611</td>
      <td>615.76</td>
      <td>185.00623</td>
      <td>3425.99377</td>
    </tr>
    <tr>
      <th>69.97</th>
      <td>3499</td>
      <td>504.42</td>
      <td>73.15959</td>
      <td>3425.84041</td>
    </tr>
    <tr>
      <th>69.98</th>
      <td>3408</td>
      <td>414.08</td>
      <td>-17.68718</td>
      <td>3425.68718</td>
    </tr>
    <tr>
      <th>69.99</th>
      <td>3426</td>
      <td>432.74</td>
      <td>0.46592</td>
      <td>3425.53408</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 4 columns</p>
</div>




```python
print(df_total2.iloc[856:860])
```

           Intensity_raw  BG_python     BG_truth    Baseline
    Angle                                                   
    18.56          38638   36004.75  35605.72495  3032.27505
    18.57          40559   37924.96  37525.48943  3033.51057
    18.58          40573   37938.16  37538.25510  3034.74490
    18.59          39447   36811.36  36411.02197  3035.97803



```python
print(df_total2.iloc[3455:3459])
```

           Intensity_raw  BG_python     BG_truth    Baseline
    Angle                                                   
    44.55          17708   13954.31  13796.88327  3911.11673
    44.56          17792   14038.24  13880.98638  3911.01362
    44.57          17896   14142.18  13985.08966  3910.91034
    44.58          17814   14060.12  13903.19310  3910.80690



```python
37938.16/14142.18
```




    2.68262460243046




```python
37538.25510/13985.08966
```




    2.684162634106416




```python
df_total2.to_csv(csv_before.split('.')[0]+'_removed'+'.csv')
```


```python

```
