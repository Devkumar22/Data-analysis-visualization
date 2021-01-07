# Data-analysis-visualization
analysis using pandas & seaborn,Matplotlib visualization
pandas


import pandas as pd
import numpy as np
s = pd.Series([1,2,3,4,5,6,np.nan,8,9,10])
s


d = pd.date_range('20200604' , periods=10)
d

df = pd.DataFrame(np.random.randn(10,4), index=d,columns=['A','B','C','D'])
df

q = pd.DataFrame({'A': [1,2,3,4,5,6,7],
                  'B': pd.Timestamp('20200604'),
                  'C': pd.Series(data=44, index=list(range(7)), dtype='int32'),
                  'D': 'Edureka',
                  'E': pd.Categorical(['crisp','flip','desk','trick','comp','crust','hubber',])
})
q


q.head()
q.tail()
q.describe()

q.sort_index(axis=0, ascending=True)
q.sort_values(by='A')
q[0:4] 



df.loc['20200604':'20200609',['A','C']]


df2=df.reindex(index=d[0:4], columns=list(df.columns) + ['E'])


df2.dropna()

df2.loc[d[0]:d[1],['E']]=1
df2

df2.fillna(value=2)


df2=[df[:3],df[3:7],df[7:]]

df2

pd.concat(df2)

df3=df.reindex(index=[], columns=['A','B','C','D'])

import numpy as np
import pandas as pd

multiindex

my_tuple=list[zip*[1,2,3,4,5,6], [7,8,9,10,11,12,]]
multiindex=pd.Multiindex.from_tuples(my_tuple, names=index1,index2)
df4= pd.DataFrame(np.randn(8,2), index=multiindex, columns=['Y','Z'])

g=pd.DataFrame({ 'A': ['a','b','c','d']*3,
                 'B': ['A','B','C']*4,
                'C': ['p','p','p','q','q','q'],
                'D': np.random.randn(12),
                'E': np.random.randn(12)
})

pd.pivot_table(g, values='D', index=['A','B'], columns=['C'])

plotting

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
ts = pd.Series(np.random.randn(30), index = pd.date_range('1/6/2020', periods= 30))

ts.plot()

ts.to_csv("ts.csv")

pd.read_csv(r'typelocation in it')

MATPLOTLIB
DATAVISUALISATION



import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,3])
plt.show()

x=[1,2,3]
y=[4,2,7]
plt.plot(x,y)
plt.title("plots")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

from matplotlib import style
x=[1,2,3]
y=[5,9,7]
x2=[9,8,7]
y2=[4,3,8]
plt.plot(x,y,'g',label='line-one',linewidth=5)
plt.plot(x2,y2,'b',label='line-two',linewidth=7)
plt.title('tilt')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()     #for showing scale
plt.grid(True,color='r')
plt.show()

bargraphs


import matplotlib.pyplot as plt

plt.bar([1,2,3,4,5],[5,6,7,8,2], label="example-one")
plt.bar([4,3,5,2,7],[6,9,1,8,9],label="example-two",color='g')
plt.legend()
plt.xlabel('bar-number')
plt.ylabel('bar-height')
plt.show()

histograms(used for quantitative data)

population=[22,34,33,66,74,87,56,45,98,23,78,56,78,45,96,19]
bins=[0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population,bins,histtype='bar',rwidth=0.8)
plt.xlabel('bins')
plt.ylabel('ages')
plt.legend()
plt.title('histogram')
plt.show()

scatterplots-----(used to compare two or more variables)


x=[1,2,3,4,5,6,7]
y=[4,5,2,6,8,3,8]
plt.scatter(x,y, label='points',color='r')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()

# seaborn dataviz



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Tabulation


import urllib.request
urllib.request.urlretrieve(url, 'data.json')

styling tabulation

def colour_negative_red(x):
    color = 'red' if x < 0 else 'white'
    return 'color: ' + color
df.style.applymap(color_red_negative)


df.style.highlight_max(color= 'red').highlight_min(color= 'green' )

df.style.background_gradient(cmap= 'Reds')
#can also usesubsets
df.style.background_gradient(cmap= 'Reds', subset='mh','tn')

df.style.bar()

df[['mh','tn','dl']].style.bar()

#histograms

import seaborn as sns
import numpy as np
sns.set(color_codes=True)
x=np.random.normal(size=1000)
sns.distplot(x)

sns.kdeplot(x)

sns.distplot(x, kde=False, rug=True, bins=50)

sns.kdeplot(x, shade=True)

#Boxplot

x=np.random.normal(size=1000)
sns.boxplot(x)

sns.boxplot(x, whis=0.8)

sns.boxplot(x,  whis=0.5, fliersize=2)



#joint distribution of two variables


import seaborn as sns
import numpy as np
import pandas as pd
sns.set(color_codes=True)

x=np.random.normal(size=1000)
y=np.random.normal(size=1000)


df= pd.DataFrame({'x':x, 'y':y})
sns.jointplot('x','y', data=df)

sns.jointplot('x','y', data=df, kind='kde',shade=True);

z=np.random.normal(size=100)
sns.swarmplot(z)

#violin plot

x=np.random.normal(size=1000)
sns.violinplot(x)

import matplotlib.pyplot as plt

fig, axs= plt.subplots(nrows=4)
fig.set_size_inches(5,10)
sns.violinplot(x,ax=axs[0])
sns.boxplot(x,ax=axs[1])
sns.swarmplot(x,ax=axs[2])
sns.distplot(x,ax=axs[3])

import pandas as pd
c = pd.read_csv("movie_dataset.csv")
df = pd.DataFrame(data=c)
df.head()



import seaborn as sns
import matplotlib.pyplot as plt

c = pd.read_csv("nifty.csv")
df = pd.DataFrame(data=c)
sns.kdeplot(df.Date)


