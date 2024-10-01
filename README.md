# Experiments

Result:

![img.png](img.png)

Configuration:

![img_1.png](img_1.png)


Result:

![img_2.png](img_2.png)

Configuration:

![img_3.png](img_3.png)


Result:

![img_5.png](img_5.png)

Configuration:

![img_4.png](img_4.png)

# Structure

This project has the following scripts

| Script        | Description                                          |
|---------------|------------------------------------------------------|
| config.py     | In this file are the global variables, such as paths |
| clustering.py | This script preprocess and clusters the data         |
| plot_utils.py | Utilities for the plots                              |
| utils.py      | General utilities                                    |

## PREDICTION

Given a pretrained DenseClus  and a set of unlabeled 
observations, the function predict in utils.py, produces the approximated labels 
