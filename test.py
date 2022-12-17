import numpy as np
import matplotlib.pyplot as plt


A = [0.42,                       
15.77,                              
34.24 ,                             
47.12  ,                           
57.55   ,                          
68.7     ,                         
83.69     ,                        
94.52      ,                       
108.42      ,                      
125.11       ,                     
140.59        ,                    
153.59         ,                   
165.96          ,
180.39]
# accumilated sum
A = np.diff(A).mean()
print(A)                  