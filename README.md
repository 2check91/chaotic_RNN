# Chaotic Recurrent Neural Networks
Implementation of RNN model described in "Generating Coherent Patterns of Activity from Chaotic Neural Networks" (David Sussillo and L. F. Abbott, 2009, http://www.theswartzfoundation.org/docs/Sussillo-Abbott-Coherent-Patterns-August-2009.pdf)

**filtered_data.mat** is obtained from VRep Robotics simulator (Asimo robot). Smoothing using moving average filter was applied to data. 

To obtain correct output, change following in **scipy/sparse/construct.py**:

use  
vals = random_state.randn(k).astype(dtype) *# returns a sample from the “standard normal” distribution*  
instead of  
vals = random_state.rand(k).astype(dtype) *#  returns a sample from a uniform distribution over [0, 1)*
