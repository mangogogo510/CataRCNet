# CataRCNet
COMP0132 UCL MSc Robotics and Computation Dissertation 

CataRCNet: Surgical Workflow Analysis for Cataracts

1. Data Pre-processing: Convert videos to frames using "Step1_data_preprocessing.ipynb"
2. Generate the paths and labels of train set (train06-train25, index 5-24), validation set (train01-train05, index 0-4, and test set (test01-test25).
3. Train M1,M2,M3 for long-range feature bank 
4. Train M4-M9, (first training, we need to load feature bank)
5. Test: test->pkl->csv->challenge eval
