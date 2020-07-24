# dien

# document link:  
https://docs.google.com/document/d/1F5bXb-Tb1LnPraOv09UI0Il_P_9cND-MywXpEDEwQJI/edit  
<br/>
# running instruction:  
enable virtual environment  
unzip data-collection.zip and move all the data file into script/ seperately    

cd script/  
python train.py train DIEN  
or running in the background:  
python train.py train DIEN > dien_train.log 2>&1 &  
<br/>

# data preparation:  
header:  
label|uid|mid|...extra features...|mid_history|...extra feature history...  
uid: user id  
mid: item id  
extra feature: category, price etc  
extra history feature: history of category, price, etc. <br/> 
the order of extra feature need to be relatively consistent with extra feature history  
Eg:  
label|uid|mid|category|price|mid_history|category_history|price_history  <br/> 

# feature_voc.pkl files:  
each feature including uid and mid obtains a voc file that contains the encoding of it for mathematical calculation in the model.  

# feature config file:  
Num_feature_except_uid: the count of extra features plus mid  
voc_list: the list of voc file for each feature. It starts with uid_voc, mid_voc, and then extra features.  
The order of voc files in voc_list need to be consistent with the orders of features in the data.  <br/> 

# item-info:  
index file connect each item with its extra features  
data iterator will read this file and create a dict for item search  
header:  
mid|...extra features...  
Eg:  
mid|category|price  <br/> 
 
# reviews-info:  
index file contains action of user giving review score.  
data iterator will read this file and create negative item samples and history for each data sample.  
header:  
uid|mid|review score(review score is not used in the model)  





