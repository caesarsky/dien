# dien

data preparation:
header: 
label|uid|mid|...extra features...|mid_history|...extra history feature...

uid: user id
mid: item id
extra feature: category, price etc
extra history feature: history of category, price, etc

the order of extra feature need to be relatively consistent with extra history feature
Eg:
label|uid|mid|category|price|mid_history|category_history|price_history

feature config file:
Num_feature_except_uid: the count of extra features plus mid
voc_list: the list of voc file for each feature. It starts with uid_voc, mid_voc, and then extra features.
The order of voc files in voc_list need to be consistent with the orders of features in the data.


item-info:
index file connect each item with its extra features
data iterator will read this file and create a dict for item search
header:
mid|...extra features...
Eg:
mid|category|price


review-info:
index file contains the items that user reviewed.
data iterator will read this file and create negative item samples and history for each data sample.
header:
uid|mid|review score(review score is not used in the model)





