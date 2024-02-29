# Tailored Tails, Find Your Perfect Pup  
  
![puppies](https://github.com/StarkArk/Tailored_Tails/blob/main/Visualizations/images/istockphoto-puppies.png)  
  
## Finding Your Ideal Dog Companion
Our initiative, tailored_tails, is dedicated to helping potential dog owners navigate the adoption process with confidence. By collecting and analyzing diverse data within a Machine Learning Model, we provide personalized recommendations on dog breeds that best match individual lifestyles and preferences.  
  
## Our Dogs, A New York Story   
  
A New York Dog Story: [Tableau Story](https://public.tableau.com/app/profile/wingtung.lee/viz/UCB_Bootcamp_Project4-5/Story1)
  
## Meet the Minds Behind tailored_tails

![The_Team](https://github.com/StarkArk/Tailored_Tails/blob/main/Visualizations/images/Profiles_Project_Members/our_team.PNG)
  
## Data Companions
### Finding the Perfect Dog Breed for You
To identify the most suitable dog breed, we gathered the following information and integrated some of them into our machine learning model:
* **Characteristics of the Dog:** Ability to Learn, Personality, Activity Level
* **Lifestyle of the Owner:** Gender, Age, Type of Family, Living Area
* **Cost Considerations:** Regularity of Grooming, Overall Ownership Expenses  
  
## Web App  
  
![Front Page](https://github.com/StarkArk/Tailored_Tails/blob/main/Visualizations/images/web_app_frontpage.png) 
  
## Additional Visualizations to Explore  
  
[Visualizations Folder](https://github.com/StarkArk/Tailored_Tails/tree/main/Visualizations/images)  
  
## Modelling Process 
  
After pulling the dog breed, NYC, and related data from [various sources](https://github.com/StarkArk/Tailored_Tails/tree/main/Exploration/doggy_data) we created cleaned csv files. These [files](https://github.com/StarkArk/Tailored_Tails/tree/main/Exploration/cleaned_data)
were then merged together and further cleaned. The product csv [file](https://github.com/StarkArk/Tailored_Tails/blob/main/Modeling/preprocessed_doggy.csv) was then used in our modelling [notebooks](https://github.com/StarkArk/Tailored_Tails/tree/main/Modeling) to produce 
our ['Random Forest Model'](https://github.com/StarkArk/Tailored_Tails/blob/main/Modeling/breed_rf_model.pkl). 24 features are used to predict 50 possible targets(Breeds). With some experimentation we settled on Random Forest Classifier model. Other models were explored as well and
,after some optimization, they produced results that were in line with what was obtained from the Random Forest Classifier.

## Model Performance  
  
![RandomForest_Classification_Table](https://github.com/StarkArk/Tailored_Tails/blob/main/Visualizations/images/RandomForest_Model_Classification_Report.png)  
  
![RandomForest_Classification_Summary_Table](https://github.com/StarkArk/Tailored_Tails/blob/main/Visualizations/images/RandomForest_Model_Classification_Summary.png)
  
## Model Discussion  
  
Our predictive model demonstrated a high degree of accuracy for 49 out of the 50 top breeds(American Kennel Club) with an overall accuracy exceeding 99%. The 'Bloodhound' was the odd dog out of the group. We believe this was due to the 'Bloodhound'
having similar traits to other dogs in the dataset and having fewer examples represented in the data compared to the other breeds. 

#### Why isn't the model able to predict 'Bloodhounds' accurately?  
  
Possible Reasons:  
  
- There are only 71 'Bloodhound' entries and the model was trained on a total 206,816 registrations. Perhaps there were not enough entries to pick up a distinct pattern.
- The 'Bloodhound' could share common traits with other more prevalent Breeds.
- The owner profiles for those who licensed a 'Bloodhound' could be diverse and the model may not be able to find a distinquishable pattern in them.  
  
## Resources
### Libraries and Dependencies  
Python Pandas  
Python Matplotlib  
Python Seaborn  
Python Flask  
Python Scikit-learn  
Javascript  
HTML  
Google Colab   
Tableau  
  
## References  
  
#### NYC Open Data

- Dog Bites: DOHMH Dog Bite Data, updated to 1/19/2022
[NYC Dog Bites Data](https://data.cityofnewyork.us/Health/DOHMH-Dog-Bite-Data/rsgh-akpg/about_data)
- NYC Dog Licensing Dataset, updated 2/6/2024
[NYC Dog Licensing](https://data.cityofnewyork.us/Health/NYC-Dog-Licensing-Dataset/nu7n-tubp/about_data)
- Dog Runs and Off Leash Areas, updated 12/29/2020
[Dog Runs/Off Leash Locations](https://data.cityofnewyork.us/Recreation/Directory-of-Dog-Runs-and-Off-Leash-Areas/ipbu-mtcs/about_data)
- NYC Parks Dog Runs, GIS Data
[GIS Data](https://data.cityofnewyork.us/Recreation/NYC-Parks-Dog-Runs/8nac-uner)
[More GIS Data](https://data.cityofnewyork.us/Recreation/DogRuns_20190417/hxx3-bwgv/about_data)  
- NYC Demographics, updated 10/10/2018 [NYC Demograghics and Housing by Borough Data](https://data.cityofnewyork.us/City-Government/Demographic-and-Housing-Profiles-by-Borough/cu9u-3r5e/about_data)  
 
#### American Kennel Club  
  
- American Kennel Club Breeds [AKCBreeds](https://www.akc.org/dog-breeds/)

#### IRS 

- IRS Income Data, Tax Year 2020 [IRS Data](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2020-zip-code-data-soi)

#### Dog Costs  
  
- The iHeartDogs [iheartdogs](https://www.iHeartDogs.com)