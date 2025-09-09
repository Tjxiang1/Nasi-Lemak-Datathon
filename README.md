# Content description for directories
**/AImodel**
- source code for training of machine learning model
- covers backend logic for dashboard decision making
- cleaned dataset used for model training
- requirements.txt contains required Python libraries

**/Data Preprocessing**
- .ipynb codes that covers entire Data Cleaning process and Exploratory Data Analysis

**/docs**
- source code to dashboard web development
- contains relevant libraries and files for dashboard setup

# Steps to setup the dashboard
1. Prerequisites 
    - NodeJS
    - Python

2. Create and Run a Virtual Environment
    - Run the following command in /AImodel
        - python -m venv .venv
        - .venv\Scripts\activate    
        - pip install -r requirements.txt

3. Activate the server 

    - Run the following command in root folder 
        - main.bat

4. View the website 

    - Type 'localhost:3000' in your search engine (ie: Chrome)
