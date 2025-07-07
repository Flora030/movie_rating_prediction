A machine learning project that predicts future movie ratings based on historical trends using a fully automated ETL pipeline and a Random Forest regression model.

Tech Stack
- **Python**
- **Pandas**, **NumPy** – for data manipulation and preprocessing  
- **Scikit-learn** – for building and training the Random Forest model  
- **Flask** – to expose the prediction model as a REST API  
- **Firebase** – for storing and accessing processed data  

Features
- Built an end-to-end **ETL pipeline** that:
  - Extracts movie rating data from online sources
  - Cleans and normalizes the data by removing duplicates and filling missing values
  - Prepares the dataset for machine learning
- Trained a **Random Forest** model using Scikit-learn to predict future movie ratings
- Hosted the model using Flask and integrated Firebase for real-time data access and storage

Prediction Target
The model predicts numerical ratings for upcoming or newly released movies based on similar historical data patterns.

Future Improvements
- Integrate external APIs for real-time movie metadata
- Improve model accuracy with additional features like genre, cast, and user ratings
- Deploy the application with Docker and host on cloud 
