# Diabetes Analysis Project

## Overview

This study delves into an examination of a dataset, on diabetes investigating aspects, like exercise habits, age, income, body mass index (BMI) mental well being and dietary choices to comprehend how they influence an individuals diabetes condition.

## Data Source

The dataset was obtained from [your source if applicable, otherwise just state "publicly available datasets"] and includes the following columns:

- `Diabetes_012`: Indicates diabetes status (0: No Diabetes, 1: Prediabetes, 2: Diabetes)
- `PhysActivity`: Physical activity status (0: No, 1: Yes)
- `Age`: Age of the individuals
- `Income`: Income level of the individuals
- `BMI`: Body Mass Index of the individuals
- `MentHlth`: Number of days with poor mental health
- `Sex`: Gender (0: Female, 1: Male)
- `Fruits`, `Veggies`: Consumption of fruits and vegetables
- `PhysHlth`: Number of days with poor physical health

## Project Structure

- `load_data.py`: Script for loading the diabetes dataset.
- `eda.py`: Script for performing exploratory data analysis.
- `visual.py`: Script for generating visualizations.
- `notebooks/Diabetes_EDA.ipynb`: Jupyter Notebook containing the exploratory data analysis and visualizations.

## Visualizations

The project showcases the following representations:

1. The connection, between activity and the development of diabetes.
2. Age distribution by diabetes status.
3. The average BMI according to income level and diabetes status.
4. Distribution of mental health days by diabetes status and gender.
5. The influence of diet and income, on ones well being based on their diabetes status.

## How to Run the Project

1. Clone the repository.
2. Install the required dependencies listed in the `requirements.txt` (if you created one).
3. Run the scripts in the following order:
   - `load_data.py`
   - `eda.py`
   - `visual.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Victor Ruiz**  
Email: odinruiz52@yahoo.com
