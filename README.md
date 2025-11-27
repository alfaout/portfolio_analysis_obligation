# Portfolio Analysis Web Application

## Overview
This Streamlit application provides comprehensive portfolio analysis and optimization, allowing users to:
- Select multiple financial assets
- Analyze portfolio performance
- Visualize efficient frontier
- View asset correlation matrices
- Explore historical price trends

## Features
- Interactive asset selection
- Portfolio optimization techniques
  - Maximum Sharpe Ratio Portfolio
  - Minimum Risk Portfolio
- Detailed portfolio metrics
- Efficient Frontier visualization
- Asset correlation heatmap
- Historical price evolution charts

## Prerequisites
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Local Deployment

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run portfolio_analysis_app.py
```

## Deployment Options

### Streamlit Cloud
1. Go to [Streamlit.io](https://streamlit.io/)
2. Connect your GitHub account
3. Select the repository
4. Choose the main branch
5. Deploy

### Heroku
1. Create a `Procfile`
2. Add Heroku configuration
3. Deploy using Heroku CLI

## Project Structure
```
project_folder/
│
├── portfolio_analysis_app.py
├── requirements.txt
│
└── data/
    ├── IAM.xlsx
    ├── CIH.xlsx
    ├── Apple_data_1an.xlsx
    ├── Tesla_data_1an.xlsx
    └── gold_futures_history.csv
```

## Data Requirements
- Place financial data files in the `data/` directory
- Supported formats: Excel (.xlsx) and CSV
- Data should include date and price columns
- Supported files: Stock prices, futures data, etc.

## Customization
- Modify `file_config` dictionary to add or remove assets
- Adjust risk-free rate in `portfolio_optimization` function
- Customize visualizations as needed

## Contributing
Contributions are welcome! Please submit pull requests.

## License
[Specify your license]
