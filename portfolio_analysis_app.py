import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Portfolio Analysis", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-container {
    display: flex;
    justify-content: space-around;
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.interpretation {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 Portfolio Optimization & Analysis")

# Load data function
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from Excel or CSV file
    """
    try:
        # Read file
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            # For CSV, try different date parsing strategies
            try:
                # Try parsing with US format first (m/d/Y)
                df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=False)
            except:
                try:
                    # Try parsing with European format (d/m/Y)
                    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
                except:
                    # Fallback to reading without date parsing
                    df = pd.read_csv(file_path)
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            raise ValueError("Unsupported file type")
        
        # Identify price column (use Obligation_5Y for obligation data)
        price_columns = ["Close", "Price", "Cours", "Cours ajusté", "Adjusted Close", "Obligation_5Y"]
        price_col = next((col for col in price_columns if col in df.columns), df.columns[1])
        
        # Create processed DataFrame
        processed_df = pd.DataFrame({
            "Date": pd.to_datetime(df["Date"], errors="coerce"),
            "Price": pd.to_numeric(df[price_col], errors="coerce")
        })
        
        # Remove any rows with NaN values
        processed_df.dropna(inplace=True)
        
        # Rename price column based on filename
        processed_df.rename(columns={"Price": file_path.split("/")[-1].split(".")[0]}, inplace=True)
        
        # Set Date as index and sort
        processed_df.set_index("Date", inplace=True)
        processed_df.sort_index(inplace=True)
        
        # Remove duplicate dates
        return processed_df[~processed_df.index.duplicated(keep="first")]
    
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# File paths
file_config = {
    'IAM': 'IAM.xlsx',
    'CIH': 'CIH.xlsx',
    'Apple': 'Apple_data_1an.xlsx',
    'Tesla': 'Tesla_data_1an.xlsx',
    'Gold': 'gold_futures_history.csv',
    'Obligation': 'obligation_5Y_sample.csv'
}

# Welcome and introduction
def welcome_section():
    st.markdown("""
    ## 🌟 Welcome to Portfolio Optimization & Analysis 

    This interactive tool helps you:
    - Select multiple financial assets
    - Analyze portfolio performance
    - Optimize investment strategies
    - Visualize risk and return characteristics

    ### How to Use:
    1. Select companies/assets from the sidebar
    2. Explore portfolio metrics
    3. Understand risk and return trade-offs
    """)
    
    st.info("👈 Use the sidebar to select your assets and start analyzing!")

# Portfolio optimization function
def portfolio_optimization(returns, trading_days=252):
    """
    Perform portfolio optimization
    """
    # Calculate key metrics
    mean_ret = returns.mean() * trading_days
    cov_mat = returns.cov() * trading_days
    corr_mat = returns.corr()
    vol_ind = returns.std() * np.sqrt(trading_days)
    
    # Monte Carlo simulation
    sim_results = []
    for _ in range(15000):
        w = np.random.random(len(returns.columns))
        w /= np.sum(w)
        p_ret = np.dot(w, mean_ret)
        p_var = np.dot(w.T, np.dot(cov_mat, w))
        p_std = np.sqrt(p_var)
        # Theoretical risk-free rate 3%
        p_sharpe = (p_ret - 0.03) / p_std
        sim_results.append((p_ret, p_std, p_sharpe, w))
    
    # Convert to numpy array for efficient processing
    results_array = np.array([x[:3] for x in sim_results])
    weights_array = np.array([x[3] for x in sim_results])
    
    # Find optimal portfolios
    idx_max_sharpe = results_array[:, 2].argmax()
    best_sharpe = results_array[idx_max_sharpe]
    best_weights = weights_array[idx_max_sharpe]
    
    idx_min_risk = results_array[:, 1].argmin()
    min_risk = results_array[idx_min_risk]
    min_risk_weights = weights_array[idx_min_risk]
    
    return {
        'mean_returns': mean_ret,
        'volatilities': vol_ind,
        'correlation_matrix': corr_mat,
        'best_sharpe_portfolio': {
            'return': best_sharpe[0],
            'risk': best_sharpe[1],
            'sharpe_ratio': best_sharpe[2],
            'weights': dict(zip(returns.columns, best_weights))
        },
        'min_risk_portfolio': {
            'return': min_risk[0],
            'risk': min_risk[1],
            'sharpe_ratio': min_risk[2],
            'weights': dict(zip(returns.columns, min_risk_weights))
        },
        'results_array': results_array
    }

# Main Streamlit app
def main():
    # Welcome section
    welcome_section()
    
    # Sidebar for company selection
    st.sidebar.header("🔍 Portfolio Configuration")
    
    # Load and process individual
