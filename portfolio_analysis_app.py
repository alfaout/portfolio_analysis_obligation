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
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        # Find date and price columns
        date_cols = ['Date', 'date', 'Séance', 'Trading Date']
        price_cols = ['Close', 'Price', 'Cours', 'Cours ajusté', 'Adjusted Close']
        
        # Identify date column
        date_col = next((col for col in date_cols if col in df.columns), df.columns[0])
        
        # Identify price column
        price_col = next((col for col in price_cols if col in df.columns), df.columns[1])
        
        # Create DataFrame with date and price
        processed_df = pd.DataFrame({
            'Date': pd.to_datetime(df[date_col], errors='coerce'),
            'Price': pd.to_numeric(df[price_col], errors='coerce')
        })
        
        # Drop rows with NaN values
        processed_df.dropna(inplace=True)
        
        # Rename price column to ticker
        processed_df.rename(columns={'Price': file_path.split('/')[-1].split('.')[0]}, inplace=True)
        
        # Set index and sort
        processed_df.set_index('Date', inplace=True)
        processed_df.sort_index(inplace=True)
        
        return processed_df[~processed_df.index.duplicated(keep='first')]
    
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
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
    
    # Load and process individual datasets
    processed_data = {}
    for ticker, file_path in file_config.items():
        try:
            df = load_and_preprocess_data(file_path)
            if df is not None and not df.empty:
                processed_data[ticker] = df
            else:
                st.sidebar.warning(f"No valid data for {ticker}")
        except Exception as e:
            st.sidebar.error(f"Error processing {ticker}: {e}")
    
    # Multiselect for companies
    available_companies = list(processed_data.keys())
    
    # Ensure at least some companies are available
    if not available_companies:
        st.error("No valid data found in any of the uploaded files.")
        return
    
    # --- ONLY CALL TO MULTISELECT WIDGET (INPUT) ---
    selected_companies = st.sidebar.multiselect(
        "Select Companies", 
        available_companies, 
        default=[],
        key="portfolio_analysis_company_selection"
    )
    
    if len(selected_companies) < 2:
        st.warning("Please select at least 2 companies for portfolio analysis.")
        return
    
    # Prepare returns data
    try:
        # Debug: Print selected companies
        st.write("Selected Companies:", selected_companies)
        
        # Prepare data for selected companies
        selected_data = pd.concat([processed_data[ticker] for ticker in selected_companies], axis=1)
        
        # Debug: Print selected data info
        st.write("Selected Data Shape:", selected_data.shape)
        st.write("Selected Data Columns:", list(selected_data.columns))
        
        # Additional check to ensure numeric data
        for col in selected_data.columns:
            selected_data[col] = pd.to_numeric(selected_data[col], errors="coerce")
        
        # Drop any rows that became NaN after conversion
        selected_data.dropna(inplace=True)
        
        # Debug: Print data after cleaning
        st.write("Cleaned Data Shape:", selected_data.shape)
        
        # Calculate returns
        returns = selected_data.pct_change().dropna()
        
        # Debug: Print returns info
        st.write("Returns Shape:", returns.shape)
        st.write("Returns Columns:", list(returns.columns))
        
        # Ensure returns are valid
        if returns.empty:
            st.error("Not enough valid data to calculate returns. Please check your data.")
            return
        
        # Perform portfolio optimization
        results = portfolio_optimization(returns)
        
        # Get mean returns and volatilities for individual assets
        mean_ret = returns.mean() * 252
        vol_ind = returns.std() * np.sqrt(252)
        
        # Results Section
        st.header("📊 Portfolio Analysis Results")
        
        # Portfolio Performance Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Sharpe Ratio Portfolio")
            st.markdown(f"**Return:** {results['best_sharpe_portfolio']['return']:.2%}")
            st.markdown(f"**Risk:** {results['best_sharpe_portfolio']['risk']:.2%}")
            st.markdown(f"**Sharpe Ratio:** {results['best_sharpe_portfolio']['sharpe_ratio']:.2f}")
            
            st.markdown("**Weights:**")
            for asset, weight in results['best_sharpe_portfolio']['weights'].items():
                st.markdown(f"- {asset}: {weight:.2%}")

            # Interpretation
            st.markdown("""
            <div class="interpretation">
            🔍 **Interpretation:**
            - Higher Sharpe Ratio indicates better risk-adjusted return
            - This portfolio maximizes return per unit of risk
            - Ideal for investors seeking optimal performance
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🛡️ Minimum Risk Portfolio")
            st.markdown(f"**Return:** {results['min_risk_portfolio']['return']:.2%}")
            st.markdown(f"**Risk:** {results['min_risk_portfolio']['risk']:.2%}")
            st.markdown(f"**Sharpe Ratio:** {results['min_risk_portfolio']['sharpe_ratio']:.2f}")
            
            st.markdown("**Weights:**")
            for asset, weight in results['min_risk_portfolio']['weights'].items():
                st.markdown(f"- {asset}: {weight:.2%}")

            # Interpretation
            st.markdown("""
            <div class="interpretation">
            🔍 **Interpretation:**
            - Lowest possible portfolio volatility
            - Conservative strategy for risk-averse investors
            - Prioritizes capital preservation
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations Section
        st.header("🔍 Asset Performance Analysis")
        
        # Asset Price Evolution
        st.subheader("📈 Asset Price Trends")
        plt.figure(figsize=(14, 8))
        for column in selected_data.columns:
            plt.plot(selected_data.index, selected_data[column], label=column)
        plt.title("Historical Asset Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # Normalized Asset Performance
        st.subheader("📊 Normalized Asset Performance")
        normalized_data = selected_data / selected_data.iloc[0] * 100
        plt.figure(figsize=(14, 8))
        for column in normalized_data.columns:
            plt.plot(normalized_data.index, normalized_data[column], label=column)
        plt.title("Normalized Asset Performance (Starting Value = 100)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (%)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # Enhanced Efficient Frontier Visualization
        st.subheader("🎯 Advanced Efficient Frontier Analysis")
        plt.figure(figsize=(14, 10))
        
        # Scatter plot of individual assets
        plt.scatter(
            vol_ind, 
            mean_ret, 
            marker="o", 
            s=200, 
            alpha=0.7, 
            c=mean_ret/vol_ind,  # Color based on Sharpe ratio
            cmap="viridis",
            label="Individual Assets"
        )
        
        # Annotate individual assets
        for i, ticker in enumerate(returns.columns):
            plt.annotate(
                ticker, 
                (vol_ind[i], mean_ret[i]), 
                xytext=(10, 10),
                textcoords="offset points"
            )
        
        # Plot efficient frontier
        plt.scatter(
            results["results_array"][:, 1], 
            results["results_array"][:, 0], 
            c=results["results_array"][:, 2], 
            cmap="viridis", 
            alpha=0.3,
            label="Possible Portfolios"
        )
        
        # Highlight optimal portfolios
        plt.scatter(
            results["best_sharpe_portfolio"]["risk"], 
            results["best_sharpe_portfolio"]["return"], 
            color="red", 
            marker="*", 
            s=500, 
            label="Max Sharpe Portfolio"
        )
        plt.scatter(
            results["min_risk_portfolio"]["risk"], 
            results["min_risk_portfolio"]["return"], 
            color="green", 
            marker="*", 
            s=500, 
            label="Minimum Risk Portfolio"
        )
        
        plt.title("Advanced Efficient Frontier Analysis")
        plt.xlabel("Portfolio Risk (Volatility)")
        plt.ylabel("Expected Portfolio Return")
        plt.colorbar(label="Sharpe Ratio")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())

# Run the app
if __name__ == "__main__":
    main()
