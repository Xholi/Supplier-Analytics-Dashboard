import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF  # Ensure you have installed fpdf

class SupplierAnalytics:
    def __init__(self, df):
        """Initialize with a copy of the dataset"""
        self.df = df.copy()
        self.supplier_segments = None
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Prepares data by adding necessary calculated fields
        """
        # Ensure necessary columns exist
        required_columns = ['ReceivedQty', 'CancelledQty', 'OrderQty', 'OrderDate', 'DueDate']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Convert dates to datetime, handling errors
        self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'], errors='coerce', dayfirst=True)
        self.df['DueDate'] = pd.to_datetime(self.df['DueDate'], errors='coerce', dayfirst=True)

        # Remove rows where dates are invalid
        self.df = self.df.dropna(subset=['OrderDate', 'DueDate'])

        # Ensure dates are within a reasonable range (e.g., 2010-01-01 to today)
        min_valid_date = pd.Timestamp("2010-01-01")
        max_valid_date = pd.Timestamp.today()
        self.df = self.df[
            (self.df['OrderDate'] >= min_valid_date) & (self.df['OrderDate'] <= max_valid_date) &
            (self.df['DueDate'] >= min_valid_date) & (self.df['DueDate'] <= max_valid_date)
        ]

        # Calculate "DeliveryLeadTime"
        self.df['DeliveryLeadTime'] = (self.df['DueDate'] - self.df['OrderDate']).dt.days

        # Ensure no negative lead times
        self.df = self.df[self.df['DeliveryLeadTime'] >= 0]

        # Create "DeliveryStatus" based on given conditions
        self.df['DeliveryStatus'] = np.select(
            [
                self.df['ReceivedQty'] + self.df['CancelledQty'] == self.df['OrderQty'],  # Fully delivered
                self.df['ReceivedQty'] >= 0.75 * self.df['OrderQty'],  # 75% or more fulfilled
                self.df['ReceivedQty'] >= 0.50 * self.df['OrderQty'],  # 50% or more fulfilled
            ],
            [
                "Delivered",
                "Partially Delivered",
                "Half Filled"
            ],
            default="Pending Delivery"
        )

        # Calculate Perfect Order Rate (POR)
        self.df['PerfectOrder'] = (
            (self.df['DeliveryStatus'] == 'Delivered') & 
            (self.df['ReceivedQty'] == self.df['OrderQty']) & 
            (self.df['OrderDate'] + pd.to_timedelta(self.df['DeliveryLeadTime'], unit='D') <= self.df['DueDate'])
        ).astype(int)

        self.df['PerfectOrderRate'] = self.df.groupby('SupCode')['PerfectOrder'].transform(lambda x: x.mean() * 100)

        # Additional KPIs
        self.df['OTIF'] = ((self.df['DeliveryStatus'] == 'Delivered') &
                          (self.df['ReceivedQty'] == self.df['OrderQty'])).astype(int) * 100

        self.df['SupplierDefectRate'] = ((self.df['OrderQty'] - self.df['ReceivedQty']) / 
                                        self.df['OrderQty'] * 100)

        self.df['CostVariance'] = self.df.groupby('Code')['Cost'].transform(
            lambda x: (x - x.mean()) / x.mean() * 100
        )

        self.df['LeadTimeReliability'] = self.df.groupby('SupCode')['DeliveryLeadTime'].transform(
            lambda x: 1 - (x.std() / x.mean())
        ) * 100

        # Standard Cost & Purchase Price Variance
        self.df['StandardCost'] = self.df.groupby('Code')['Cost'].transform('mean')
        self.df['PPV'] = (self.df['StandardCost'] - self.df['Cost']) * self.df['ReceivedQty']

        # Days Payable Outstanding (assuming 30-day payment terms)
        self.df['DPO'] = 30
    def display_analysis(self):
        """Display supplier analysis, metrics, and visualizations in a Streamlit app."""
        st.title("Supplier Performance Dashboard")

        supplier_codes = ['All Suppliers'] + self.df['SupCode'].unique().tolist()
        supplier_code = st.selectbox("Select Supplier", supplier_codes, index=0)
        selected_data = self.df if supplier_code == 'All Suppliers' else self.df[self.df['SupCode'] == supplier_code]

        metrics = self.calculate_advanced_metrics(supplier_code if supplier_code != 'All Suppliers' else None)
        statistical_analysis = self.perform_statistical_analysis(supplier_code if supplier_code != 'All Suppliers' else None)
        visualizations = self.generate_visualizations(supplier_code if supplier_code != 'All Suppliers' else None)

        st.subheader("Performance Metrics")
        for category, category_metrics in metrics.items():
            st.markdown(f"### {category.title()}")
            for metric, value in category_metrics.items():
                st.write(f"**{metric.replace('_', ' ').title()}**: {value:.2f}")

        st.subheader("Visualizations")
        for name, fig in visualizations.items():
            st.markdown(f"### {name.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Dictionary")
        data_dict = {
            "OTIF": "On-Time In-Full Delivery Percentage",
            "PerfectOrderRate": "Percentage of orders delivered perfectly",
            "SupplierDefectRate": "Percentage of defective supplier deliveries",
            "CostVariance": "Difference between standard and actual cost",
            "LeadTimeReliability": "Supplier consistency in lead times"
        }
        st.table(pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description']))
            
    def calculate_advanced_metrics(self, supplier_code=None):
        """
        Calculate advanced industry-standard KPIs
        """
        if supplier_code:
            data = self.df[self.df['SupCode'] == supplier_code]
        else:
            data = self.df
            
        metrics = {
            'supplier_performance': {
                'OTIF': data['OTIF'].mean(),
                'perfect_order_rate': data['PerfectOrderRate'].mean(),
                'quality_rating': 100 - data['SupplierDefectRate'].mean(),
                'lead_time_reliability': data['LeadTimeReliability'].mean(),
                'cost_competitiveness': self._calculate_cost_competitiveness(data),
                'innovation_index': self._calculate_innovation_index(data),
                'relationship_strength': self._calculate_relationship_strength(data)
            },
            'financial_metrics': {
                'total_spend': data['TotPoVal2'].sum(),
                'cost_savings': data['PPV'].sum(),
                'spend_velocity': self._calculate_spend_velocity(data),
                'payment_efficiency': data['DPO'].mean(),
                'cost_avoidance': self._calculate_cost_avoidance(data)
            },
            'risk_metrics': {
                'supply_risk_index': self._calculate_supply_risk_index(data),
                'financial_risk_score': self._calculate_financial_risk(data),
                'concentration_risk': self._calculate_concentration_risk(data),
                'compliance_score': self._calculate_compliance_score(data)
            }
        }
        
        return metrics
    
    def _calculate_cost_competitiveness(self, data):
        """Calculate cost competitiveness score"""
        market_benchmark = data.groupby('Code')['Cost'].transform('mean')
        cost_ratio = data['Cost'] / market_benchmark
        return (1 - (cost_ratio.mean() - 1)) * 100
    
    def _calculate_innovation_index(self, data):
        """Calculate supplier innovation index"""
        # Simplified version - would normally include factors like:
        # - New product introduction rate
        # - Process improvement suggestions
        # - R&D collaboration
        return data['CostVariance'].abs().mean()
    
    def _calculate_relationship_strength(self, data):
        """Calculate relationship strength index"""
        factors = {
            'longevity': len(data['OrderDate'].dt.to_period('Y').unique()),
            'engagement': data['TotPoVal2'].sum() / self.df['TotPoVal2'].sum(),
            'reliability': data['OTIF'].mean() / 100
        }
        return np.mean(list(factors.values())) * 100
    
    def perform_statistical_analysis(self, supplier_code=None):
        """
        Perform advanced statistical analysis
        """
        if supplier_code:
            data = self.df[self.df['SupCode'] == supplier_code]
        else:
            data = self.df
            
        analysis = {
            'trend_analysis': self._perform_trend_analysis(data),
            'seasonality': self._analyze_seasonality(data),
            'correlation_analysis': self._perform_correlation_analysis(data),
            'outlier_analysis': self._perform_outlier_analysis(data),
            'hypothesis_tests': self._perform_hypothesis_tests(data)
        }
        
        return analysis
    
    def _perform_trend_analysis(self, data):
        """Analyze trends in key metrics"""
        monthly_metrics = data.groupby(data['OrderDate'].dt.to_period('M')).agg({
            'TotPoVal2': 'sum',
            'OTIF': 'mean',
            'SupplierDefectRate': 'mean'
        })
        
        trends = {}
        for column in monthly_metrics.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(monthly_metrics)),
                monthly_metrics[column].values
            )
            trends[column] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value
            }
        
        return trends
    
    def _analyze_seasonality(self, data):
        """Analyze seasonality in ordering patterns"""
        monthly_orders = data.groupby(data['OrderDate'].dt.to_period('M'))['TotPoVal2'].sum()

        if len(monthly_orders) < 24:
            return {"message": "Not enough data for seasonal analysis. Showing trend instead."}

        decomposition = seasonal_decompose(
            monthly_orders,
            period=12,
            extrapolate_trend='freq'
        )

        return {
            'seasonal_pattern': decomposition.seasonal.tolist(),
            'trend': decomposition.trend.tolist(),
            'residual': decomposition.resid.tolist()
        }
    
    def generate_visualizations(self, supplier_code=None):
        """
        Generate comprehensive visualizations and return them as a dictionary.
        """
        if supplier_code:
            data = self.df[self.df['SupCode'] == supplier_code]
        else:
            data = self.df
            
        figs = {}

        # 1. Performance Radar Chart
        fig = go.Figure()
        metrics = self.calculate_advanced_metrics(supplier_code)

        fig.add_trace(go.Scatterpolar(
            r=[
                metrics['supplier_performance']['OTIF'],
                metrics['supplier_performance']['quality_rating'],
                metrics['supplier_performance']['lead_time_reliability'],
                metrics['supplier_performance']['cost_competitiveness'],
                metrics['supplier_performance']['relationship_strength']
            ],
            theta=['OTIF', 'Quality', 'Reliability', 'Cost', 'Relationship'],
            fill='toself',
            name='Performance Metrics'
        ))
        
        figs['performance_radar'] = fig

        # 2. Trend Analysis Chart
        fig_trend = go.Figure()
        monthly_trends = data.groupby(data['OrderDate'].dt.to_period('M')).agg({
            'TotPoVal2': 'sum',
            'OTIF': 'mean',
            'SupplierDefectRate': 'mean'
        }).reset_index()

        fig_trend.add_trace(go.Scatter(
            x=monthly_trends['OrderDate'].astype(str),
            y=monthly_trends['TotPoVal2'],
            name='Order Value'
        ))

        fig_trend.add_trace(go.Scatter(
            x=monthly_trends['OrderDate'].astype(str),
            y=monthly_trends['OTIF'],
            name='OTIF',
            yaxis='y2'
        ))
        figs['trend_analysis'] = fig_trend

        # 3. Risk Heatmap (if risk metrics are available)
        risk_metrics = self.perform_risk_analysis(supplier_code)
        if 'volatility' in risk_metrics and isinstance(risk_metrics['volatility'], dict):
            risk_matrix = pd.DataFrame([risk_metrics['volatility']])  # Ensure it's a row, not a column

            fig_risk = px.imshow(
                risk_matrix,
                labels=dict(x='Risk Factors', y='Metric', color='Risk Score'),
                title='Supplier Risk Heatmap'
            )
            figs['risk_heatmap'] = fig_risk

        return figs

    def _perform_correlation_analysis(self, data):
        """
        Perform correlation analysis on key supplier metrics.
        Returns a correlation matrix.
        """
        # Select numeric columns relevant to supplier performance
        correlation_columns = ['TotPoVal2', 'OTIF', 'SupplierDefectRate', 
                            'LeadTimeReliability', 'PPV', 'CostVariance']
        
        # Ensure only existing columns are used
        correlation_columns = [col for col in correlation_columns if col in data.columns]
        
        if not correlation_columns:
            raise ValueError("No valid numerical columns found for correlation analysis.")

        # Compute correlation matrix
        correlation_matrix = data[correlation_columns].corr()

        # Optional: Visualize the correlation matrix using seaborn (for debugging)
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Supplier Performance Correlation Matrix")
        plt.show()

        return correlation_matrix

    def generate_automated_report(self, supplier_code=None, output_path='supplier_report.pdf'):
        """
        Generate comprehensive PDF report
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Supplier Performance Report', 0, 1, 'C')
        
        # Basic Information
        pdf.set_font('Arial', '', 12)
        if supplier_code:
            supplier_name = self.df[self.df['SupCode'] == supplier_code]['SupName'].iloc[0]
            pdf.cell(0, 10, f'Supplier: {supplier_name} ({supplier_code})', 0, 1)
        
        # Performance Metrics
        metrics = self.calculate_advanced_metrics(supplier_code)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Key Performance Indicators', 0, 1)
        
        pdf.set_font('Arial', '', 12)
        for category, category_metrics in metrics.items():
            pdf.cell(0, 10, f'\n{category.title()}:', 0, 1)
            for metric, value in category_metrics.items():
                if isinstance(value, (int, float)):
                    pdf.cell(0, 10, f'{metric.replace("_", " ").title()}: {value:.2f}', 0, 1)
        
        # Save visualizations to the report
        figs = self.generate_visualizations(supplier_code)
        for name, fig in figs.items():
            fig.write_image(f'temp_{name}.png')
            pdf.image(f'temp_{name}.png', x=10, y=None, w=190)
            pdf.add_page()
        
        # Recommendations
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Recommendations', 0, 1)
        
        recommendations = self.generate_strategic_recommendations()
        if supplier_code:
            recommendations = recommendations[recommendations['supplier_code'] == supplier_code]
        
        pdf.set_font('Arial', '', 12)
        for _, rec in recommendations.iterrows():
            pdf.multi_cell(0, 10, f'- {rec["recommendation"]}')
        
        pdf.output(output_path)
        return output_path

    def _calculate_spend_velocity(self, data):
        """Calculate spend velocity (rate of spend over time)"""
        time_range = (data['OrderDate'].max() - data['OrderDate'].min()).days / 365
        return data['TotPoVal2'].sum() / time_range if time_range > 0 else 0
    
    def _calculate_cost_avoidance(self, data):
        """Calculate cost avoidance through negotiation and process improvement"""
        return data['PPV'][data['PPV'] > 0].sum()
    
    def _calculate_supply_risk_index(self, data):
        """Calculate comprehensive supply risk index"""
        factors = {
            'delivery_risk': 1 - (data['OTIF'].mean() / 100),
            'quality_risk': data['SupplierDefectRate'].mean() / 100,
            'financial_risk': self._calculate_financial_risk(data),
            'concentration_risk': self._calculate_concentration_risk(data)
        }
        weights = {'delivery_risk': 0.3, 'quality_risk': 0.3, 
                  'financial_risk': 0.2, 'concentration_risk': 0.2}
        
        return sum(score * weights[factor] for factor, score in factors.items()) * 100
    
    def _calculate_financial_risk(self, data):
        """Calculate financial risk score"""
        # Simplified version - would normally include:
        # - Financial stability indicators
        # - Payment history
        # - Credit rating
        return (data['CostVariance'].std() / 100)
    
    def _calculate_concentration_risk(self, data):
        """Calculate supply concentration risk"""
        total_spend = self.df['TotPoVal2'].sum()
        supplier_spend = data['TotPoVal2'].sum()
        return supplier_spend / total_spend if total_spend > 0 else 0
    
    def _calculate_compliance_score(self, data):
        """Calculate supplier compliance score"""
        # Simplified version - would normally include:
        # - Documentation compliance
        # - Regulatory compliance
        # - Quality certifications
        compliance_factors = {
            'delivery_compliance': (data['OTIF'].mean() / 100),
            'quality_compliance': (1 - data['SupplierDefectRate'].mean() / 100),
            'cost_compliance': (1 - abs(data['CostVariance']).mean() / 100)
        }
        return np.mean(list(compliance_factors.values())) * 100

    def _perform_outlier_analysis(self, data):
        """
        Identify outliers in key supplier performance metrics using the IQR method.
        Returns a dictionary with detected outliers.
        """
        # Select numeric columns for outlier detection
        outlier_columns = ['TotPoVal2', 'OTIF', 'SupplierDefectRate', 
                            'LeadTimeReliability', 'PPV', 'CostVariance']

        # Ensure only existing columns are used
        outlier_columns = [col for col in outlier_columns if col in data.columns]

        if not outlier_columns:
            raise ValueError("No valid numerical columns found for outlier analysis.")

        outliers = {}

        for col in outlier_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_values = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            
            outliers[col] = outlier_values.tolist()

        return outliers
    
    def _perform_hypothesis_tests(self, data):
        """
        Perform hypothesis testing on key supplier performance metrics.
        Returns a dictionary of test results.
        """
        test_results = {}

        # Ensure there are enough data points
        if data.shape[0] < 2:
            return {"error": "Not enough data for hypothesis testing"}

        # Compare OTIF between two supplier groups using a T-test
        high_performance = data[data['OTIF'] >= data['OTIF'].median()]['OTIF']
        low_performance = data[data['OTIF'] < data['OTIF'].median()]['OTIF']

        if len(high_performance) > 1 and len(low_performance) > 1:
            t_stat, p_value = stats.ttest_ind(high_performance, low_performance, equal_var=False)
            test_results["OTIF_TTest"] = {"t_statistic": t_stat, "p_value": p_value}
        
        # One-Way ANOVA to compare supplier cost variance across different suppliers
        if "SupCode" in data.columns:
            supplier_groups = [group["CostVariance"].dropna() for _, group in data.groupby("SupCode") if len(group) > 1]
            if len(supplier_groups) > 1:
                f_stat, p_value = stats.f_oneway(*supplier_groups)
                test_results["CostVariance_ANOVA"] = {"f_statistic": f_stat, "p_value": p_value}
        
        return test_results
    
    def perform_risk_analysis(self, supplier_code=None):
        """
        Perform risk analysis on suppliers and return a dictionary of risk scores.
        """
        if supplier_code:
            data = self.df[self.df['SupCode'] == supplier_code]
        else:
            data = self.df

        risk_metrics = {}

        # Supply Risk Index: (Delivery Risk + Quality Risk)
        risk_metrics["supply_risk_index"] = (1 - (data['OTIF'].mean() / 100)) + (data['SupplierDefectRate'].mean() / 100)

        # Financial Risk Score: Based on cost variance
        risk_metrics["financial_risk_score"] = data['CostVariance'].std() / 100

        # Concentration Risk: Supplier dependency on a single client
        total_spend = self.df['TotPoVal2'].sum()
        supplier_spend = data['TotPoVal2'].sum()
        risk_metrics["concentration_risk"] = supplier_spend / total_spend if total_spend > 0 else 0

        # Compliance Score: Delivery & quality compliance
        compliance_factors = {
            'delivery_compliance': (data['OTIF'].mean() / 100),
            'quality_compliance': (1 - data['SupplierDefectRate'].mean() / 100),
            'cost_compliance': (1 - abs(data['CostVariance']).mean() / 100)
        }
        risk_metrics["compliance_score"] = np.mean(list(compliance_factors.values())) * 100

        # Volatility Index: Measures performance fluctuations
        volatility = {
            'OTIF': data['OTIF'].std(),
            'SupplierDefectRate': data['SupplierDefectRate'].std(),
            'TotPoVal2': data['TotPoVal2'].std(),
            'LeadTimeReliability': data['LeadTimeReliability'].std()
        }

        risk_metrics["volatility"] = volatility

        return risk_metrics
    

    def generate_interactive_report(self):
        """
        Display an interactive report where the user can select a supplier and view their report.
        """
        # Get unique supplier codes
        supplier_codes = self.df['SupCode'].unique().tolist()
        
        # Dropdown for selecting supplier
        supplier_dropdown = widgets.Dropdown(
            options=supplier_codes,
            description='Supplier:',
            disabled=False,
        )

        # Function to display the report when a supplier is selected
        def display_report(supplier_code):
            report_html = self.generate_automated_report(supplier_code)
            display(HTML(report_html))
        
        # Update function for dropdown
        widgets.interactive(display_report, supplier_code=supplier_dropdown)

def main():
    st.set_page_config(page_title="Advanced Supplier Analytics Dashboard", layout="wide")
    
    st.title("📊 Advanced Supplier Analytics Dashboard")
    
    # Add some top spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your supplier data (Excel file)", type=['xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        analytics = SupplierAnalytics(df)

        st.sidebar.header("Supplier Selection")
        supplier_search = st.sidebar.text_input("Search Supplier (Code or Name)")
        
        suppliers = df[['SupCode', 'SupName']].drop_duplicates()
        if supplier_search:
            suppliers = suppliers[
                suppliers['SupCode'].str.contains(supplier_search, case=False, na=False) |
                suppliers['SupName'].str.contains(supplier_search, case=False, na=False)
            ]
        
        selected_supplier = st.sidebar.selectbox(
            "Select Supplier",
            suppliers['SupCode'].tolist(),
            format_func=lambda x: f"{x} - {suppliers[suppliers['SupCode'] == x]['SupName'].values[0] if not suppliers[suppliers['SupCode'] == x].empty else 'Unknown Supplier'}"
        )
        
        if selected_supplier:
            # Get all metrics and analyses
            metrics = analytics.calculate_advanced_metrics(selected_supplier)
            statistical_analysis = analytics.perform_statistical_analysis(selected_supplier)
            risk_analysis = analytics.perform_risk_analysis(selected_supplier)
            visualizations = analytics.generate_visualizations(selected_supplier)
            
            # Display supplier name
            st.header(f"Supplier: {suppliers[suppliers['SupCode']==selected_supplier]['SupName'].iloc[0]}")
            
            # Create tabs for different sections with adequate spacing using the gap parameter
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Financial Analysis", "Risk Assessment", "Statistical Analysis"])
            
            with tab1:
                # Create three columns with larger gaps between them
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.subheader("Key Performance Indicators")
                    for metric, value in metrics['supplier_performance'].items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(visualizations['performance_radar'], use_container_width=True)
                
                with col3:
                    st.subheader("Delivery Status")
                    supplier_data = analytics.df[analytics.df['SupCode'] == selected_supplier]
                    pie_chart = px.pie(supplier_data, names='DeliveryStatus', title="Delivery Status Distribution")
                    st.plotly_chart(pie_chart, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.subheader("Financial Metrics")
                    for metric, value in metrics['financial_metrics'].items():
                        if 'total' in metric or 'savings' in metric:
                            st.metric(metric.replace('_', ' ').title(), f"R{value:,.2f}")
                        else:
                            st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(visualizations['trend_analysis'], use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.subheader("Risk Metrics")
                    for metric, value in risk_analysis.items():
                        if isinstance(value, dict):
                            st.markdown("#### Volatility Metrics")
                            for k, v in value.items():
                                st.metric(k, f"{v:.2f}")
                        else:
                            st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                
                with col2:
                    # Display risk heatmap if available
                    if 'risk_heatmap' in visualizations:
                        st.plotly_chart(visualizations['risk_heatmap'], use_container_width=True)
            
            with tab4:
                st.subheader("Statistical Analysis")
                
                st.markdown("#### Trend Analysis")
                trend_df = pd.DataFrame(statistical_analysis['trend_analysis']).T
                st.dataframe(trend_df)
                
                st.markdown("#### Correlation Analysis")
                corr_matrix = statistical_analysis['correlation_analysis']
                st.plotly_chart(px.imshow(corr_matrix, color_continuous_scale='Viridis'), use_container_width=True)
                
                st.markdown("#### Recent Orders")
                supplier_data = analytics.df[analytics.df['SupCode'] == selected_supplier]
                recent_orders = supplier_data.sort_values('OrderDate', ascending=False).head(10)
                st.dataframe(
                    recent_orders[['OrderDate', 'OrderQty', 'ReceivedQty','CancelledQty', 'TotPoVal2', 'DeliveryStatus', 'PerfectOrder']]
                )

            data_dict = {
             "OTIF": "On-Time In-Full Delivery Percentage",
            "PerfectOrderRate": "Percentage of orders delivered perfectly",
            "SupplierDefectRate": "Percentage of defective supplier deliveries",
            "CostVariance": "Difference between standard and actual cost",
            "LeadTimeReliability": "Supplier consistency in lead times"
        }
        st.write(pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description']))


    else:
        # Load pre-existing dataset (Modify path to your default file)
        df = pd.read_excel("C:/Users/Xholisile Mantshongo/Downloads/wetransfer_currentreports_2025-01-28_1754/CurrentReports/AllPO'stoSuppliers.xlsx")  
        # Sidebar for supplier selection
        analytics = SupplierAnalytics(df)
        st.sidebar.header("Supplier Selection")
        supplier_search = st.sidebar.text_input("Search Supplier (Code or Name)")
        
        suppliers = df[['SupCode', 'SupName']].drop_duplicates()
        if supplier_search:
            suppliers = suppliers[
                suppliers['SupCode'].str.contains(supplier_search, case=False, na=False) |
                suppliers['SupName'].str.contains(supplier_search, case=False, na=False)
            ]
        
        selected_supplier = st.sidebar.selectbox(
            "Select Supplier",
            suppliers['SupCode'].tolist(),
            format_func=lambda x: f"{x} - {suppliers[suppliers['SupCode'] == x]['SupName'].values[0] if not suppliers[suppliers['SupCode'] == x].empty else 'Unknown Supplier'}"
        )
        
        if selected_supplier:
            # Get all metrics and analyses
            metrics = analytics.calculate_advanced_metrics(selected_supplier)
            statistical_analysis = analytics.perform_statistical_analysis(selected_supplier)
            risk_analysis = analytics.perform_risk_analysis(selected_supplier)
            visualizations = analytics.generate_visualizations(selected_supplier)
            
            # Display supplier name
            st.header(f"Supplier: {suppliers[suppliers['SupCode']==selected_supplier]['SupName'].iloc[0]}")
            
            # Create tabs for different sections with adequate spacing using the gap parameter
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Financial Analysis", "Risk Assessment", "Statistical Analysis"])
            
            with tab1:
                # Create three columns with larger gaps between them
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.subheader("Key Performance Indicators")
                    for metric, value in metrics['supplier_performance'].items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(visualizations['performance_radar'], use_container_width=True)
                
                with col3:
                    st.subheader("Delivery Status")
                    supplier_data = analytics.df[analytics.df['SupCode'] == selected_supplier]
                    pie_chart = px.pie(supplier_data, names='DeliveryStatus', title="Delivery Status Distribution")
                    st.plotly_chart(pie_chart, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.subheader("Financial Metrics")
                    for metric, value in metrics['financial_metrics'].items():
                        if 'total' in metric or 'savings' in metric:
                            st.metric(metric.replace('_', ' ').title(), f"R{value:,.2f}")
                        else:
                            st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                    st.markdown("<br>", unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(visualizations['trend_analysis'], use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.subheader("Risk Metrics")
                    for metric, value in risk_analysis.items():
                        if isinstance(value, dict):
                            st.markdown("#### Volatility Metrics")
                            for k, v in value.items():
                                st.metric(k, f"{v:.2f}")
                        else:
                            st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")
                
                with col2:
                    # Display risk heatmap if available
                    if 'risk_heatmap' in visualizations:
                        st.plotly_chart(visualizations['risk_heatmap'], use_container_width=True)
            
            with tab4:
                st.subheader("Statistical Analysis")
                
                st.markdown("#### Trend Analysis")
                trend_df = pd.DataFrame(statistical_analysis['trend_analysis']).T
                st.dataframe(trend_df)
                
                st.markdown("#### Correlation Analysis")
                corr_matrix = statistical_analysis['correlation_analysis']
                st.plotly_chart(px.imshow(corr_matrix, color_continuous_scale='Viridis'), use_container_width=True)
                
                st.markdown("#### Recent Orders")
                supplier_data = analytics.df[analytics.df['SupCode'] == selected_supplier]
                recent_orders = supplier_data.sort_values('OrderDate', ascending=False).head(10)
                st.dataframe(
                    recent_orders[['OrderDate', 'OrderQty', 'ReceivedQty','CancelledQty', 'TotPoVal2', 'DeliveryStatus', 'PerfectOrder']]
                )
        data_dict = {
             "OTIF": "On-Time In-Full Delivery Percentage",
            "PerfectOrderRate": "Percentage of orders delivered perfectly",
            "SupplierDefectRate": "Percentage of defective supplier deliveries",
            "CostVariance": "Difference between standard and actual cost",
            "LeadTimeReliability": "Supplier consistency in lead times"
        }
        st.write(pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description']))


if __name__ == "__main__":
    main()
