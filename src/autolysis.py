
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
import logging
from typing import Dict, List, Any, Optional, Tuple
import re

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Set style for consistent, professional visualizations
plt.style.use('default')
sns.set_palette("husl")

def convert_numpy_types(obj):
    if isinstance(obj, (np.generic, pd.Int64Dtype)):
        return obj.item() if hasattr(obj, 'item') else int(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


class DataAnalyzer:
    """Comprehensive data analysis class with LLM integration."""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.analysis_results = {}
        self.visualizations = []
        self.api_key = os.environ.get("AIPROXY_TOKEN")
        if not self.api_key:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
        
        # AI API configuration
        self.api_base = "https://api.deepseek.com/v1"  # DeepSeek's official API endpoint
        self.model = "deepseek-chat"  # Using DeepSeek's chat model
        
    def load_data(self) -> pd.DataFrame:
        """Load and basic preprocessing of CSV data."""
        try:
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        self.df = pd.read_csv(self.csv_file, encoding=encoding, sep=sep)
                        if self.df.shape[1] > 1:  # Valid if more than 1 column
                            logging.info(f"Successfully loaded data with encoding: {encoding}, separator: '{sep}'")
                            return self.df
                    except:
                        continue
            
            # Fallback
            self.df = pd.read_csv(self.csv_file)
            logging.info("Loaded data with default settings")
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def basic_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive basic analysis of the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {k: str(v) for k, v in self.df.dtypes.to_dict().items()},
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
        }
        
        # Numeric columns analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            analysis['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
            analysis['correlations'] = self.df[numeric_cols].corr().to_dict()
        
        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            analysis['categorical_summary'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 to avoid token overuse
                analysis['categorical_summary'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head(5).to_dict()
                }
        
        # Sample data
        analysis['sample_data'] = self.df.head(3).to_dict('records')
        
        self.analysis_results['basic'] = analysis
        return analysis
    
    def advanced_analysis(self) -> Dict[str, Any]:
        """Perform advanced statistical analysis."""
        advanced = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Outlier detection using Isolation Forest
            try:
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = isolation_forest.fit_predict(self.df[numeric_cols].fillna(0))
                advanced['outliers_detected'] = int(np.sum(outlier_pred == -1))
                advanced['outlier_percentage'] = float(np.sum(outlier_pred == -1) / len(outlier_pred) * 100)
            except:
                advanced['outliers_detected'] = 0
        
        # Statistical tests for normality
        if numeric_cols:
            advanced['normality_tests'] = {}
            for col in numeric_cols[:3]:  # Test first 3 numeric columns
                try:
                    stat, p_value = stats.shapiro(self.df[col].dropna().sample(min(5000, len(self.df))))
                    advanced['normality_tests'][col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                except:
                    continue
        
        # Clustering analysis
        if len(numeric_cols) >= 2:
            try:
                # Prepare data for clustering
                cluster_data = self.df[numeric_cols].fillna(0)
                if len(cluster_data) > 1:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    
                    # Determine optimal clusters using elbow method
                    max_k = min(10, len(cluster_data) // 2)
                    if max_k >= 2:
                        inertias = []
                        k_range = range(2, max_k + 1)
                        for k in k_range:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(scaled_data)
                            inertias.append(kmeans.inertia_)
                        
                        # Use elbow method to find optimal k
                        optimal_k = 3 if len(k_range) >= 2 else 2
                        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_data)
                        
                        advanced['clustering'] = {
                            'optimal_clusters': optimal_k,
                            'cluster_sizes': np.bincount(cluster_labels).tolist()
                        }
            except Exception as e:
                logging.warning(f"Clustering analysis failed: {e}")
        
        self.analysis_results['advanced'] = advanced
        return advanced
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_llm(self, messages: List[Dict[str, str]], functions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Make API call to LLM with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        if functions:
            payload["functions"] = functions
            payload["function_call"] = "auto"
        
        with httpx.Client() as client:
            response = client.post(f"{self.api_base}/chat/completions", 
                                 headers=headers, 
                                 json=payload, 
                                 timeout=60.0)
            response.raise_for_status()
            return response.json()
    
    def get_llm_insights(self) -> Dict[str, Any]:
        """Get insights and analysis suggestions from LLM."""
        # Prepare concise data summary for LLM
        data_summary = {
            'filename': os.path.basename(self.csv_file),
            'shape': self.analysis_results['basic']['shape'],
            'columns': self.analysis_results['basic']['columns'][:10],  # Limit columns
            'dtypes': {k: str(v) for k, v in list(self.analysis_results['basic']['dtypes'].items())[:10]},
            'missing_data': {k: v for k, v in self.analysis_results['basic']['missing_values'].items() if v > 0},
            'sample': self.analysis_results['basic']['sample_data'][:2]
        }
        
        # Add numeric summary if available
        if 'numeric_summary' in self.analysis_results['basic']:
            numeric_summary = self.analysis_results['basic']['numeric_summary']
            data_summary['numeric_insights'] = {
                col: {
                    'mean': round(stats['mean'], 3) if not np.isnan(stats['mean']) else None,
                    'std': round(stats['std'], 3) if not np.isnan(stats['std']) else None,
                    'min': round(stats['min'], 3) if not np.isnan(stats['min']) else None,
                    'max': round(stats['max'], 3) if not np.isnan(stats['max']) else None
                }
                for col, stats in list(numeric_summary.items())[:5]
            }
        
        # Add advanced analysis results
        if 'advanced' in self.analysis_results:
            data_summary['advanced_insights'] = self.analysis_results['advanced']
        
        messages = [
            {
                "role": "system",
                "content": """You are a senior data analyst. Analyze the provided dataset summary and provide insights, patterns, and recommendations. Focus on:
1. Key patterns and relationships in the data
2. Potential data quality issues
3. Interesting insights and anomalies
4. Business implications and recommendations
5. Suggested additional analyses

Be concise but insightful. Provide actionable insights."""
            },
            {
                "role": "user",
                "content": f"Analyze this dataset summary and provide key insights:\n\n{json.dumps(convert_numpy_types(data_summary), indent=2)}"
            }
        ]
        
        try:
            response = self.call_llm(messages)
            insights = response['choices'][0]['message']['content']
            self.analysis_results['llm_insights'] = insights
            return {'insights': insights}
        except Exception as e:
            logging.error(f"Error getting LLM insights: {e}")
            return {'insights': 'Unable to generate LLM insights due to API error.'}
    
    def create_visualizations(self) -> List[str]:
        """Create comprehensive visualizations based on data characteristics."""
        viz_files = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Set consistent figure parameters
        plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
        
        # Visualization 1: Data Overview Dashboard
        if len(numeric_cols) > 0 or len(categorical_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Data Overview: {os.path.basename(self.csv_file)}', fontsize=16, fontweight='bold')
            
            # Missing values heatmap
            missing_data = self.df.isnull().sum()
            if missing_data.sum() > 0:
                missing_data[missing_data > 0].plot(kind='bar', ax=axes[0,0], color='coral')
                axes[0,0].set_title('Missing Values by Column')
                axes[0,0].set_ylabel('Count')
                axes[0,0].tick_params(axis='x', rotation=45)
            else:
                axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('Missing Values Status')
            
            # Data types distribution
            dtype_counts = self.df.dtypes.value_counts()
            axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0,1].set_title('Data Types Distribution')
            
            # Numeric columns distribution (if available)
            if numeric_cols:
                if len(numeric_cols) == 1:
                    self.df[numeric_cols[0]].hist(bins=30, ax=axes[1,0], alpha=0.7, color='skyblue')
                    axes[1,0].set_title(f'Distribution: {numeric_cols[0]}')
                else:
                    # Correlation heatmap for multiple numeric columns
                    corr_matrix = self.df[numeric_cols[:8]].corr()  # Limit to 8 columns
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                              ax=axes[1,0], fmt='.2f', square=True)
                    axes[1,0].set_title('Correlation Matrix')
            else:
                axes[1,0].text(0.5, 0.5, 'No Numeric Data', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Numeric Analysis')
            
            # Categorical analysis (if available)
            if categorical_cols:
                cat_col = categorical_cols[0]
                value_counts = self.df[cat_col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[1,1], color='lightgreen')
                axes[1,1].set_title(f'Top Categories: {cat_col}')
                axes[1,1].tick_params(axis='x', rotation=45)
            else:
                axes[1,1].text(0.5, 0.5, 'No Categorical Data', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Categorical Analysis')
            
            plt.tight_layout()
            viz_file = 'data_overview.png'
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            viz_files.append(viz_file)
        
        # Visualization 2: Advanced Analysis
        if len(numeric_cols) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Advanced Statistical Analysis', fontsize=16, fontweight='bold')
            
            # Scatter plot of two main numeric variables
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            scatter_data = self.df[[x_col, y_col]].dropna()
            
            if len(scatter_data) > 0:
                axes[0].scatter(scatter_data[x_col], scatter_data[y_col], alpha=0.6, s=50)
                axes[0].set_xlabel(x_col)
                axes[0].set_ylabel(y_col)
                axes[0].set_title(f'{x_col} vs {y_col}')
                
                # Add trend line if correlation exists
                try:
                    z = np.polyfit(scatter_data[x_col], scatter_data[y_col], 1)
                    p = np.poly1d(z)
                    axes[0].plot(scatter_data[x_col], p(scatter_data[x_col]), "r--", alpha=0.8)
                except:
                    pass
            
            # Box plots for outlier detection
            if len(numeric_cols) <= 5:
                box_data = [self.df[col].dropna() for col in numeric_cols]
                axes[1].boxplot(box_data, labels=numeric_cols)
                axes[1].set_title('Outlier Detection (Box Plots)')
                axes[1].tick_params(axis='x', rotation=45)
            else:
                # PCA visualization for high-dimensional data
                try:
                    pca_data = self.df[numeric_cols].fillna(0)
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_data)
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    axes[1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    axes[1].set_title('PCA - First Two Components')
                except Exception as e:
                    axes[1].text(0.5, 0.5, f'PCA Error: {str(e)[:50]}', ha='center', va='center', transform=axes[1].transAxes)
            
            plt.tight_layout()
            viz_file = 'advanced_analysis.png'
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            viz_files.append(viz_file)
        
        # Visualization 3: Domain-specific insights
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if len(numeric_cols) >= 1:
            # Time series plot if date column detected
            date_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month'])]
            
            if date_cols and len(numeric_cols) >= 1:
                try:
                    date_col = date_cols[0]
                    numeric_col = numeric_cols[0]
                    
                    # Try to parse date column
                    plot_data = self.df[[date_col, numeric_col]].dropna()
                    if len(plot_data) > 1:
                        plot_data[date_col] = pd.to_datetime(plot_data[date_col], errors='coerce')
                        plot_data = plot_data.dropna().sort_values(date_col)
                        
                        if len(plot_data) > 1:
                            ax.plot(plot_data[date_col], plot_data[numeric_col], marker='o', linewidth=2, markersize=4)
                            ax.set_xlabel(date_col)
                            ax.set_ylabel(numeric_col)
                            ax.set_title(f'Time Series: {numeric_col} over {date_col}')
                            plt.xticks(rotation=45)
                        else:
                            raise ValueError("Insufficient time series data")
                    else:
                        raise ValueError("No valid date-numeric pairs")
                        
                except Exception as e:
                    # Fallback: Distribution plot
                    main_numeric = numeric_cols[0]
                    self.df[main_numeric].hist(bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_xlabel(main_numeric)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution Analysis: {main_numeric}')
                    ax.grid(True, alpha=0.3)
            else:
                # Distribution with statistics
                main_numeric = numeric_cols[0]
                data = self.df[main_numeric].dropna()
                
                data.hist(bins=30, alpha=0.7, color='steelblue', edgecolor='black', ax=ax)
                
                # Add statistical lines
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                
                ax.set_xlabel(main_numeric)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Statistical Distribution: {main_numeric}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        elif categorical_cols:
            # Categorical analysis for non-numeric data
            cat_col = categorical_cols[0]
            value_counts = self.df[cat_col].value_counts().head(15)
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral', edgecolor='darkred')
            ax.set_xlabel(cat_col)
            ax.set_ylabel('Count')
            ax.set_title(f'Category Distribution: {cat_col}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                       str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        viz_file = 'insights_analysis.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        viz_files.append(viz_file)
        
        self.visualizations = viz_files
        return viz_files
    
    def generate_narrative(self) -> str:
        """Generate comprehensive narrative using LLM."""
        # Prepare comprehensive summary for narrative generation
        narrative_data = {
            'dataset_info': {
                'filename': os.path.basename(self.csv_file),
                'shape': self.analysis_results['basic']['shape'],
                'columns': len(self.analysis_results['basic']['columns']),
                'data_types': self.analysis_results['basic']['dtypes']
            },
            'key_findings': {},
            'visualizations': self.visualizations
        }
        
        # Add key statistical findings
        if 'numeric_summary' in self.analysis_results['basic']:
            numeric_cols = list(self.analysis_results['basic']['numeric_summary'].keys())
            if numeric_cols:
                narrative_data['key_findings']['numeric_insights'] = f"Dataset contains {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:5])}"
        
        if 'categorical_summary' in self.analysis_results['basic']:
            cat_info = self.analysis_results['basic']['categorical_summary']
            narrative_data['key_findings']['categorical_insights'] = f"Found {len(cat_info)} categorical variables with varying cardinalities"
        
        # Add advanced findings
        if 'advanced' in self.analysis_results:
            adv = self.analysis_results['advanced']
            if 'outliers_detected' in adv:
                narrative_data['key_findings']['outliers'] = f"Detected {adv['outliers_detected']} potential outliers ({adv.get('outlier_percentage', 0):.1f}% of data)"
            
            if 'clustering' in adv:
                narrative_data['key_findings']['clusters'] = f"Identified {adv['clustering']['optimal_clusters']} natural groupings in the data"
        
        # Include LLM insights if available
        if 'llm_insights' in self.analysis_results:
            narrative_data['llm_insights'] = self.analysis_results['llm_insights']
        
        # Craft narrative prompt
        prompt = f"""
Create a comprehensive data analysis report in Markdown format for the dataset: {os.path.basename(self.csv_file)}

Use this analysis summary: {json.dumps(narrative_data, indent=2)}

Structure your report with the following sections:
# Data Analysis Report: {os.path.basename(self.csv_file)}

## Dataset Overview
Describe the dataset size, structure, and general characteristics.

## Data Quality Assessment  
Discuss missing values, data types, and any quality issues discovered.

## Key Findings
Present the most important insights and patterns found in the data.

## Statistical Analysis
Explain the statistical methods used and their results.

## Visualizations
Reference the generated charts (data_overview.png, advanced_analysis.png, insights_analysis.png) and explain what they reveal.

## Business Implications  
Provide actionable recommendations based on the analysis.

## Methodology
Briefly explain the analytical approach and tools used.

Make the report engaging, professional, and actionable. Use specific numbers and insights from the analysis. Keep it concise but comprehensive.
"""

        try:
            messages = [
                {"role": "system", "content": "You are a senior data scientist creating a professional analysis report. Write in clear, engaging prose with specific insights and actionable recommendations."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.call_llm(messages)
            narrative = response['choices'][0]['message']['content']
            
            # Ensure proper image references
            for viz_file in self.visualizations:
                if viz_file not in narrative:
                    narrative = narrative.replace("## Visualizations", f"## Visualizations\n\n![{viz_file}]({viz_file})\n")
            
            return narrative
            
        except Exception as e:
            logging.error(f"Error generating narrative: {e}")
            # Fallback narrative
            return self._generate_fallback_narrative()
    
    def _generate_fallback_narrative(self) -> str:
        """Generate a basic narrative without LLM if API fails."""
        basic = self.analysis_results['basic']
        
        narrative = f"""# Data Analysis Report: {os.path.basename(self.csv_file)}

## Dataset Overview
This dataset contains {basic['shape'][0]:,} rows and {basic['shape'][1]} columns. The data includes various types of information with both numeric and categorical variables.

## Data Quality Assessment
- **Missing Values**: {sum(basic['missing_values'].values())} total missing values across all columns
- **Data Types**: The dataset contains {len([k for k, v in basic['dtypes'].items() if 'object' in str(v)])} text columns and {len([k for k, v in basic['dtypes'].items() if 'int' in str(v) or 'float' in str(v)])} numeric columns

## Key Findings
- Dataset shape: {basic['shape'][0]} rows × {basic['shape'][1]} columns
- Memory usage: {basic['memory_usage'] / (1024*1024):.1f} MB
"""

        if 'numeric_summary' in basic:
            narrative += "\n### Numeric Analysis\n"
            for col in list(basic['numeric_summary'].keys())[:3]:
                stats = basic['numeric_summary'][col]
                narrative += f"- **{col}**: Mean = {stats['mean']:.2f}, Std = {stats['std']:.2f}\n"

        if self.visualizations:
            narrative += "\n## Visualizations\n\n"
            for viz in self.visualizations:
                narrative += f"![{viz}]({viz})\n\n"

        narrative += """
## Methodology
This analysis was performed using Python with pandas for data manipulation, matplotlib and seaborn for visualization, and scikit-learn for advanced statistical analysis.
"""
        
        return narrative
    
    def run_analysis(self) -> None:
        """Execute the complete analysis pipeline."""
        logging.info(f"Starting analysis of {self.csv_file}")
        
        # Load and analyze data
        self.load_data()
        logging.info(f"Loaded dataset with shape: {self.df.shape}")
        
        # Perform analyses
        self.basic_analysis()
        self.advanced_analysis()
        
        # Get LLM insights
        self.get_llm_insights()
        
        # Create visualizations
        viz_files = self.create_visualizations()
        logging.info(f"Created visualizations: {viz_files}")
        
        # Generate narrative
        narrative = self.generate_narrative()
        
        # Save README.md
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(narrative)
        
        logging.info("Analysis complete! Generated README.md and visualization files.")

def main():
    """Main function to run the analysis script."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        sys.exit(1)
    
    try:
        analyzer = DataAnalyzer(csv_file)
        analyzer.run_analysis()
        print("Analysis completed successfully!")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
