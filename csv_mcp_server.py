import asyncio
import json
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path
from typing import Dict, Any, List
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime

warnings.filterwarnings('ignore')

class WorkspaceManager:
    @staticmethod
    def get_workspace_dir() -> str:
        workspace = os.environ.get('MCP_WORKSPACE_DIR')
        if workspace and os.path.exists(workspace):
            return os.path.abspath(workspace)
        
        workspace = os.path.join(os.getcwd(), 'mcp_workspace')
        os.makedirs(workspace, exist_ok=True)
        return workspace
    
    @staticmethod
    def get_output_dir(subdir: str = "visualizations") -> str:
        workspace = WorkspaceManager.get_workspace_dir()
        output_dir = os.path.join(workspace, subdir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    @staticmethod
    def resolve_file_path(file_path: str) -> str:
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        
        workspace = WorkspaceManager.get_workspace_dir()
        workspace_path = os.path.join(workspace, file_path)
        if os.path.exists(workspace_path):
            return workspace_path
        
        current_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(current_path):
            return current_path
        
        return workspace_path

class CSVAnalyzer:
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        try:
            resolved_path = WorkspaceManager.resolve_file_path(file_path)
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(resolved_path, encoding=encoding, **kwargs)
                    return df
                except UnicodeDecodeError:
                    continue
            
            df = pd.read_csv(resolved_path, encoding='utf-8', errors='ignore', **kwargs)
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    @staticmethod
    def generate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
        stats_dict = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            numeric_stats = df[numeric_cols].describe()
            stats_dict["numeric_summary"] = numeric_stats.to_dict()
            
            for col in numeric_cols:
                if col in stats_dict["numeric_summary"]:
                    try:
                        stats_dict["numeric_summary"][col].update({
                            "skewness": float(df[col].skew()) if not df[col].empty else 0,
                            "kurtosis": float(df[col].kurtosis()) if not df[col].empty else 0,
                            "variance": float(df[col].var()) if not df[col].empty else 0
                        })
                    except:
                        pass
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                unique_vals = df[col].unique()
                stats_dict["categorical_summary"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "unique_values": [str(val) for val in unique_vals[:10]],  
                    "value_counts": {str(k): int(v) for k, v in df[col].value_counts().head().items()},
                    "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
                }
            except:
                pass
        
        return stats_dict
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str] = None, method: str = "iqr") -> Dict[str, Any]:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_dict = {"method": method, "outliers_by_column": {}, "total_outliers": 0}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            outlier_indices = []
            
            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df[outlier_mask].index.tolist()
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > 3
                outlier_indices = col_data[outlier_mask].index.tolist()
            
            elif method == "dbscan":
                if len(col_data) > 5:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(col_data.values.reshape(-1, 1))
                    
                    dbscan = DBSCAN(eps=0.5, min_samples=3)
                    clusters = dbscan.fit_predict(scaled_data)
                    
                    outlier_indices = col_data[clusters == -1].index.tolist()
            
            outliers_dict["outliers_by_column"][col] = {
                "count": len(outlier_indices),
                "indices": outlier_indices,
                "values": df.loc[outlier_indices, col].tolist() if outlier_indices else []
            }
            outliers_dict["total_outliers"] += len(outlier_indices)
        
        return outliers_dict
    
    @staticmethod
    def calculate_correlations(df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        if numeric_df.shape[1] < 2:
            return {"error": "At least 2 numeric columns required for correlation analysis"}
        
        try:
            corr_matrix = numeric_df.corr(method=method)
            
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": round(corr_value, 4),
                            "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                        })
            
            return {
                "method": method,
                "correlation_matrix": corr_matrix.round(4).to_dict(),
                "strong_correlations": strong_correlations,
                "summary": {
                    "total_pairs": len(strong_correlations),
                    "columns_analyzed": list(numeric_df.columns)
                }
            }
        except Exception as e:
            return {"error": f"Error calculating correlations: {str(e)}"}
    
    @staticmethod
    def clean_data(df: pd.DataFrame, operations: List[str]) -> Dict[str, Any]:
        cleaned_df = df.copy()
        operations_log = []
        
        initial_shape = cleaned_df.shape
        
        for operation in operations:
            if operation == "drop_duplicates":
                before_count = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                after_count = len(cleaned_df)
                operations_log.append({
                    "operation": "drop_duplicates",
                    "removed_rows": before_count - after_count
                })
            
            elif operation == "drop_missing":
                before_count = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                after_count = len(cleaned_df)
                operations_log.append({
                    "operation": "drop_missing",
                    "removed_rows": before_count - after_count
                })
            
            elif operation == "fill_missing_mean":
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    missing_count = cleaned_df[col].isnull().sum()
                    if missing_count > 0:
                        cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        operations_log.append({
                            "operation": "fill_missing_mean",
                            "column": col,
                            "filled_values": missing_count
                        })
            
            elif operation == "fill_missing_mode":
                categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    missing_count = cleaned_df[col].isnull().sum()
                    if missing_count > 0 and not cleaned_df[col].mode().empty:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                        operations_log.append({
                            "operation": "fill_missing_mode",
                            "column": col,
                            "filled_values": missing_count
                        })
        
        return {
            "initial_shape": initial_shape,
            "final_shape": cleaned_df.shape,
            "operations_performed": operations_log,
            "cleaned_data": cleaned_df
        }
    
    @staticmethod
    def filter_data(df: pd.DataFrame, filters: List[Dict]) -> Dict[str, Any]:
        filtered_df = df.copy()
        filter_log = []
        
        for filter_config in filters:
            column = filter_config.get("column")
            operator = filter_config.get("operator")
            value = filter_config.get("value")
            
            if column not in df.columns:
                continue
            
            before_count = len(filtered_df)
            
            if operator == "greater_than":
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == "less_than":
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == "equals":
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == "not_equals":
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == "contains":
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
            elif operator == "between":
                if isinstance(value, list) and len(value) == 2:
                    filtered_df = filtered_df[(filtered_df[column] >= value[0]) & (filtered_df[column] <= value[1])]
            
            after_count = len(filtered_df)
            filter_log.append({
                "column": column,
                "operator": operator,
                "value": value,
                "rows_remaining": after_count,
                "rows_filtered": before_count - after_count
            })
        
        return {
            "original_rows": len(df),
            "filtered_rows": len(filtered_df),
            "filters_applied": filter_log,
            "filtered_data": filtered_df
        }
    
    @staticmethod
    def group_data(df: pd.DataFrame, group_by: List[str], aggregations: Dict[str, str]) -> Dict[str, Any]:
        try:
            if not all(col in df.columns for col in group_by):
                return {"error": "One or more grouping columns not found in dataframe"}
            
            grouped = df.groupby(group_by)
            
            agg_results = {}
            for column, operation in aggregations.items():
                if column not in df.columns:
                    continue
                
                if operation == "mean":
                    agg_results[f"{column}_mean"] = grouped[column].mean()
                elif operation == "sum":
                    agg_results[f"{column}_sum"] = grouped[column].sum()
                elif operation == "count":
                    agg_results[f"{column}_count"] = grouped[column].count()
                elif operation == "min":
                    agg_results[f"{column}_min"] = grouped[column].min()
                elif operation == "max":
                    agg_results[f"{column}_max"] = grouped[column].max()
                elif operation == "std":
                    agg_results[f"{column}_std"] = grouped[column].std()
            
            result_df = pd.DataFrame(agg_results).reset_index()
            
            return {
                "grouped_by": group_by,
                "aggregations_applied": aggregations,
                "result_shape": result_df.shape,
                "grouped_data": result_df.to_dict('records')
            }
        
        except Exception as e:
            return {"error": f"Error in grouping operation: {str(e)}"}
    
    @staticmethod
    def create_visualization(df: pd.DataFrame, plot_type: str, columns: List[str], save_path: str, **kwargs) -> str:
        try:
            plt.figure(figsize=(12, 8))
            
            if not save_path.endswith('.png'):
                save_path += '.png'
            
            abs_save_path = os.path.abspath(save_path)
            parent_dir = os.path.dirname(abs_save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            if plot_type == "histogram":
                if len(columns) == 1 and columns[0] in df.columns:
                    data = df[columns[0]].dropna()
                    if len(data) == 0:
                        plt.close()
                        return "ERROR: No valid data in specified column"
                    
                    plt.hist(data, bins=kwargs.get('bins', 30), alpha=0.7, edgecolor='black')
                    plt.title(f'Histogram of {columns[0]}')
                    plt.xlabel(columns[0])
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                else:
                    plt.close()
                    return f"ERROR: Column '{columns[0]}' not found. Available: {list(df.columns)}"
            
            elif plot_type == "boxplot":
                numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
                if not numeric_cols:
                    plt.close()
                    return "ERROR: No valid numeric columns found"
                
                data_to_plot = [df[col].dropna() for col in numeric_cols]
                plt.boxplot(data_to_plot, labels=numeric_cols)
                plt.title('Boxplot of Numeric Variables')
                plt.ylabel('Values')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            elif plot_type == "scatter":
                if len(columns) >= 2:
                    col_x, col_y = columns[0], columns[1]
                    if col_x in df.columns and col_y in df.columns:
                        plt.scatter(df[col_x], df[col_y], alpha=0.6, s=50)
                        plt.xlabel(col_x)
                        plt.ylabel(col_y)
                        plt.title(f'Scatter Plot: {col_x} vs {col_y}')
                        plt.grid(True, alpha=0.3)
                    else:
                        plt.close()
                        return f"ERROR: Columns {col_x}, {col_y} do not exist"
                else:
                    plt.close()
                    return "ERROR: At least 2 columns needed for scatter plot"
            
            elif plot_type == "correlation_heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, square=True)
                    plt.title('Correlation Matrix')
                else:
                    plt.close()
                    return "ERROR: No numeric columns for correlation"
            
            elif plot_type == "bar":
                if len(columns) >= 1:
                    col = columns[0]
                    if col in df.columns:
                        value_counts = df[col].value_counts().head(kwargs.get('top_n', 10))
                        plt.bar(range(len(value_counts)), value_counts.values)
                        plt.title(f'Top Values - {col}')
                        plt.xlabel('Categories')
                        plt.ylabel('Frequency')
                        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                    else:
                        plt.close()
                        return f"ERROR: Column '{col}' not found"
                else:
                    plt.close()
                    return "ERROR: At least one column needed for bar chart"
            
            else:
                plt.close()
                return f"ERROR: Plot type '{plot_type}' not supported"
            
            plt.tight_layout()
            plt.savefig(abs_save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            if os.path.exists(abs_save_path):
                file_size = os.path.getsize(abs_save_path)
                return f"SUCCESS: Visualization saved to {abs_save_path} ({file_size} bytes)"
            else:
                return f"ERROR: Could not save file to {abs_save_path}"
                
        except Exception as e:
            plt.close()
            return f"ERROR creating visualization: {str(e)}"

class MCPServer:
    def __init__(self):
        self.tools = {
            "analyze_csv": self.analyze_csv,
            "detect_outliers_in_csv": self.detect_outliers_in_csv,
            "calculate_correlations_csv": self.calculate_correlations_csv,
            "clean_csv_data": self.clean_csv_data,
            "filter_csv_data": self.filter_csv_data,
            "group_csv_data": self.group_csv_data,
            "create_csv_visualization": self.create_csv_visualization,
            "debug_workspace": self.debug_workspace
        }
    
    def create_response(self, request_id: str, result: Any = None, error: Any = None):
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        
        if error:
            response["error"] = {
                "code": -32000,
                "message": str(error)
            }
        else:
            response["result"] = result
        
        return response
    
    def create_text_content(self, text: str):
        return {
            "type": "text",
            "text": text
        }
    
    def create_tool_result(self, content_list: List[Dict], is_error: bool = False):
        return {
            "content": content_list,
            "isError": is_error
        }
    
    async def analyze_csv(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            df = CSVAnalyzer.load_csv(file_path)
            stats = CSVAnalyzer.generate_descriptive_stats(df)
            
            summary = f"""CSV ANALYSIS REPORT: {file_path}
Shape: {stats['shape'][0]} rows x {stats['shape'][1]} columns
Memory usage: {stats['memory_usage']}
Columns: {', '.join(stats['columns'])}

NUMERIC ANALYSIS:"""
            
            if stats['numeric_summary']:
                for col, col_stats in stats['numeric_summary'].items():
                    try:
                        summary += f"""
{col}:
  Mean: {col_stats['mean']:.2f}
  Median: {col_stats['50%']:.2f}
  Std Dev: {col_stats['std']:.2f}
  Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]"""
                        if 'skewness' in col_stats:
                            summary += f"\n  Skewness: {col_stats['skewness']:.2f}"
                    except:
                        summary += f"\n{col}: Error in statistics"
            
            summary += f"\n\nCATEGORICAL ANALYSIS:"
            if stats['categorical_summary']:
                for col, col_stats in stats['categorical_summary'].items():
                    summary += f"""
{col}:
  Unique values: {col_stats['unique_count']}
  Most frequent: {col_stats['mode']}
  Sample: {', '.join(col_stats['unique_values'][:5])}"""
            
            summary += f"\n\nMISSING VALUES:"
            missing_total = sum(stats['missing_values'].values())
            if missing_total > 0:
                for col, missing_count in stats['missing_values'].items():
                    if missing_count > 0:
                        percentage = (missing_count / stats['shape'][0]) * 100
                        summary += f"\n{col}: {missing_count} ({percentage:.1f}%)"
            else:
                summary += "\nNo missing values found"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error loading CSV file: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def detect_outliers_in_csv(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            columns = arguments.get("columns", None)
            method = arguments.get("method", "iqr")
            
            df = CSVAnalyzer.load_csv(file_path)
            outliers_result = CSVAnalyzer.detect_outliers(df, columns, method)
            
            summary = f"""OUTLIER DETECTION REPORT: {file_path}
Method: {outliers_result['method'].upper()}
Total outliers found: {outliers_result['total_outliers']}

OUTLIERS BY COLUMN:"""
            
            for col, col_outliers in outliers_result['outliers_by_column'].items():
                summary += f"""
{col}: {col_outliers['count']} outliers"""
                if col_outliers['values']:
                    values_str = ', '.join([f"{v:.2f}" if isinstance(v, (int, float)) else str(v) 
                                          for v in col_outliers['values'][:10]])
                    summary += f"\n  Sample values: {values_str}"
                    if len(col_outliers['values']) > 10:
                        summary += f" (and {len(col_outliers['values'])-10} more)"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error detecting outliers: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def calculate_correlations_csv(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            method = arguments.get("method", "pearson")
            
            df = CSVAnalyzer.load_csv(file_path)
            corr_result = CSVAnalyzer.calculate_correlations(df, method)
            
            if "error" in corr_result:
                text_content = self.create_text_content(corr_result["error"])
                return self.create_tool_result([text_content], is_error=True)
            
            summary = f"""CORRELATION ANALYSIS: {file_path}
Method: {corr_result['method'].title()}
Columns analyzed: {', '.join(corr_result['summary']['columns_analyzed'])}

STRONG CORRELATIONS (|r| > 0.5):"""
            
            if corr_result['strong_correlations']:
                for corr in corr_result['strong_correlations']:
                    summary += f"""
{corr['variable1']} vs {corr['variable2']}: {corr['correlation']} ({corr['strength']})"""
            else:
                summary += "\nNo strong correlations found"
            
            summary += f"\n\nCORRELATION MATRIX:"
            corr_matrix = corr_result['correlation_matrix']
            columns = list(corr_matrix.keys())
            
            summary += f"\n{'':<12}" + "".join([f"{col:<12}" for col in columns])
            for row_col in columns:
                summary += f"\n{row_col:<12}"
                for col_col in columns:
                    value = corr_matrix[row_col][col_col]
                    summary += f"{value:<12.3f}"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error calculating correlations: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def clean_csv_data(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            operations = arguments.get("operations", ["drop_duplicates"])
            save_cleaned = arguments.get("save_cleaned", False)
            
            df = CSVAnalyzer.load_csv(file_path)
            clean_result = CSVAnalyzer.clean_data(df, operations)
            
            summary = f"""DATA CLEANING REPORT: {file_path}
Initial shape: {clean_result['initial_shape']}
Final shape: {clean_result['final_shape']}
Rows removed: {clean_result['initial_shape'][0] - clean_result['final_shape'][0]}

OPERATIONS PERFORMED:"""
            
            for operation in clean_result['operations_performed']:
                if operation['operation'] == 'drop_duplicates':
                    summary += f"\nDrop duplicates: {operation['removed_rows']} rows removed"
                elif operation['operation'] == 'drop_missing':
                    summary += f"\nDrop missing values: {operation['removed_rows']} rows removed"
                elif 'fill_missing' in operation['operation']:
                    summary += f"\nFill missing in {operation['column']}: {operation['filled_values']} values filled"
            
            if save_cleaned:
                cleaned_dir = WorkspaceManager.get_output_dir("cleaned_data")
                base_name = Path(file_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cleaned_filename = f"{base_name}_cleaned_{timestamp}.csv"
                cleaned_path = os.path.join(cleaned_dir, cleaned_filename)
                
                clean_result['cleaned_data'].to_csv(cleaned_path, index=False)
                summary += f"\n\nCleaned data saved to: {cleaned_path}"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error cleaning data: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def filter_csv_data(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            filters = arguments.get("filters", [])
            save_filtered = arguments.get("save_filtered", False)
            
            df = CSVAnalyzer.load_csv(file_path)
            filter_result = CSVAnalyzer.filter_data(df, filters)
            
            summary = f"""DATA FILTERING REPORT: {file_path}
Original rows: {filter_result['original_rows']}
Filtered rows: {filter_result['filtered_rows']}
Rows removed: {filter_result['original_rows'] - filter_result['filtered_rows']}

FILTERS APPLIED:"""
            
            for filter_info in filter_result['filters_applied']:
                summary += f"""
{filter_info['column']} {filter_info['operator']} {filter_info['value']}:
  Rows remaining: {filter_info['rows_remaining']}
  Rows filtered: {filter_info['rows_filtered']}"""
            
            if save_filtered:
                filtered_dir = WorkspaceManager.get_output_dir("filtered_data")
                base_name = Path(file_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filtered_filename = f"{base_name}_filtered_{timestamp}.csv"
                filtered_path = os.path.join(filtered_dir, filtered_filename)
                
                filter_result['filtered_data'].to_csv(filtered_path, index=False)
                summary += f"\n\nFiltered data saved to: {filtered_path}"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error filtering data: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def group_csv_data(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            group_by = arguments.get("group_by", [])
            aggregations = arguments.get("aggregations", {})
            save_grouped = arguments.get("save_grouped", False)
            
            df = CSVAnalyzer.load_csv(file_path)
            group_result = CSVAnalyzer.group_data(df, group_by, aggregations)
            
            if "error" in group_result:
                text_content = self.create_text_content(group_result["error"])
                return self.create_tool_result([text_content], is_error=True)
            
            summary = f"""DATA GROUPING REPORT: {file_path}
Grouped by: {', '.join(group_result['grouped_by'])}
Aggregations: {group_result['aggregations_applied']}
Result shape: {group_result['result_shape']}

GROUPED DATA (first 10 rows):"""
            
            for i, row in enumerate(group_result['grouped_data'][:10]):
                summary += f"\nRow {i+1}: {row}"
            
            if len(group_result['grouped_data']) > 10:
                summary += f"\n... and {len(group_result['grouped_data'])-10} more rows"
            
            if save_grouped:
                grouped_dir = WorkspaceManager.get_output_dir("grouped_data")
                base_name = Path(file_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                grouped_filename = f"{base_name}_grouped_{timestamp}.csv"
                grouped_path = os.path.join(grouped_dir, grouped_filename)
                
                pd.DataFrame(group_result['grouped_data']).to_csv(grouped_path, index=False)
                summary += f"\n\nGrouped data saved to: {grouped_path}"
            
            text_content = self.create_text_content(summary)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error grouping data: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def create_csv_visualization(self, arguments: Dict) -> Dict:
        try:
            file_path = arguments.get("file_path")
            plot_type = arguments.get("plot_type")
            columns = arguments.get("columns", [])
            filename = arguments.get("filename")
            plot_options = arguments.get("plot_options", {})
            
            df = CSVAnalyzer.load_csv(file_path)
            
            if not filename:
                base_name = Path(file_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{base_name}_{plot_type}_{timestamp}.png"
            elif not filename.endswith('.png'):
                filename += '.png'
            
            viz_dir = WorkspaceManager.get_output_dir("visualizations")
            save_path = os.path.join(viz_dir, filename)
            
            visualization_result = CSVAnalyzer.create_visualization(df, plot_type, columns, save_path, **plot_options)
            
            if "SUCCESS" in visualization_result:
                success_message = f"""VISUALIZATION CREATED
Type: {plot_type}
File: {file_path}
Columns: {', '.join(columns)}
Saved as: {filename}
Directory: {viz_dir}

{visualization_result}"""
                text_content = self.create_text_content(success_message)
                return self.create_tool_result([text_content])
            else:
                error_message = f"VISUALIZATION ERROR:\n{visualization_result}"
                text_content = self.create_text_content(error_message)
                return self.create_tool_result([text_content], is_error=True)
                
        except Exception as e:
            error_msg = f"Error in visualization: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def debug_workspace(self, arguments: Dict) -> Dict:
        try:
            workspace = WorkspaceManager.get_workspace_dir()
            current_dir = os.getcwd()
            
            debug_info = f"""WORKSPACE DEBUG:
Current directory: {current_dir}
Workspace configured: {workspace}
MCP_WORKSPACE_DIR: {os.environ.get('MCP_WORKSPACE_DIR', 'Not configured')}

WORKSPACE STRUCTURE:"""
            
            try:
                if os.path.exists(workspace):
                    for root, dirs, files in os.walk(workspace):
                        level = root.replace(workspace, '').count(os.sep)
                        indent = ' ' * 2 * level
                        debug_info += f"\n{indent}{os.path.basename(root)}/"
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            debug_info += f"\n{subindent}{file} ({file_size} bytes)"
                else:
                    debug_info += f"\nWorkspace does not exist: {workspace}"
            except Exception as e:
                debug_info += f"\nError accessing workspace: {str(e)}"
            
            debug_info += f"""

OUTPUT DIRECTORIES:
- Visualizations: {WorkspaceManager.get_output_dir('visualizations')}
- Cleaned data: {WorkspaceManager.get_output_dir('cleaned_data')}
- Filtered data: {WorkspaceManager.get_output_dir('filtered_data')}
- Grouped data: {WorkspaceManager.get_output_dir('grouped_data')}
"""
            
            text_content = self.create_text_content(debug_info)
            return self.create_tool_result([text_content])
            
        except Exception as e:
            error_msg = f"Error in debug: {str(e)}"
            text_content = self.create_text_content(error_msg)
            return self.create_tool_result([text_content], is_error=True)
    
    async def handle_initialize(self, request_id: str, params: Dict):
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "Advanced CSV Analysis Server",
                "version": "1.0.0"
            }
        }
        return self.create_response(request_id, result)
    
    async def handle_tools_list(self, request_id: str, params: Dict):
        tools = [
            {
                "name": "analyze_csv",
                "description": "Analyze a CSV file and provide comprehensive statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file to analyze"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "detect_outliers_in_csv",
                "description": "Detect outliers in CSV data using various methods",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to analyze for outliers (default: all numeric)"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore", "dbscan"],
                            "description": "Method for outlier detection",
                            "default": "iqr"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "calculate_correlations_csv",
                "description": "Calculate correlations between numeric variables",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "description": "Correlation method",
                            "default": "pearson"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "clean_csv_data",
                "description": "Clean CSV data by removing duplicates, handling missing values",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["drop_duplicates", "drop_missing", "fill_missing_mean", "fill_missing_mode"]
                            },
                            "description": "Cleaning operations to perform"
                        },
                        "save_cleaned": {
                            "type": "boolean",
                            "description": "Whether to save the cleaned data",
                            "default": False
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "filter_csv_data",
                "description": "Filter CSV data based on specified conditions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "operator": {
                                        "type": "string",
                                        "enum": ["greater_than", "less_than", "equals", "not_equals", "contains", "between"]
                                    },
                                    "value": {}
                                },
                                "required": ["column", "operator", "value"]
                            },
                            "description": "Filter conditions to apply"
                        },
                        "save_filtered": {
                            "type": "boolean",
                            "description": "Whether to save the filtered data",
                            "default": False
                        }
                    },
                    "required": ["file_path", "filters"]
                }
            },
            {
                "name": "group_csv_data",
                "description": "Group CSV data and perform aggregations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to group by"
                        },
                        "aggregations": {
                            "type": "object",
                            "description": "Aggregation operations for each column",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["mean", "sum", "count", "min", "max", "std"]
                            }
                        },
                        "save_grouped": {
                            "type": "boolean",
                            "description": "Whether to save the grouped data",
                            "default": False
                        }
                    },
                    "required": ["file_path", "group_by", "aggregations"]
                }
            },
            {
                "name": "create_csv_visualization",
                "description": "Create visualizations from CSV data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file"
                        },
                        "plot_type": {
                            "type": "string",
                            "enum": ["histogram", "boxplot", "scatter", "correlation_heatmap", "bar"],
                            "description": "Type of plot to create"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to use in the visualization"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Custom filename for the visualization",
                            "default": None
                        },
                        "plot_options": {
                            "type": "object",
                            "description": "Additional options for the plot",
                            "default": {}
                        }
                    },
                    "required": ["file_path", "plot_type", "columns"]
                }
            },
            {
                "name": "debug_workspace",
                "description": "Debug workspace configuration and file structure",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        result = {"tools": tools}
        return self.create_response(request_id, result)
    
    async def handle_tools_call(self, request_id: str, params: Dict):
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name in self.tools:
                result = await self.tools[tool_name](arguments)
                return self.create_response(request_id, result)
            else:
                error = f"Unknown tool: {tool_name}"
                return self.create_response(request_id, error=error)
                
        except Exception as e:
            error = f"Error executing tool: {str(e)}"
            return self.create_response(request_id, error=error)
    
    async def handle_request(self, request: Dict):
        try:
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})
            
            if method == "initialize":
                return await self.handle_initialize(request_id, params)
            elif method == "tools/list":
                return await self.handle_tools_list(request_id, params)
            elif method == "tools/call":
                return await self.handle_tools_call(request_id, params)
            else:
                error = f"Unknown method: {method}"
                return self.create_response(request_id, error=error)
                
        except Exception as e:
            error = f"Error processing request: {str(e)}"
            return self.create_response(request.get("id"), error=error)

async def main():
    server = MCPServer()
    
    workspace_env = os.environ.get('MCP_WORKSPACE_DIR')
    if workspace_env:
        print(f"Using workspace from environment: {workspace_env}", file=sys.stderr)
    else:
        default_workspace = WorkspaceManager.get_workspace_dir()
        print(f"Using default workspace: {default_workspace}", file=sys.stderr)
    
    print("Advanced CSV MCP Server started", file=sys.stderr)
    
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON Error: {e}", file=sys.stderr)
                continue
            
            response = await server.handle_request(request)
            
            response_json = json.dumps(response)
            print(response_json, flush=True)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())