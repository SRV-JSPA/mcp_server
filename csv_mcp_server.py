import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from scipy import stats
import warnings
import json

from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult
import mcp.server.stdio

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  


server = Server("CSV Analysis Server")

class CSVAnalyzer:
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        try:
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    return df
                except UnicodeDecodeError:
                    continue
            
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', **kwargs)
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        except Exception as e:
            raise Exception(f"Error cargando CSV: {str(e)}")
    
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
                    stats_dict["numeric_summary"][col].update({
                        "skewness": float(df[col].skew()) if not df[col].empty else 0,
                        "kurtosis": float(df[col].kurtosis()) if not df[col].empty else 0,
                        "variance": float(df[col].var()) if not df[col].empty else 0
                    })
        
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_vals = df[col].unique()
            stats_dict["categorical_summary"][col] = {
                "unique_count": int(df[col].nunique()),
                "unique_values": [str(val) for val in unique_vals[:10]],  
                "value_counts": {str(k): int(v) for k, v in df[col].value_counts().head().items()},
                "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
            }
        
        return stats_dict
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outliers_info[col] = {"method": method, "outliers": []}
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_info[col]["outliers"] = [float(x) for x in df[outlier_mask][col].tolist()]
                outliers_info[col]["bounds"] = {"lower": float(lower_bound), "upper": float(upper_bound)}
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > 3
                outliers_info[col]["outliers"] = [float(x) for x in df[col][outlier_mask].tolist()]
                outliers_info[col]["threshold"] = 3
            
            outliers_info[col]["count"] = len(outliers_info[col]["outliers"])
        
        return outliers_info
    
    @staticmethod
    def clean_data(df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        df_clean = df.copy()
        
        for operation in operations:
            if operation == "drop_duplicates":
                df_clean = df_clean.drop_duplicates()
            elif operation == "fill_numeric_mean":
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            elif operation == "fill_numeric_median":
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            elif operation == "drop_missing":
                df_clean = df_clean.dropna()
            elif operation == "fill_categorical_mode":
                categorical_cols = df_clean.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
        
        return df_clean
    
    @staticmethod
    def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No hay columnas numericas para calcular correlaciones"}
        
        corr_matrix = numeric_df.corr()
        
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value):
                    corr_pairs.append({
                        "variables": [col1, col2],
                        "correlation": float(corr_value),
                        "abs_correlation": float(abs(corr_value))
                    })
        
        
        corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        
        return {
            "correlation_matrix": {col: {row: float(val) if not pd.isna(val) else 0 
                                       for row, val in corr_matrix[col].items()} 
                                 for col in corr_matrix.columns},
            "strong_correlations": corr_pairs[:5],  
            "weak_correlations": corr_pairs[-5:] if len(corr_pairs) >= 5 else []
        }
    
    @staticmethod
    def create_visualization(df: pd.DataFrame, plot_type: str, columns: List[str], save_path: str) -> str:
        plt.figure(figsize=(10, 6))
        
        try:
            if plot_type == "histogram":
                if len(columns) == 1 and columns[0] in df.columns:
                    plt.hist(df[columns[0]].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'Histograma de {columns[0]}')
                    plt.xlabel(columns[0])
                    plt.ylabel('Frecuencia')
                
            elif plot_type == "boxplot":
                if len(columns) >= 1:
                    numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
                    if numeric_cols:
                        df[numeric_cols].boxplot()
                        plt.title('Boxplot')
                        plt.xticks(rotation=45)
                
            elif plot_type == "scatter":
                if len(columns) >= 2:
                    col_x, col_y = columns[0], columns[1]
                    if col_x in df.columns and col_y in df.columns:
                        plt.scatter(df[col_x], df[col_y], alpha=0.6)
                        plt.xlabel(col_x)
                        plt.ylabel(col_y)
                        plt.title(f'Grafico de dispersion: {col_x} vs {col_y}')
                
            elif plot_type == "correlation_heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
                    plt.title('Matriz de Correlacion')
                    plt.tight_layout()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"Visualizacion guardada en: {save_path}"
            
        except Exception as e:
            plt.close()
            return f"Error creando visualizacion: {str(e)}"



@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_csv",
            description="Analyze a CSV file and provide comprehensive statistics and information",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file to analyze"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="detect_outliers_in_csv",
            description="Detect outliers in CSV data using different methods",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "method": {
                        "type": "string",
                        "description": "Method to use for outlier detection: 'iqr' or 'zscore'",
                        "default": "iqr"
                    },
                    "column": {
                        "type": "string",
                        "description": "Specific column to analyze (optional)",
                        "default": None
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="clean_csv_data",
            description="Clean CSV data using specified operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "operations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of cleaning operations to apply"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save cleaned data (optional)"
                    }
                },
                "required": ["file_path", "operations"]
            }
        ),
        Tool(
            name="calculate_correlations_csv",
            description="Calculate correlations between numeric columns in CSV",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="create_csv_visualization",
            description="Create visualizations from CSV data",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "plot_type": {
                        "type": "string",
                        "description": "Type of plot: 'histogram', 'boxplot', 'scatter', 'correlation_heatmap'"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to use in the visualization"
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path to save the visualization (optional)"
                    }
                },
                "required": ["file_path", "plot_type", "columns"]
            }
        ),
        Tool(
            name="filter_csv_data",
            description="Filter CSV data based on specified criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Dictionary of filters to apply"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save filtered data (optional)"
                    }
                },
                "required": ["file_path", "filters"]
            }
        )
    ]



@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "analyze_csv":
            file_path = arguments.get("file_path")
            df = CSVAnalyzer.load_csv(file_path)
            stats = CSVAnalyzer.generate_descriptive_stats(df)
            
            summary = f"""
ANALISIS COMPLETO DEL ARCHIVO: {file_path}

INFORMACION GENERAL:
- Dimensiones: {stats['shape'][0]} filas × {stats['shape'][1]} columnas
- Memoria utilizada: {stats['memory_usage']}
- Columnas: {', '.join(stats['columns'])}

COLUMNAS NUMERICAS:
"""
            
            if stats['numeric_summary']:
                for col, col_stats in stats['numeric_summary'].items():
                    summary += f"\n  • {col}:"
                    summary += f"\n    - Promedio: {col_stats['mean']:.2f}"
                    summary += f"\n    - Mediana: {col_stats['50%']:.2f}"
                    summary += f"\n    - Desviacion estandar: {col_stats['std']:.2f}"
                    summary += f"\n    - Rango: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]"
                    summary += f"\n    - Asimetria: {col_stats['skewness']:.2f}"
            
            summary += f"\n\nCOLUMNAS CATEGÓRICAS:"
            if stats['categorical_summary']:
                for col, col_stats in stats['categorical_summary'].items():
                    summary += f"\n  • {col}:"
                    summary += f"\n    - Valores unicos: {col_stats['unique_count']}"
                    summary += f"\n    - Valor mas frecuente: {col_stats['mode']}"
            
            summary += f"\n\nVALORES FALTANTES:"
            missing_total = sum(stats['missing_values'].values())
            if missing_total > 0:
                for col, missing_count in stats['missing_values'].items():
                    if missing_count > 0:
                        percentage = (missing_count / stats['shape'][0]) * 100
                        summary += f"\n  • {col}: {missing_count} ({percentage:.1f}%)"
            else:
                summary += "\n No hay valores faltantes"
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        elif name == "detect_outliers_in_csv":
            file_path = arguments.get("file_path")
            method = arguments.get("method", "iqr")
            column = arguments.get("column")
            
            df = CSVAnalyzer.load_csv(file_path)
            
            if column and column not in df.columns:
                error_msg = f"Error: La columna '{column}' no existe en el dataset"
                return CallToolResult(content=[TextContent(type="text", text=error_msg)])
            
            if column:
                df_analyze = df[[column]]
            else:
                df_analyze = df
            
            outliers_info = CSVAnalyzer.detect_outliers(df_analyze, method)
            
            summary = f"DETECCION DE VALORES ATIPICOS - Metodo: {method.upper()}\n"
            summary += "="*50 + "\n"
            
            total_outliers = 0
            for col, info in outliers_info.items():
                if info['count'] > 0:
                    total_outliers += info['count']
                    summary += f"\nColumna: {col}"
                    summary += f"\n   - Valores atipicos encontrados: {info['count']}"
                    summary += f"\n   - Valores: {info['outliers'][:10]}"  
                    if 'bounds' in info:
                        summary += f"\n   - Limites: [{info['bounds']['lower']:.2f}, {info['bounds']['upper']:.2f}]"
                    if len(info['outliers']) > 10:
                        summary += f"\n   - (y {len(info['outliers']) - 10} mas...)"
                    summary += "\n"
            
            if total_outliers == 0:
                summary += "\nNo se encontraron valores atipicos significativos"
            else:
                summary += f"\nTotal de valores atipicos: {total_outliers}"
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        elif name == "clean_csv_data":
            file_path = arguments.get("file_path")
            operations = arguments.get("operations", [])
            output_path = arguments.get("output_path")
            
            df_original = CSVAnalyzer.load_csv(file_path)
            df_clean = CSVAnalyzer.clean_data(df_original, operations)
            
            summary = "PROCESO DE LIMPIEZA DE DATOS\n"
            summary += "="*40 + "\n"
            summary += f"Archivo original: {file_path}\n"
            summary += f"Operaciones aplicadas: {', '.join(operations)}\n\n"
            
            summary += "CAMBIOS REALIZADOS:\n"
            summary += f"   - Filas antes: {len(df_original)}\n"
            summary += f"   - Filas despues: {len(df_clean)}\n"
            summary += f"   - Filas eliminadas: {len(df_original) - len(df_clean)}\n\n"
            
            missing_before = df_original.isnull().sum().sum()
            missing_after = df_clean.isnull().sum().sum()
            summary += f"   - Valores faltantes antes: {missing_before}\n"
            summary += f"   - Valores faltantes despues: {missing_after}\n"
            summary += f"   - Valores faltantes corregidos: {missing_before - missing_after}\n\n"
            
            if output_path:
                df_clean.to_csv(output_path, index=False)
                summary += f"Archivo limpiado guardado en: {output_path}\n"
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        elif name == "calculate_correlations_csv":
            file_path = arguments.get("file_path")
            df = CSVAnalyzer.load_csv(file_path)
            corr_info = CSVAnalyzer.calculate_correlations(df)
            
            if "error" in corr_info:
                return CallToolResult(content=[TextContent(type="text", text=corr_info["error"])])
            
            summary = "ANALISIS DE CORRELACIONES\n"
            summary += "="*30 + "\n\n"
            
            summary += "CORRELACIONES MAS FUERTES:\n"
            for i, corr in enumerate(corr_info["strong_correlations"][:5], 1):
                summary += f"{i}. {corr['variables'][0]} ← {corr['variables'][1]}: "
                summary += f"{corr['correlation']:.3f}\n"
            
            summary += "\nCORRELACIONES MAS DEBILES:\n"
            for i, corr in enumerate(corr_info["weak_correlations"][-5:], 1):
                summary += f"{i}. {corr['variables'][0]} ← {corr['variables'][1]}: "
                summary += f"{corr['correlation']:.3f}\n"
            
            summary += "\nINTERPRETACION:\n"
            summary += "  • |r| > 0.8: Correlacion muy fuerte\n"
            summary += "  • |r| > 0.6: Correlacion fuerte\n"
            summary += "  • |r| > 0.4: Correlacion moderada\n"
            summary += "  • |r| < 0.2: Correlacion debil\n"
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        elif name == "create_csv_visualization":
            file_path = arguments.get("file_path")
            plot_type = arguments.get("plot_type")
            columns = arguments.get("columns", [])
            save_path = arguments.get("save_path")
            
            df = CSVAnalyzer.load_csv(file_path)
            
            if not save_path:
                base_name = Path(file_path).stem
                save_path = f"{base_name}_{plot_type}.png"
            
            result = CSVAnalyzer.create_visualization(df, plot_type, columns, save_path)
            
            summary = f"VISUALIZACION CREADA\n"
            summary += "="*25 + "\n"
            summary += f"Tipo: {plot_type}\n"
            summary += f"Datos: {file_path}\n"
            summary += f"Columnas: {', '.join(columns)}\n"
            summary += f"Guardado en: {save_path}\n\n"
            summary += result
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        elif name == "filter_csv_data":
            file_path = arguments.get("file_path")
            filters = arguments.get("filters", {})
            output_path = arguments.get("output_path")
            
            df = CSVAnalyzer.load_csv(file_path)
            df_filtered = df.copy()
            
            applied_filters = []
            
            for column, filter_config in filters.items():
                if column not in df.columns:
                    error_msg = f"Error: La columna '{column}' no existe en el dataset"
                    return CallToolResult(content=[TextContent(type="text", text=error_msg)])
                
                operator = filter_config.get("operator", "==")
                value = filter_config.get("value")
                
                if operator == ">":
                    df_filtered = df_filtered[df_filtered[column] > value]
                elif operator == ">=":
                    df_filtered = df_filtered[df_filtered[column] >= value]
                elif operator == "<":
                    df_filtered = df_filtered[df_filtered[column] < value]
                elif operator == "<=":
                    df_filtered = df_filtered[df_filtered[column] <= value]
                elif operator == "==":
                    df_filtered = df_filtered[df_filtered[column] == value]
                elif operator == "!=":
                    df_filtered = df_filtered[df_filtered[column] != value]
                elif operator == "contains":
                    df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(str(value), na=False)]
                
                applied_filters.append(f"{column} {operator} {value}")
            
            summary = "FILTRADO DE DATOS\n"
            summary += "="*20 + "\n"
            summary += f"Archivo: {file_path}\n"
            summary += f"Filtros aplicados:\n"
            for filter_desc in applied_filters:
                summary += f"   - {filter_desc}\n"
            
            summary += f"\nRESULTADOS:\n"
            summary += f"   - Filas originales: {len(df)}\n"
            summary += f"   - Filas despues del filtro: {len(df_filtered)}\n"
            summary += f"   - Filas filtradas: {len(df) - len(df_filtered)}\n"
            
            if output_path:
                df_filtered.to_csv(output_path, index=False)
                summary += f"\nDatos filtrados guardados en: {output_path}\n"
            
            return CallToolResult(content=[TextContent(type="text", text=summary)])
        
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Herramienta desconocida: {name}")]
            )
    
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error ejecutando {name}: {str(e)}")]
        )


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())