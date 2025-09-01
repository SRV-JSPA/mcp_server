import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from scipy import stats
import warnings

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  

mcp = FastMCP("CSV Analysis Server")

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
            "dtypes": df.dtypes.to_dict(),
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
                stats_dict["numeric_summary"][col].update({
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "variance": df[col].var()
                })
        
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            stats_dict["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "unique_values": df[col].unique().tolist()[:10],  
                "value_counts": df[col].value_counts().head().to_dict(),
                "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None
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
                outliers_info[col]["outliers"] = df[outlier_mask][col].tolist()
                outliers_info[col]["bounds"] = {"lower": lower_bound, "upper": upper_bound}
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > 3
                outliers_info[col]["outliers"] = df[col][outlier_mask].tolist()
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
            return {"error": "No hay columnas numéricas para calcular correlaciones"}
        
        corr_matrix = numeric_df.corr()
        
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append({
                    "variables": [col1, col2],
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value)
                })
        
        
        corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": corr_pairs[:5],  
            "weak_correlations": corr_pairs[-5:]   
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
                        plt.title(f'Gráfico de dispersión: {col_x} vs {col_y}')
                
            elif plot_type == "correlation_heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
                    plt.title('Matriz de Correlación')
                    plt.tight_layout()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"Visualización guardada en: {save_path}"
            
        except Exception as e:
            plt.close()
            return f"Error creando visualización: {str(e)}"



@mcp.tool()
def analyze_csv(file_path: str) -> str:
    try:
        df = CSVAnalyzer.load_csv(file_path)
        stats = CSVAnalyzer.generate_descriptive_stats(df)
        
        
        summary = f"""
ANÁLISIS COMPLETO DEL ARCHIVO: {file_path}

INFORMACIÓN GENERAL:
- Dimensiones: {stats['shape'][0]} filas × {stats['shape'][1]} columnas
- Memoria utilizada: {stats['memory_usage']}
- Columnas: {', '.join(stats['columns'])}

COLUMNAS NUMÉRICAS:
"""
        
        if stats['numeric_summary']:
            for col, col_stats in stats['numeric_summary'].items():
                summary += f"\n  • {col}:"
                summary += f"\n    - Promedio: {col_stats['mean']:.2f}"
                summary += f"\n    - Mediana: {col_stats['50%']:.2f}"
                summary += f"\n    - Desviación estándar: {col_stats['std']:.2f}"
                summary += f"\n    - Rango: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]"
                summary += f"\n    - Asimetría: {col_stats['skewness']:.2f}"
        
        summary += f"\n\nCOLUMNAS CATEGÓRICAS:"
        if stats['categorical_summary']:
            for col, col_stats in stats['categorical_summary'].items():
                summary += f"\n  • {col}:"
                summary += f"\n    - Valores únicos: {col_stats['unique_count']}"
                summary += f"\n    - Valor más frecuente: {col_stats['mode']}"
        
        summary += f"\n\nVALORES FALTANTES:"
        missing_total = sum(stats['missing_values'].values())
        if missing_total > 0:
            for col, missing_count in stats['missing_values'].items():
                if missing_count > 0:
                    percentage = (missing_count / stats['shape'][0]) * 100
                    summary += f"\n  • {col}: {missing_count} ({percentage:.1f}%)"
        else:
            summary += "\n No hay valores faltantes"
        
        return summary
        
    except Exception as e:
        return f"Error analizando CSV: {str(e)}"

@mcp.tool()
def detect_outliers_in_csv(file_path: str, method: str = "iqr", column: str = None) -> str:
    try:
        df = CSVAnalyzer.load_csv(file_path)
        
        if column and column not in df.columns:
            return f"Error: La columna '{column}' no existe en el dataset"
        
        if column:
            df_analyze = df[[column]]
        else:
            df_analyze = df
        
        outliers_info = CSVAnalyzer.detect_outliers(df_analyze, method)
        
        summary = f"DETECCIÓN DE VALORES ATÍPICOS - Método: {method.upper()}\n"
        summary += "="*50 + "\n"
        
        total_outliers = 0
        for col, info in outliers_info.items():
            if info['count'] > 0:
                total_outliers += info['count']
                summary += f"\nColumna: {col}"
                summary += f"\n   - Valores atípicos encontrados: {info['count']}"
                summary += f"\n   - Valores: {info['outliers'][:10]}"  
                if 'bounds' in info:
                    summary += f"\n   - Límites: [{info['bounds']['lower']:.2f}, {info['bounds']['upper']:.2f}]"
                if len(info['outliers']) > 10:
                    summary += f"\n   - (y {len(info['outliers']) - 10} más...)"
                summary += "\n"
        
        if total_outliers == 0:
            summary += "\nNo se encontraron valores atípicos significativos"
        else:
            summary += f"\nTotal de valores atípicos: {total_outliers}"
        
        return summary
        
    except Exception as e:
        return f"Error detectando outliers: {str(e)}"

@mcp.tool()
def clean_csv_data(file_path: str, operations: List[str], output_path: str = None) -> str:
    try:
        df_original = CSVAnalyzer.load_csv(file_path)
        df_clean = CSVAnalyzer.clean_data(df_original, operations)
        
        
        summary = "PROCESO DE LIMPIEZA DE DATOS\n"
        summary += "="*40 + "\n"
        summary += f"Archivo original: {file_path}\n"
        summary += f"Operaciones aplicadas: {', '.join(operations)}\n\n"
        
        summary += "CAMBIOS REALIZADOS:\n"
        summary += f"   - Filas antes: {len(df_original)}\n"
        summary += f"   - Filas después: {len(df_clean)}\n"
        summary += f"   - Filas eliminadas: {len(df_original) - len(df_clean)}\n\n"
        
        
        missing_before = df_original.isnull().sum().sum()
        missing_after = df_clean.isnull().sum().sum()
        summary += f"   - Valores faltantes antes: {missing_before}\n"
        summary += f"   - Valores faltantes después: {missing_after}\n"
        summary += f"   - Valores faltantes corregidos: {missing_before - missing_after}\n\n"
        
        
        if output_path:
            df_clean.to_csv(output_path, index=False)
            summary += f"Archivo limpiado guardado en: {output_path}\n"
        
        return summary
        
    except Exception as e:
        return f"Error limpiando datos: {str(e)}"

@mcp.tool()
def calculate_correlations_csv(file_path: str) -> str:
    try:
        df = CSVAnalyzer.load_csv(file_path)
        corr_info = CSVAnalyzer.calculate_correlations(df)
        
        if "error" in corr_info:
            return corr_info["error"]
        
        summary = "ANÁLISIS DE CORRELACIONES\n"
        summary += "="*30 + "\n\n"
        
        summary += "CORRELACIONES MÁS FUERTES:\n"
        for i, corr in enumerate(corr_info["strong_correlations"][:5], 1):
            summary += f"{i}. {corr['variables'][0]} ↔ {corr['variables'][1]}: "
            summary += f"{corr['correlation']:.3f}\n"
        
        summary += "\nCORRELACIONES MÁS DÉBILES:\n"
        for i, corr in enumerate(corr_info["weak_correlations"][-5:], 1):
            summary += f"{i}. {corr['variables'][0]} ↔ {corr['variables'][1]}: "
            summary += f"{corr['correlation']:.3f}\n"
        
        summary += "\nINTERPRETACIÓN:\n"
        summary += "  • |r| > 0.8: Correlación muy fuerte\n"
        summary += "  • |r| > 0.6: Correlación fuerte\n"
        summary += "  • |r| > 0.4: Correlación moderada\n"
        summary += "  • |r| < 0.2: Correlación débil\n"
        
        return summary
        
    except Exception as e:
        return f"Error calculando correlaciones: {str(e)}"

@mcp.tool()
def create_csv_visualization(file_path: str, plot_type: str, columns: List[str], 
                           save_path: str = None) -> str:
    try:
        df = CSVAnalyzer.load_csv(file_path)
        
        
        if not save_path:
            base_name = Path(file_path).stem
            save_path = f"{base_name}_{plot_type}.png"
        
        result = CSVAnalyzer.create_visualization(df, plot_type, columns, save_path)
        
        summary = f"VISUALIZACIÓN CREADA\n"
        summary += "="*25 + "\n"
        summary += f"Tipo: {plot_type}\n"
        summary += f"Datos: {file_path}\n"
        summary += f"Columnas: {', '.join(columns)}\n"
        summary += f"Guardado en: {save_path}\n\n"
        summary += result
        
        return summary
        
    except Exception as e:
        return f"Error creando visualización: {str(e)}"

@mcp.tool()
def filter_csv_data(file_path: str, filters: Dict[str, Any], output_path: str = None) -> str:
    try:
        df = CSVAnalyzer.load_csv(file_path)
        df_filtered = df.copy()
        
        applied_filters = []
        
        for column, filter_config in filters.items():
            if column not in df.columns:
                return f"Error: La columna '{column}' no existe en el dataset"
            
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
        summary += f"   - Filas después del filtro: {len(df_filtered)}\n"
        summary += f"   - Filas filtradas: {len(df) - len(df_filtered)}\n"
        
        if output_path:
            df_filtered.to_csv(output_path, index=False)
            summary += f"\nDatos filtrados guardados en: {output_path}\n"
        
        return summary
        
    except Exception as e:
        return f"Error filtrando datos: {str(e)}"

async def main():    
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())