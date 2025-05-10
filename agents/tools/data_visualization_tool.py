# agents/tools/data_visualization_tool.py
from .base_tool import Tool, ToolException
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv
import re
from typing import List, Dict, Any, Union, Optional
import base64
from io import BytesIO

class DataVisualizationTool(Tool):
    name = "visualize_data"
    description = ("Creates data visualizations (charts, graphs) from provided data. "
                   "The tool can generate bar charts, line graphs, pie charts, scatter plots, "
                   "and other common data visualizations to help illustrate patterns and trends.")
    argument_schema = {
        "data_source": "str: Path to a data file (.csv, .json), or a JSON string of data, or 'example' for sample data",
        "chart_type": "str: Type of visualization ('bar', 'line', 'pie', 'scatter', 'histogram')",
        "title": "str: Title for the visualization",
        "x_column": "str (optional): Name of the column to use for x-axis (required for bar, line, scatter)",
        "y_column": "str (optional): Name of the column to use for y-axis (required for bar, line, scatter)",
        "color": "str (optional): Color name or hex code for the visualization",
        "output_path": "str (optional): Path to save the visualization",
        "width": "int (optional): Width of the visualization in inches (default: 10)",
        "height": "int (optional): Height of the visualization in inches (default: 6)",
        "additional_options": "dict (optional): Additional options specific to the chart type"
    }

    def __init__(self, quartermaster=None):
        super().__init__()
        self.quartermaster = quartermaster  # For reading files
        self.output_dir = "visualizations"
        # Create output directory if quartermaster is provided
        if self.quartermaster:
            try:
                # This will create the directory if it doesn't exist
                self.quartermaster.list_files(self.output_dir)
            except:
                # Make sure the directory exists
                os.makedirs(os.path.join(self.quartermaster.output_dir, self.output_dir), exist_ok=True)

    def _load_data(self, data_source: str) -> Dict[str, Any]:
        """
        Load data from the specified source.
        """
        try:
            # Sample data for testing
            if data_source.lower() == 'example':
                return {
                    'columns': ['Category', 'Value'],
                    'data': [
                        ['A', 5],
                        ['B', 10],
                        ['C', 15],
                        ['D', 8],
                        ['E', 12]
                    ]
                }
            
            # Check if data_source is a JSON string
            if data_source.strip().startswith('{') or data_source.strip().startswith('['):
                try:
                    # Parse JSON string
                    parsed_data = json.loads(data_source)
                    
                    # Handle different JSON structures
                    if isinstance(parsed_data, list):
                        # List of dictionaries
                        if parsed_data and isinstance(parsed_data[0], dict):
                            columns = list(parsed_data[0].keys())
                            data = [[item.get(col, None) for col in columns] for item in parsed_data]
                            return {'columns': columns, 'data': data}
                        # List of lists
                        elif parsed_data and isinstance(parsed_data[0], list):
                            return {'columns': [f'Column{i+1}' for i in range(len(parsed_data[0]))], 'data': parsed_data}
                    
                    # Dictionary with column names and data arrays
                    elif isinstance(parsed_data, dict):
                        if 'columns' in parsed_data and 'data' in parsed_data:
                            return parsed_data
                        else:
                            # Convert dictionary of arrays to columns/data format
                            columns = list(parsed_data.keys())
                            # Find the length of the first array
                            first_array = list(parsed_data.values())[0]
                            if not isinstance(first_array, list):
                                return {'columns': columns, 'data': [[col, parsed_data[col]] for col in columns]}
                            
                            data_length = len(first_array)
                            data = []
                            for i in range(data_length):
                                row = [parsed_data[col][i] if i < len(parsed_data[col]) else None for col in columns]
                                data.append(row)
                            return {'columns': columns, 'data': data}
                    
                    raise ValueError("Unrecognized JSON data structure")
                    
                except json.JSONDecodeError:
                    # Not valid JSON, might be a file path
                    pass
            
            # If quartermaster is available, try to load file
            if self.quartermaster:
                file_content = self.quartermaster.read_file(data_source)
                
                # Process based on file extension
                if data_source.lower().endswith('.csv'):
                    # Parse CSV
                    csv_data = list(csv.reader(file_content.splitlines()))
                    if not csv_data:
                        raise ValueError("CSV file is empty")
                    
                    columns = csv_data[0]
                    data = csv_data[1:]
                    
                    # Convert numeric strings to numbers
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            if re.match(r'^-?\d+\.?\d*$', data[i][j]):
                                try:
                                    data[i][j] = float(data[i][j])
                                    if data[i][j].is_integer():
                                        data[i][j] = int(data[i][j])
                                except ValueError:
                                    pass  # Keep as string if conversion fails
                    
                    return {'columns': columns, 'data': data}
                
                elif data_source.lower().endswith('.json'):
                    # Parse JSON file
                    json_data = json.loads(file_content)
                    
                    # Handle different JSON structures (same as above)
                    if isinstance(json_data, list):
                        if json_data and isinstance(json_data[0], dict):
                            columns = list(json_data[0].keys())
                            data = [[item.get(col, None) for col in columns] for item in json_data]
                            return {'columns': columns, 'data': data}
                        elif json_data and isinstance(json_data[0], list):
                            return {'columns': [f'Column{i+1}' for i in range(len(json_data[0]))], 'data': json_data}
                    
                    elif isinstance(json_data, dict):
                        if 'columns' in json_data and 'data' in json_data:
                            return json_data
                        else:
                            columns = list(json_data.keys())
                            first_array = list(json_data.values())[0]
                            if not isinstance(first_array, list):
                                return {'columns': columns, 'data': [[col, json_data[col]] for col in columns]}
                            
                            data_length = len(first_array)
                            data = []
                            for i in range(data_length):
                                row = [json_data[col][i] if i < len(json_data[col]) else None for col in columns]
                                data.append(row)
                            return {'columns': columns, 'data': data}
                    
                    raise ValueError("Unrecognized JSON structure in file")
                
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
            else:
                raise ValueError("Cannot read file: quartermaster not available")
        
        except Exception as e:
            raise ValueError(f"Error loading data from {data_source}: {str(e)}")
    
    def _get_column_index(self, columns: List[str], column_name: str) -> int:
        """Get the index of a column by name"""
        try:
            return columns.index(column_name)
        except ValueError:
            # If exact match fails, try case-insensitive match
            for i, col in enumerate(columns):
                if col.lower() == column_name.lower():
                    return i
            raise ValueError(f"Column '{column_name}' not found in data")
    
    def _prepare_figure(self, width: int, height: int, title: str) -> plt.Figure:
        """Prepare a matplotlib figure with the specified dimensions and title"""
        plt.figure(figsize=(width, height))
        plt.title(title)
        return plt.gcf()
    
    def _save_figure(self, fig: plt.Figure, output_path: str) -> str:
        """Save the figure to the specified path"""
        if not output_path:
            timestamp = self._get_timestamp()
            output_path = f"{self.output_dir}/visualization_{timestamp}.png"
        
        # Ensure the output path has the correct extension
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            output_path += '.png'
        
        # If path is not absolute and quartermaster is available, save to output directory
        if not os.path.isabs(output_path) and self.quartermaster:
            full_path = os.path.join(self.quartermaster.output_dir, output_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path, bbox_inches='tight')
            return output_path
        else:
            # Save directly to the specified path
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            return output_path
    
    def _get_timestamp(self) -> str:
        """Get a timestamp string for file naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_bar_chart(self, data_obj: Dict[str, Any], x_column: str, y_column: str, 
                         color: str, additional_options: Dict[str, Any]) -> None:
        """Create a bar chart"""
        columns = data_obj['columns']
        data = data_obj['data']
        
        x_idx = self._get_column_index(columns, x_column)
        y_idx = self._get_column_index(columns, y_column)
        
        x_values = [row[x_idx] for row in data]
        y_values = [row[y_idx] for row in data]
        
        # Convert y_values to numeric
        y_values = [float(y) if isinstance(y, (int, float)) or (isinstance(y, str) and y.replace('.', '', 1).isdigit()) else 0 for y in y_values]
        
        plt.bar(x_values, y_values, color=color)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        # Apply additional options
        if additional_options:
            if 'horizontal' in additional_options and additional_options['horizontal']:
                plt.figure(figsize=plt.gcf().get_size_inches())  # Create a new figure of the same size
                plt.barh(x_values, y_values, color=color)
                plt.xlabel(y_column)
                plt.ylabel(x_column)
            
            if 'grid' in additional_options and additional_options['grid']:
                plt.grid(True, linestyle='--', alpha=0.7)
    
    def _create_line_chart(self, data_obj: Dict[str, Any], x_column: str, y_column: str, 
                          color: str, additional_options: Dict[str, Any]) -> None:
        """Create a line chart"""
        columns = data_obj['columns']
        data = data_obj['data']
        
        x_idx = self._get_column_index(columns, x_column)
        y_idx = self._get_column_index(columns, y_column)
        
        x_values = [row[x_idx] for row in data]
        y_values = [row[y_idx] for row in data]
        
        # Convert y_values to numeric
        y_values = [float(y) if isinstance(y, (int, float)) or (isinstance(y, str) and y.replace('.', '', 1).isdigit()) else 0 for y in y_values]
        
        # Sort by x values if they are numeric
        if all(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) for x in x_values):
            x_numeric = [float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else 0 for x in x_values]
            sorted_pairs = sorted(zip(x_numeric, y_values))
            x_values = [x for x, _ in sorted_pairs]
            y_values = [y for _, y in sorted_pairs]
        
        plt.plot(x_values, y_values, color=color, marker='o')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        # Apply additional options
        if additional_options:
            if 'marker' in additional_options:
                plt.plot(x_values, y_values, color=color, marker=additional_options['marker'])
            
            if 'grid' in additional_options and additional_options['grid']:
                plt.grid(True, linestyle='--', alpha=0.7)
    
    def _create_pie_chart(self, data_obj: Dict[str, Any], x_column: str, y_column: str, 
                         additional_options: Dict[str, Any]) -> None:
        """Create a pie chart"""
        columns = data_obj['columns']
        data = data_obj['data']
        
        # For pie charts, x_column represents labels, y_column represents values
        x_idx = self._get_column_index(columns, x_column)
        y_idx = self._get_column_index(columns, y_column)
        
        labels = [str(row[x_idx]) for row in data]
        values = [row[y_idx] for row in data]
        
        # Convert values to numeric
        values = [float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) else 0 for v in values]
        
        # Generate colors
        default_colors = plt.cm.tab10(np.arange(len(labels)) % 10)
        
        # Create the pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=default_colors)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        
        # Apply additional options
        if additional_options:
            if 'donut' in additional_options and additional_options['donut']:
                # Create a donut chart by adding a white circle in the middle
                centre_circle = plt.Circle((0, 0), 0.7, color='white', fc='white', linewidth=0)
                plt.gca().add_artist(centre_circle)
    
    def _create_scatter_plot(self, data_obj: Dict[str, Any], x_column: str, y_column: str, 
                            color: str, additional_options: Dict[str, Any]) -> None:
        """Create a scatter plot"""
        columns = data_obj['columns']
        data = data_obj['data']
        
        x_idx = self._get_column_index(columns, x_column)
        y_idx = self._get_column_index(columns, y_column)
        
        x_values = [row[x_idx] for row in data]
        y_values = [row[y_idx] for row in data]
        
        # Convert values to numeric
        x_values = [float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else 0 for x in x_values]
        y_values = [float(y) if isinstance(y, (int, float)) or (isinstance(y, str) and y.replace('.', '', 1).isdigit()) else 0 for y in y_values]
        
        # Create scatter plot
        plt.scatter(x_values, y_values, color=color)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        # Apply additional options
        if additional_options:
            if 'size' in additional_options:
                size = additional_options['size']
                if isinstance(size, (int, float)):
                    plt.scatter(x_values, y_values, color=color, s=size)
                elif size in columns:
                    size_idx = self._get_column_index(columns, size)
                    size_values = [row[size_idx] for row in data]
                    size_values = [float(s) if isinstance(s, (int, float)) or (isinstance(s, str) and s.replace('.', '', 1).isdigit()) else 10 for s in size_values]
                    plt.scatter(x_values, y_values, color=color, s=size_values)
            
            if 'grid' in additional_options and additional_options['grid']:
                plt.grid(True, linestyle='--', alpha=0.7)
    
    def _create_histogram(self, data_obj: Dict[str, Any], x_column: str, 
                         color: str, additional_options: Dict[str, Any]) -> None:
        """Create a histogram"""
        columns = data_obj['columns']
        data = data_obj['data']
        
        x_idx = self._get_column_index(columns, x_column)
        
        values = [row[x_idx] for row in data]
        
        # Convert values to numeric
        values = [float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) else 0 for v in values]
        
        # Set default bins
        bins = 10
        if additional_options and 'bins' in additional_options:
            bins = additional_options['bins']
        
        plt.hist(values, bins=bins, color=color, alpha=0.7)
        plt.xlabel(x_column)
        plt.ylabel('Frequency')
        
        # Apply additional options
        if additional_options:
            if 'grid' in additional_options and additional_options['grid']:
                plt.grid(True, linestyle='--', alpha=0.7)
    
    def execute(self, **kwargs) -> str:
        # Extract and validate parameters
        data_source = kwargs.get("data_source")
        chart_type = kwargs.get("chart_type")
        title = kwargs.get("title", "Data Visualization")
        x_column = kwargs.get("x_column")
        y_column = kwargs.get("y_column")
        color = kwargs.get("color", "blue")
        output_path = kwargs.get("output_path", "")
        width = kwargs.get("width", 10)
        height = kwargs.get("height", 6)
        additional_options = kwargs.get("additional_options", {})
        
        # Validate required parameters
        if not data_source:
            raise ToolException("DataVisualizationTool: 'data_source' parameter is required.")
        if not chart_type:
            raise ToolException("DataVisualizationTool: 'chart_type' parameter is required.")
        
        # Validate chart type
        valid_chart_types = ['bar', 'line', 'pie', 'scatter', 'histogram']
        if chart_type.lower() not in valid_chart_types:
            raise ToolException(f"DataVisualizationTool: 'chart_type' must be one of {valid_chart_types}.")
        
        try:
            # Load the data
            data_obj = self._load_data(data_source)
            
            # Validate column parameters based on chart type
            if chart_type.lower() in ['bar', 'line', 'scatter']:
                if not x_column or not y_column:
                    raise ToolException(f"DataVisualizationTool: '{chart_type}' chart requires both 'x_column' and 'y_column' parameters.")
            elif chart_type.lower() == 'pie':
                if not x_column or not y_column:
                    raise ToolException("DataVisualizationTool: 'pie' chart requires both 'x_column' (for labels) and 'y_column' (for values) parameters.")
            elif chart_type.lower() == 'histogram':
                if not x_column:
                    raise ToolException("DataVisualizationTool: 'histogram' chart requires 'x_column' parameter.")
            
            # Prepare the figure
            fig = self._prepare_figure(width, height, title)
            
            # Create the specific type of chart
            if chart_type.lower() == 'bar':
                self._create_bar_chart(data_obj, x_column, y_column, color, additional_options)
            elif chart_type.lower() == 'line':
                self._create_line_chart(data_obj, x_column, y_column, color, additional_options)
            elif chart_type.lower() == 'pie':
                self._create_pie_chart(data_obj, x_column, y_column, additional_options)
            elif chart_type.lower() == 'scatter':
                self._create_scatter_plot(data_obj, x_column, y_column, color, additional_options)
            elif chart_type.lower() == 'histogram':
                self._create_histogram(data_obj, x_column, color, additional_options)
            
            # Save the figure
            saved_path = self._save_figure(fig, output_path)
            
            # Include a base64 version of the image if quartermaster is available
            image_data = ""
            if self.quartermaster:
                # Save to a BytesIO object
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                
                # Encode as base64
                image_data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                # Save the base64 data to a file for embedded use
                b64_path = saved_path.rsplit('.', 1)[0] + '_b64.txt'
                self.quartermaster.write_file(b64_path, image_data)
                
                # Return success message with both paths
                return (f"Visualization created successfully!\n"
                        f"Saved to: {saved_path}\n"
                        f"Base64 data saved to: {b64_path}\n"
                        f"To use the base64 data in HTML or Markdown, you can use: \n"
                        f"<img src=\"data:image/png;base64,{image_data[:20]}...\" />")
            else:
                plt.close(fig)
                return f"Visualization created successfully! Saved to: {saved_path}"
        
        except ValueError as ve:
            plt.close()
            raise ToolException(f"DataVisualizationTool: {str(ve)}")
        except Exception as e:
            plt.close()
            raise ToolException(f"DataVisualizationTool: Failed to create visualization. Error: {str(e)}")