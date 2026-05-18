"""
Utility functions for reading configuration from XML file.
"""
import xml.etree.ElementTree as ET
import pandas as pd

def read_xml_config(file_path='config.xml'):
    """
    Read the XML configuration file and return the root element.
    
    Args:
        file_path (str): Path to the XML configuration file
        
    Returns:
        ET.Element: Root element of the XML file
    """
    tree = ET.parse(file_path)
    return tree.getroot()

def get_cleaning_metadata():
    """
    Read cleaning metadata from the XML configuration file.
    
    Returns:
        pd.DataFrame: DataFrame containing cleaning metadata
    """
    root = read_xml_config()
    metadata = root.find('cleaning_metadata')
    
    data = {
        'DRIVE': [metadata.find('drive').text],
        'CASE': [metadata.find('case').text],
        'N_runs': [int(metadata.find('n_runs').text)],
        'Varlist': [metadata.find('varlist').text],
        'Parameter_loc': [metadata.find('parameter_loc').text],
        'timestep': [metadata.find('timestep').text]
    }
    
    return pd.DataFrame(data)

def get_emulator_metadata():
    """
    Read emulator metadata from the XML configuration file.
    
    Returns:
        pd.DataFrame: DataFrame containing emulator metadata
    """
    root = read_xml_config()
    metadata = root.find('emulator_metadata')
    
    data = {
        'Var': [ 'SaveName', 'thres', 'EmulatorDrive', 
                'FATES_samples', 'model_type', 'nn_config'],
        'Path': [
            metadata.find('save_name').text,
            metadata.find('thres').text,
            metadata.find('emulator_drive').text,
            metadata.find('fates_samples').text,
            metadata.find('model_type').text,
            metadata.find('nn_config').text,
        ],
        'Explanation': [
            '',
            '',
            '',
            '',
            "Model type: 'rf' for Random Forest or 'nn' for Neural Network",
            "Path to neural network configuration file (used only when model_type='nn')"
        ]
    }
    
    return pd.DataFrame(data).reset_index()

def get_emulator_settings():
    """
    Read emulator settings from the XML configuration file.
    
    Returns:
        pd.DataFrame: DataFrame containing emulator settings
    """
    root = read_xml_config()
    variables = root.find('emulator_settings').findall('variable')
    
    data = []
    for i, var in enumerate(variables):
        # Check if the values are in list format (wrapped in [])
        var_name_text = var.find('var_name').text
        fates_runs_text = var.find('fates_runs').text
        xs_text = var.find('xs').text
        xe_text = var.find('xe').text
        y_i_text = var.find('y_i').text
        scaler_text = var.find('scaler').text
        downsample= var.find('downsample').text
        # If the values are in list format, parse them manually
        if var_name_text.startswith('[') and var_name_text.endswith(']'):
            try:
                # Remove brackets and split by comma
                var_names = [name.strip() for name in var_name_text[1:-1].split(',')]
                fates_runs = [run.strip() for run in fates_runs_text[1:-1].split(',')]
                xs_values = [int(x.strip()) for x in xs_text[1:-1].split(',')]
                xe_values = [int(x.strip()) for x in xe_text[1:-1].split(',')]
                y_i_values = [int(y.strip()) for y in y_i_text[1:-1].split(',')]
                scaler_values = [float(s.strip()) for s in scaler_text[1:-1].split(',')]
                downsample = [float(s.strip()) for s in downsample[1:-1].split(',')]
                # Create a DataFrame row for each variable in the lists
                for j in range(len(var_names)):
                    data.append({
                        '': len(data),
                        'var_name': var_names[j],
                        'FATESruns': fates_runs[j],
                        'xs': xs_values[j],
                        'xe': xe_values[j],
                        'y_i': y_i_values[j],
                        'Scaler': scaler_values[j],
                        "downsample":downsample[j]
                    })
            except (ValueError, IndexError) as e:
                print(f"Error parsing list values: {e}")
                # Fall back to single value format
                data.append({
                    '': i,
                    'var_name': var_name_text,
                    'FATESruns': fates_runs_text,
                    'xs': int(xs_text),
                    'xe': int(xe_text),
                    'y_i': int(y_i_text),
                    'Scaler': float(scaler_text),
                    'downsample':float(downsample)
                })
        else:
            # Handle single value format
            data.append({
                '': i,
                'var_name': var_name_text,
                'FATESruns': fates_runs_text,
                'xs': int(xs_text),
                'xe': int(xe_text),
                'y_i': int(y_i_text),
                'Scaler': float(scaler_text),
                'downsample':float(downsample)
            })
    
    return pd.DataFrame(data)

def get_bayes_settings():
    """
    Read Bayes settings from the XML configuration file.
    
    Returns:
        pd.DataFrame: DataFrame containing Bayes settings
    """
    root = read_xml_config()
    settings = root.find('bayes_settings')
    
    data = {
        'parameter': [
            'initialpos', 'steps', 'adapt_interval', 'burnin',
            'Model_List', 'Scaler_List', 'Obs_List', 'list1',
            'samples_file', 'obsfile', 'output_file', 'sampler_method',
            'dream_n_chains', 'dream_delta', 'dream_c', 'dream_c_star',
            'dream_n_crossover', 'dream_p_gamma_unity', 'dream_thin'
        ],
        'value': [
            settings.find('initialpos').text,
            settings.find('steps').text,
            settings.find('adapt_interval').text,
            settings.find('burnin').text,
            settings.find('model_list').text,
            settings.find('scaler_list').text,
            settings.find('obs_list').text,
            settings.find('list1').text,
            settings.find('samples_file').text,
            settings.find('obsfile').text,
            settings.find('output_file').text,
            settings.find('sampler_method').text,
            settings.find('dream_n_chains').text,
            settings.find('dream_delta').text,
            settings.find('dream_c').text,
            settings.find('dream_c_star').text,
            settings.find('dream_n_crossover').text,
            settings.find('dream_p_gamma_unity').text,
            settings.find('dream_thin').text
        ]
    }
    
    return pd.DataFrame(data).set_index('parameter')

def get_nn_config():
    """
    Read neural network configuration from the XML configuration file.
    
    Returns:
        pd.DataFrame: DataFrame containing neural network configuration
    """
    root = read_xml_config()
    nn_config = root.find('nn_config')
    
    data = {
        'layer_sizes': [nn_config.find('layer_sizes').text],
        'activation': [nn_config.find('activation').text],
        'optimizer': [nn_config.find('optimizer').text],
        'loss': [nn_config.find('loss').text],
        'epochs': [int(nn_config.find('epochs').text)],
        'batch_size': [int(nn_config.find('batch_size').text)]
    }
    
    return pd.DataFrame(data)
