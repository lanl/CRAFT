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
        'Parameter_loc': [metadata.find('parameter_loc').text]
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
        'Var': ['Settings_File', 'SaveName', 'thres', 'EmulatorDrive', 
                'FATES_samples', 'model_type', 'nn_config'],
        'Path': [
            metadata.find('settings_file').text,
            metadata.find('save_name').text,
            metadata.find('thres').text,
            metadata.find('emulator_drive').text,
            metadata.find('fates_samples').text,
            metadata.find('model_type').text,
            metadata.find('nn_config').text
        ],
        'Explanation': [
            '',
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
        data.append({
            '': i,
            'var_name': var.find('var_name').text,
            'FATESruns': var.find('fates_runs').text,
            'xs': int(var.find('xs').text),
            'xe': int(var.find('xe').text),
            'y_i': int(var.find('y_i').text),
            'Scaler': float(var.find('scaler').text)
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
            'samples_file', 'obsfile'
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
            settings.find('obsfile').text
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
