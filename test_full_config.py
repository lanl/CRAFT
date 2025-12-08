#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
from config_utils import (
    get_cleaning_metadata,
    get_emulator_metadata,
    get_emulator_settings,
    get_bayes_settings,
    get_nn_config
)

def test_all_config():
    """Test all configuration reading functions"""
    print("\n===== Testing Cleaning Metadata =====")
    cleaning_meta = get_cleaning_metadata()
    print(cleaning_meta)
    
    print("\n===== Testing Emulator Metadata =====")
    emulator_meta = get_emulator_metadata()
    print(emulator_meta)
    
    print("\n===== Testing Emulator Settings =====")
    emulator_settings = get_emulator_settings()
    print(emulator_settings)
    
    print("\n===== Testing Bayes Settings =====")
    bayes_settings = get_bayes_settings()
    print(bayes_settings)
    
    print("\n===== Testing Neural Network Config =====")
    nn_config = get_nn_config()
    print(nn_config)
    
    # Test that we can iterate through multiple variables in emulator settings
    print("\n===== Testing Iteration Through Multiple Variables =====")
    for i in range(len(emulator_settings)):
        var_name = emulator_settings.loc[i, "var_name"]
        fates_runs = emulator_settings.loc[i, "FATESruns"]
        print(f"Variable {i+1}: {var_name}, File: {fates_runs}")

if __name__ == "__main__":
    test_all_config()
