#!/usr/bin/env python
from config_utils import get_emulator_settings
import pandas as pd

# Test the updated get_emulator_settings function
print("Testing get_emulator_settings with list format...")
settings = get_emulator_settings()
print("\nEmulator Settings DataFrame:")
print(settings)
print("\nNumber of variables:", len(settings))
print("\nColumns:", settings.columns.tolist())
print("\nDatatypes:")
print(settings.dtypes)
