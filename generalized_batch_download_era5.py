#!/usr/bin/env python3
"""
ERA5 Generalized Batch Data Download Script

Downloads ERA5 pressure level variables in 3-year batches based on user input.
Prompts user for API key, variables, pressure levels, and year range.
Each file contains 3 years of data for a specific variable and pressure level.

Usage:
    python generalized_batch_download_era5.py
"""

import sys
import cdsapi
import os
from pathlib import Path
import time
import getpass

# Add debug print function that flushes immediately
def debug_print(message):
    print(message)
    sys.stdout.flush()

debug_print("SCRIPT STARTED - Generalized ERA5 Batch Download")

def get_user_input():
    """Get all user input for the download."""
    debug_print("\n" + "="*60)
    debug_print("ERA5 GENERALIZED BATCH DOWNLOAD CONFIGURATION")
    debug_print("="*60)
    
    # Get API key
    print("\n1. CDS API Configuration:")
    print("   You need a CDS API key from https://cds.climate.copernicus.eu/api-how-to")
    print("   IMPORTANT: Use only the API key part (after the colon), NOT the full UID:key format")
    print("   Example: If your key shows '12345:abcd-efgh-1234-5678', use only 'abcd-efgh-1234-5678'")
    api_key = getpass.getpass("Enter your CDS API key (input hidden): ").strip()
    if not api_key:
        print("Error: API key is required!")
        sys.exit(1)
    
    # Clean API key - remove UID prefix if present
    if ':' in api_key:
        print("   Detected UID:key format, extracting key part only...")
        api_key = api_key.split(':', 1)[1]
        print("   Key format corrected.")
    
    # Get variables
    print("\n2. Variable Selection:")
    print("Available variables:")
    available_vars = {
        '1': 'u_component_of_wind',
        '2': 'v_component_of_wind', 
        '3': 'temperature',
        '4': 'specific_humidity',
        '5': 'relative_humidity',
        '6': 'geopotential',
        '7': 'vertical_velocity'
    }
    
    for key, var in available_vars.items():
        print(f"   {key}. {var}")
    print("   a. All variables")
    print("   c. Custom variable name")
    
    var_choice = input("\nSelect variables (comma-separated numbers, 'a' for all, or 'c' for custom): ").strip()
    
    if var_choice.lower() == 'a':
        variables = list(available_vars.values())
    elif var_choice.lower() == 'c':
        custom_vars = input("Enter custom variable names (comma-separated): ").strip()
        variables = [var.strip() for var in custom_vars.split(',') if var.strip()]
    else:
        selected_nums = [num.strip() for num in var_choice.split(',')]
        variables = []
        for num in selected_nums:
            if num in available_vars:
                variables.append(available_vars[num])
            else:
                print(f"Warning: Invalid selection '{num}' ignored")
    
    if not variables:
        print("Error: At least one variable must be selected!")
        sys.exit(1)
    
    # Get pressure levels
    print("\n3. Pressure Level Selection:")
    print("Common pressure levels (hPa):")
    print("   1. 50   2. 100  3. 150  4. 200  5. 250")
    print("   6. 300  7. 400  8. 500  9. 600  10. 700")
    print("   11. 850 12. 925 13. 1000")
    print("   a. All levels")
    print("   c. Custom levels")
    
    available_levels = {
        '1': '50', '2': '100', '3': '150', '4': '200', '5': '250',
        '6': '300', '7': '400', '8': '500', '9': '600', '10': '700',
        '11': '850', '12': '925', '13': '1000'
    }
    
    level_choice = input("\nSelect pressure levels (comma-separated numbers, 'a' for all, or 'c' for custom): ").strip()
    
    if level_choice.lower() == 'a':
        pressure_levels = list(available_levels.values())
    elif level_choice.lower() == 'c':
        custom_levels = input("Enter pressure levels in hPa (comma-separated): ").strip()
        pressure_levels = [level.strip() for level in custom_levels.split(',') if level.strip()]
    else:
        selected_nums = [num.strip() for num in level_choice.split(',')]
        pressure_levels = []
        for num in selected_nums:
            if num in available_levels:
                pressure_levels.append(available_levels[num])
            else:
                print(f"Warning: Invalid selection '{num}' ignored")
    
    if not pressure_levels:
        print("Error: At least one pressure level must be selected!")
        sys.exit(1)
    
    # Get year range
    print("\n4. Year Range Selection:")
    while True:
        try:
            start_year = int(input("Enter start year (e.g., 1995): ").strip())
            end_year = int(input("Enter end year (e.g., 2024): ").strip())
            
            if start_year > end_year:
                print("Error: Start year must be <= end year!")
                continue
            if start_year < 1940:
                print("Warning: ERA5 data starts from 1940")
            if end_year > 2024:
                print("Warning: ERA5 data may not be available for future years")
            
            break
        except ValueError:
            print("Error: Please enter valid years!")
    
    return api_key, variables, pressure_levels, start_year, end_year

def create_year_batches(start_year, end_year):
    """Create 3-year batches from the given year range."""
    batches = []
    current_year = start_year
    
    while current_year <= end_year:
        batch_end = min(current_year + 2, end_year)  # 3-year batches
        batches.append((current_year, batch_end))
        current_year = batch_end + 1
    
    return batches

def setup_cds_config(api_key):
    """Setup CDS API configuration with provided API key."""
    debug_print("Setting up CDS API configuration...")
    try:
        cds_config_dir = Path.home() / '.cdsapirc'
        
        # Create config content
        config_content = f"""url: https://cds.climate.copernicus.eu/api
key: {api_key}
"""
        
        # Write config file
        with open(cds_config_dir, 'w') as f:
            f.write(config_content)
        
        debug_print(f"CDS API configuration saved to {cds_config_dir}")
        return True
        
    except Exception as e:
        debug_print(f"Error setting up CDS configuration: {str(e)}")
        return False

def download_era5_batch(variable, pressure_level, start_year, end_year):
    """Download ERA5 data for a specific variable, pressure level, and year range."""
    
    try:
        debug_print(f"DEBUG: Creating CDS client...")
        c = cdsapi.Client()
        debug_print(f"DEBUG: CDS client created successfully!")

        # Configure parameters
        years = [str(y) for y in range(start_year, end_year + 1)]

        # Create output filename
        output_filename = f"{variable}{pressure_level}_{start_year}-{end_year}.nc"

        debug_print(f"\n{'='*60}")
        debug_print(f"Starting download...")
        debug_print(f"Variable: {variable}")
        debug_print(f"Pressure Level: {pressure_level} hPa")
        debug_print(f"Years: {start_year}-{end_year}")
        debug_print(f"Output file: {output_filename}")
        debug_print(f"{'='*60}")

        # Check if file already exists
        if os.path.exists(output_filename):
            try:
                file_size = os.path.getsize(output_filename) / (1024 * 1024)  # Convert to MB
                debug_print(f"File {output_filename} already exists ({file_size:.2f} MB). Skipping...")
                return True
            except Exception as e:
                debug_print(f"Warning: Could not check existing file size: {str(e)}")
                debug_print(f"File {output_filename} already exists. Skipping...")
                return True

        debug_print("DEBUG: About to call c.retrieve()...")

        # Download ERA5 pressure level data
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'pressure_level': pressure_level,
                'year': years,
                'month': [f"{m:02d}" for m in range(1, 13)],  # all months
                'day': [f"{d:02d}" for d in range(1, 32)],    # all days
                'time': [f"{h:02d}:00" for h in range(0, 24, 3)],  # every 3 hours
                'format': 'netcdf',
                'area': [47, 55, 0, 105],  # North, West, South, East
            },
            output_filename
        )

        debug_print(f"\nDownload completed successfully!")
        debug_print(f"File saved as: {output_filename}")

        # Check file size
        if os.path.exists(output_filename):
            try:
                file_size = os.path.getsize(output_filename) / (1024 * 1024)  # Convert to MB
                debug_print(f"File size: {file_size:.2f} MB")
            except Exception as e:
                debug_print(f"Warning: Could not determine file size: {str(e)}")

        return True

    except KeyboardInterrupt:
        debug_print(f"\nDownload interrupted by user for {variable} at {pressure_level} hPa")
        return False
    except Exception as e:
        debug_print(f"Error during download: {str(e)}")
        debug_print(f"Failed to download {variable} at {pressure_level} hPa for {start_year}-{end_year}")

        # Additional error details for common issues
        error_str = str(e).lower()
        if "authentication" in error_str or "key" in error_str:
            debug_print("Troubleshooting: Check your CDS API key")
            debug_print("  - Make sure you've accepted the license at https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products")
            debug_print("  - Ensure your API key is correct (without UID: prefix)")
        elif "connection" in error_str or "network" in error_str:
            debug_print("Troubleshooting: Check your internet connection")
        elif "request" in error_str or "404" in error_str or "endpoint" in error_str:
            debug_print("Troubleshooting: API endpoint issue")
            debug_print("  - Make sure your cdsapi library is up to date: pip install --upgrade cdsapi")
            debug_print("  - Check that .cdsapirc uses correct URL: https://cds.climate.copernicus.eu/api")

        return False

def main():
    """Main function to download all variables and pressure levels in batches."""
    
    debug_print("DEBUG: Entering main()")

    try:
        # Get user input
        api_key, variables, pressure_levels, start_year, end_year = get_user_input()
        
        # Setup CDS configuration
        debug_print("DEBUG: About to call setup_cds_config()")
        if not setup_cds_config(api_key):
            debug_print("Failed to setup CDS configuration. Exiting.")
            sys.exit(1)
        debug_print("DEBUG: setup_cds_config() completed")

        # Create year batches
        year_batches = create_year_batches(start_year, end_year)

        # Display configuration summary
        debug_print("\n" + "="*60)
        debug_print("DOWNLOAD CONFIGURATION SUMMARY")
        debug_print("="*60)
        debug_print("Variables to download:")
        for var in variables:
            debug_print(f"  - {var}")
        debug_print(f"Pressure levels: {pressure_levels} hPa")
        debug_print(f"Year range: {start_year}-{end_year}")
        debug_print(f"Year batches: {year_batches}")
        debug_print("="*60)

        # Confirm before starting
        confirm = input("\nProceed with download? (y/n): ").strip().lower()
        if confirm != 'y':
            debug_print("Download cancelled by user.")
            sys.exit(0)

        # Track download statistics
        total_downloads = len(variables) * len(pressure_levels) * len(year_batches)
        successful_downloads = 0
        failed_downloads = 0

        debug_print(f"\nTotal downloads planned: {total_downloads}")
        debug_print("\nStarting batch downloads...\n")

        # Loop through each variable
        for variable in variables:
            debug_print(f"\n{'#'*60}")
            debug_print(f"Processing variable: {variable.upper()}")
            debug_print(f"{'#'*60}")

            # Loop through each pressure level
            for pressure_level in pressure_levels:
                debug_print(f"\n{'-'*40}")
                debug_print(f"Pressure level: {pressure_level} hPa")
                debug_print(f"{'-'*40}")

                # Loop through each year batch
                for i, (batch_start, batch_end) in enumerate(year_batches):
                    try:
                        success = download_era5_batch(variable, pressure_level, batch_start, batch_end)

                        if success:
                            successful_downloads += 1
                        else:
                            failed_downloads += 1

                        # Add a small delay between downloads to be respectful to the API
                        if i < len(year_batches) - 1:  # Don't delay after the last download for this pressure level
                            debug_print("Waiting 30 seconds before next download...")
                            time.sleep(30)

                    except KeyboardInterrupt:
                        debug_print("\n\nDownload process interrupted by user!")
                        debug_print("Partial downloads may have been completed.")
                        raise
                    except Exception as e:
                        debug_print(f"Unexpected error processing {variable} at {pressure_level} hPa: {str(e)}")
                        failed_downloads += 1
                        continue

        # Print final summary
        debug_print(f"\n{'='*60}")
        debug_print("DOWNLOAD SUMMARY")
        debug_print(f"{'='*60}")
        debug_print(f"Total downloads attempted: {total_downloads}")
        debug_print(f"Successful downloads: {successful_downloads}")
        debug_print(f"Failed downloads: {failed_downloads}")
        if total_downloads > 0:
            debug_print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%")

        if failed_downloads > 0:
            debug_print(f"\n{failed_downloads} downloads failed. Check the error messages above.")
            debug_print("You can re-run this script to retry failed downloads.")
        else:
            debug_print("\nAll downloads completed successfully!")

        debug_print(f"{'='*60}")

    except KeyboardInterrupt:
        debug_print("\n\nScript interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        debug_print(f"\nFatal error in main execution: {str(e)}")
        debug_print("Please check your configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    debug_print("DEBUG: Script entry point reached")
    try:
        main()
    except KeyboardInterrupt:
        debug_print("\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        debug_print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
