#!/usr/bin/env python3
"""
Script to get versions of all packages used in the project
"""

import sys

# List of all packages identified from the project
packages = [
    'torch',
    'numpy',
    'matplotlib',
    'tqdm',
    'Pillow',  # PIL is imported from Pillow
]

def get_version(package_name):
    """Get the version of a package"""
    try:
        if package_name == 'Pillow':
            # Special case for PIL
            import PIL
            return PIL.__version__
        else:
            module = __import__(package_name)
            return getattr(module, '__version__', 'unknown')
    except ImportError:
        return 'not installed'
    except Exception as e:
        return f'error: {str(e)}'

def main():
    print("=" * 60)
    print("Package Versions for meta-learning-from-scratch")
    print("=" * 60)
    print()
    
    for package in packages:
        version = get_version(package)
        # Handle the Pillow/PIL case for display
        display_name = 'PIL (Pillow)' if package == 'Pillow' else package
        print(f"{display_name:20s} : {version}")
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    main()
