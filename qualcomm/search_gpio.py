#!/usr/bin/env python3
"""
Search filesystem for GPIO-related files and directories
"""

import os
import subprocess

def search_filesystem():
    # Keywords to search for
    keywords = ['gpio', 'pin', 'qualcomm', 'soc']
    
    # Directories to skip (to avoid hanging or permission issues)
    skip_dirs = {'/proc', '/sys/kernel/debug', '/sys/kernel/tracing', '/run'}
    
    print("Searching filesystem for GPIO-related files...")
    
    try:
        # Use find command to search entire filesystem
        cmd = ['find', '/', '-type', 'f,d', '2>/dev/null']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Process each line
        for line in result.stdout.splitlines():
            # Skip directories we want to avoid
            if any(skip in line for skip in skip_dirs):
                continue
                
            # Check if line contains any of our keywords
            if any(keyword in line.lower() for keyword in keywords):
                try:
                    # Get file/directory info
                    stat = os.stat(line)
                    file_type = 'Directory' if os.path.isdir(line) else 'File'
                    print(f"{file_type}: {line}")
                    
                    # If it's a file and seems interesting, show first few lines
                    if os.path.isfile(line) and os.access(line, os.R_OK):
                        with open(line, 'r') as f:
                            try:
                                head = ' '.join(f.readline().strip() for _ in range(2))
                                if head:
                                    print(f"  Preview: {head[:100]}...")
                            except:
                                pass
                except:
                    continue
                
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    search_filesystem() 