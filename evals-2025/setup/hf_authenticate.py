#!/usr/bin/env python3
"""
Handle HuggingFace authentication for accessing restricted datasets
"""
import os
from huggingface_hub import login
import getpass

def setup_hf_authentication():
    """Setup HuggingFace authentication"""
    print("Setting up HuggingFace authentication...")
    
    # Check if already logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Already logged in as: {user_info['name']}")
        return True
    except Exception:
        pass
    
    # Get token from user
    token = getpass.getpass("Enter your HuggingFace token (from https://huggingface.co/settings/tokens): ")
    
    try:
        login(token=token)
        print("Successfully authenticated with HuggingFace!")
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

if __name__ == "__main__":
    setup_hf_authentication()