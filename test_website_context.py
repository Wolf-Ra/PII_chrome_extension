#!/usr/bin/env python3
"""
Test script for website context analysis and PII redaction.
This script tests the enhanced LLM-based domain analysis.
"""

import os
import json
from server import _get_website_context
from pipeline import SanitizationPipeline, Config

def test_domain_analysis():
    """Test the website context analysis with different domains."""
    
    # Test domains from different industries
    test_domains = [
        "hdfcbank.com",
        "mayoclinic.com", 
        "github.com",
        "harvard.edu",
        "amazon.com",
        "linkedin.com"
    ]
    
    print("=== Testing Website Context Analysis ===\n")
    
    for domain in test_domains:
        print(f"Testing domain: {domain}")
        context = _get_website_context(domain)
        
        if context:
            print(f"  Website Type: {context.get('website_type', 'N/A')}")
            print(f"  Industry: {context.get('industry', 'N/A')}")
            print(f"  Sensitivity Level: {context.get('sensitivity_level', 'N/A')}")
            print(f"  Primary PII Types: {context.get('primary_pii_types', [])}")
            print(f"  Description: {context.get('description', 'N/A')}")
        else:
            print("  No context returned (API key missing or error)")
        
        print("-" * 50)

def test_pipeline_integration():
    """Test the pipeline integration with website context."""
    
    print("\n=== Testing Pipeline Integration ===\n")
    
    # Create a sample website context
    sample_context = {
        "website_type": "healthcare",
        "industry": "hospital",
        "primary_pii_types": ["medical_records", "patient_id", "insurance_info", "ssn"],
        "sensitivity_level": "critical",
        "description": "Healthcare provider patient portal"
    }
    
    # Initialize pipeline
    pipeline = SanitizationPipeline()
    
    # Set website context
    pipeline.set_website_context(sample_context)
    
    print(f"Pipeline purpose: {pipeline.purpose}")
    print(f"Pipeline website context: {json.dumps(pipeline.website_context, indent=2)}")
    
    print("\nPipeline is ready for context-aware PII redaction!")

if __name__ == "__main__":
    # Check if API key is available
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable not set.")
        print("Set it to test the LLM-based domain analysis.")
        print("Example: export GROQ_API_KEY='your-api-key-here'")
        print()
    
    test_domain_analysis()
    test_pipeline_integration()
    
    print("\n=== Test Complete ===")
    print("To test with actual PDF processing:")
    print("1. Set GROQ_API_KEY environment variable")
    print("2. Run the server: python server.py")
    print("3. Upload a PDF through the Chrome extension")
