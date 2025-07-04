'''
This script fetches the bill text from Congress.gov using alternative methods.
'''
import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

# Working URLs for bill text
URLS = [
    "https://www.congress.gov/119/bills/hr1/BILLS-119hr1eh.htm",
    "https://www.congress.gov/119/bills/hr1/BILLS-119hr1ih.htm"
]

# Output directory and file path
OUTPUT_DIR = "books"
FILE_NAME = "one_big_beautiful_bill.txt"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILE_NAME)

def fetch_doc():
    '''Fetches the bill text from Congress.gov and saves it as plain text.'''
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")

        # Try each working URL
        for i, url in enumerate(URLS):
            print(f"Fetching from URL {i+1}: {url}...")
            
            try:
                # Create a session to maintain cookies and connection state
                session = requests.Session()
                
                # Add comprehensive headers to make the request appear more like a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0'
                }
                
                session.headers.update(headers)
                
                # Add a small delay to be respectful
                time.sleep(2)
                
                response = session.get(url, timeout=30)
                response.raise_for_status()  # Raise an exception for HTTP errors
                print(f"Content fetched successfully from {url}")
                
                # Parse the HTML using BeautifulSoup
                print("Parsing HTML content...")
                soup = BeautifulSoup(response.content, 'html.parser')
                print("HTML parsed successfully.")
                
                # Try to extract the bill text content
                # Look for the main content area that typically contains bill text
                bill_text = None
                
                # Try different selectors that might contain the bill text
                selectors = [
                    '.bill-text-container',
                    '.generated-html-container', 
                    '.main-content',
                    '#main-content',
                    '.bill-text',
                    'main'
                ]
                
                for selector in selectors:
                    content = soup.select_one(selector)
                    if content:
                        bill_text = content.get_text(separator='\n', strip=True)
                        print(f"Found content using selector: {selector}")
                        break
                
                # If no specific selector worked, try getting all text from body
                if not bill_text:
                    bill_text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)
                    print("Using general body text extraction")
                
                # Write the extracted text to the output file
                if bill_text and len(bill_text.strip()) > 100:  # Make sure we got substantial content
                    print(f"Saving text to {OUTPUT_PATH}...")
                    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                        f.write(f"Fetched from: {url}\n")
                        f.write("="*50 + "\n\n")
                        f.write(bill_text)
                    print(f"Successfully saved bill text to {OUTPUT_PATH}")
                    print(f"Content length: {len(bill_text)} characters")
                    return  # Success, exit the function
                else:
                    print(f"Content too short or empty from {url}, trying next URL...")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching from {url}: {e}")
                continue
                
        # If we get here, all working URLs failed
        print("All working URLs failed. Please check:")
        print("1. Your internet connection")
        print("2. If Congress.gov is accessible")
        print("3. Try running the script again later")

    except IOError as e:
        print(f"Error writing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_doc()
