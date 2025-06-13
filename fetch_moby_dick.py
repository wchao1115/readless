'''
This script fetches the HTML content of Moby Dick from Project Gutenberg,
parses it using BeautifulSoup, and saves the plain text to a file.
'''
import requests
from bs4 import BeautifulSoup
import os

# URL for the HTML version of Moby Dick on Project Gutenberg
URL = "https://www.gutenberg.org/files/2701/2701-h/2701-h.htm"

# Output directory and file path
OUTPUT_DIR = "books"
FILE_NAME = "moby_dick.txt"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILE_NAME)

def fetch_and_save_moby_dick():
    '''Fetches Moby Dick from Project Gutenberg and saves it as plain text.'''
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")

        # Fetch the HTML content
        print(f"Fetching content from {URL}...")
        response = requests.get(URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Content fetched successfully.")

        # Parse the HTML using BeautifulSoup
        print("Parsing HTML content...")
        soup = BeautifulSoup(response.content, 'html.parser')
        print("HTML parsed successfully.")

        # Extract text content
        # This might need adjustment based on the specific HTML structure of the page.
        # For Gutenberg pages, often the main content is within the <body> tag.
        # We can try to get all text and then clean it up if necessary.
        # A more robust way might be to identify specific tags holding the main story.
        # For now, let's get all text from the body.
        story_text = soup.body.get_text(separator='\n', strip=True)

        # Write the extracted text to the output file
        print(f"Saving text to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(story_text)
        print(f"Successfully saved Moby Dick to {OUTPUT_PATH}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except IOError as e:
        print(f"Error writing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_and_save_moby_dick()
