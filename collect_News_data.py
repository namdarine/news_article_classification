import os
import requests
import csv

existing_csv_path = '/Users/namgyulee/Personal_Project/news_data.csv'


new_data_url = (
    'https://newsapi.org/v2/everything?'
    'domains=reuters.com,chicagotribune.com,wsj.com&'
    'from=2023-12-07&to=2023-12-08&'
    'language=en&'
    'pagesize=16&'
    'apiKey=3c8c00c811e74114bf0774a1c6e34e41'
)

response = requests.get(new_data_url)

# Add New data to exist file
if response.status_code == 200:
    new_data = response.json().get('articles', [])

    with open(existing_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row_number, article in enumerate(new_data, start=985):
            title = article.get('title', '')
            publisher = article.get('source', {}).get('name', 'N/A')
            author = article.get('author', '')
            description = article.get('description', '')
            url = article.get('url', '')
            published_at = article.get('publishedAt', '')

            new_row_data = [row_number, publisher, title, author, description, url, published_at]
            csv_writer.writerow(new_row_data)

    print(f'Data has been successfully written to {existing_csv_path}')
else:
    print(f'Error: {response.status_code}')

"""
url = ('https://newsapi.org/v2/everything?'
       'domains=reuters.com,chicagotribune.com,wsj.com&'
       'language=en&'
       'from=2023-11-17&to=2023-11-18&'
       'apiKey=3c8c00c811e74114bf0774a1c6e34e41')

response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract relevant information from the response
    articles = data.get('articles', [])

    # Specify the full path and file name for the CSV file
    csv_path = '/Users/namgyulee/Personal_Project/news_data.csv'

    # Open the CSV file in write mode
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['Row Number', 'Publisher', 'Title', 'Author', 'Description', 'URL', 'Published At'])

        # Write each article to the CSV file with a row number
        for row_number, article in enumerate(articles, start=1):
            publisher = article.get('source', {}).get('name', 'N/A')
            title = article.get('title', '')
            author = article.get('author', '')
            description = article.get('description', '')
            url = article.get('url', '')
            published_at = article.get('publishedAt', '')

            # Write a row for each article
            csv_writer.writerow([row_number, publisher, title, author, description, url, published_at])

    print(f'Data has been successfully written to {csv_path}')
else:
    print(f'Error: {response.status_code}')
    
"""