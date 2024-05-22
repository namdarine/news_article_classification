import requests
import csv
from function import get_API_key

key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 2)
api_key = api_key_instance.get_api_key(2)


# Collecting new Data
existing_csv_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/Data/new_data.csv'


new_data_url = (
    'https://newsapi.org/v2/everything?'
    'domains=reuters.com,chicagotribune.com,wsj.com&'
    'from=2024-01-04&to=2024-01-05&'
    'language=en&'
    'pagesize=15&'
    f'apiKey={api_key}'
)

response = requests.get(new_data_url)

# Add New data to exist file
if response.status_code == 200:
    new_data = response.json().get('articles', [])

    with open(existing_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row_number, article in enumerate(new_data, start=986):
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
       'from=2023-12-13&to=2023-12-14&'
       f'apiKey={api_key}')


response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract relevant information from the response
    articles = data.get('articles', [])

    # Specify the full path and file name for the CSV file
    csv_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/new_data.csv'

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

"""
existing_csv_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/news_data.csv'


new_data_url = (
    'https://newsapi.org/v2/everything?'
    'domains=reuters.com,chicagotribune.com,wsj.com&'
    'from=2023-12-11&to=2023-12-12&'
    'language=en&'
    'pagesize=7&'
    f'apiKey={api_key}'
)

response = requests.get(new_data_url)

# Add New data to exist file
if response.status_code == 200:
    new_data = response.json().get('articles', [])

    with open(existing_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row_number, article in enumerate(new_data, start=994):
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

url = ('https://newsapi.org/v2/everything?'
       'domains=reuters.com,chicagotribune.com,wsj.com&'
       'language=en&'
       'from=2023-11-22&to=2023-11-23&'
       f'apiKey={api_key}')


response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract relevant information from the response
    articles = data.get('articles', [])

    # Specify the full path and file name for the CSV file
    csv_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/news_data.csv'

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
