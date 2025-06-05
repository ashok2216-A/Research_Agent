import requests
from youtubesearchpython import VideosSearch
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
# from scholarly import scholarly, ProxyGenerator

def search_datasets(query: str) -> str:
    try:
        response = requests.get(
            "https://zenodo.org/api/records",
            params={"q": query,"type": "dataset","size": 3,"sort": "mostrecent"})

        response.raise_for_status()
        data = response.json()
        if not data.get('hits', {}).get('hits', []):
            return f"No datasets found for query: {query}"
            
        results = []
        for hit in data['hits']['hits']:
            metadata = hit['metadata']
            results.append(
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Authors: {', '.join(author.get('name', '') for author in metadata.get('creators', []))}\n"
                f"Publication Date: {metadata.get('publication_date', 'N/A')}\n"
                f"Description: {metadata.get('description', 'N/A')[:200]}...\n"
                f"DOI: {metadata.get('doi', 'N/A')}\n"
                f"URL: https://zenodo.org/record/{hit['id']}"
            )
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching datasets: {str(e)}"


def academic_search(query: str) -> str:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 3, "fields": "title,authors,year,venue,abstract,url"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses

        papers = response.json().get("data", [])
        if not papers:
            return "No academic papers found."

        results = []
        for paper in papers:
            title = paper.get("title", "N/A")
            authors = ', '.join([a.get("name", "") for a in paper.get("authors", [])])
            year = paper.get("year", "N/A")
            venue = paper.get("venue", "N/A")
            abstract = paper.get("abstract", "No abstract available")
            url = paper.get("url", "No URL available")

            results.append(
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Year: {year}\n"
                f"Venue: {venue}\n"
                f"Abstract: {abstract[:200]}...\n"
                f"URL: {url}"
            )
        return "\n\n".join(results)
    except requests.exceptions.RequestException as e:
        return f"Error fetching papers: {str(e)}"


def youtube_search(query: str, max_results: int = 3) -> str:
    """
    Search YouTube videos using YouTube Data API v3.
    Note: Requires YOUTUBE_API_KEY in environment variables.
    """
    try:
        # Get API key from environment variable
        api_key = "AIzaSyAO32s3nsYmbvmACUKSY6SDaJ-tNzMif3Q"
        if not api_key:
            return "Error: YouTube API key not found in environment variables. Please set YOUTUBE_API_KEY."

        # Create YouTube API client
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Call the search.list method to retrieve results
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=max_results,
            type='video',
            order='relevance'
        ).execute()

        if not search_response.get('items'):
            return "No videos found for your query."

        # Format results
        output = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            snippet = item['snippet']

            # Get video statistics
            video_response = youtube.videos().list(
                part='statistics,contentDetails',
                id=video_id
            ).execute()

            if video_response['items']:
                stats = video_response['items'][0]['statistics']
                duration = video_response['items'][0]['contentDetails']['duration']
                
                # Format video information
                output.append(
                    f"Title: {snippet['title']}\n"
                    f"Channel: {snippet['channelTitle']}\n"
                    f"Duration: {duration}\n"
                    f"URL: https://www.youtube.com/watch?v={video_id}\n"
                    f"Views: {stats.get('viewCount', 'N/A')}\n"
                    f"Description: {snippet['description'][:200]}..."
                )

        return "\n\n".join(output)

    except HttpError as e:
        return f"An HTTP error occurred: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    


    # AIzaSyAO32s3nsYmbvmACUKSY6SDaJ-tNzMif3Q