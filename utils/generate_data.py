import requests
import json
import os
import re
import time

# --- Configuration ---
API_BASE_URL = "http://localhost:8000/ask"
GENERATION_MODE = "dependencies"
OUTPUT_DIR = "generated_data"
REQUEST_TIMEOUT_SECONDS = 600  # Timeout for each API request
RETRY_DELAY_SECONDS = 10  # Delay between retries
MAX_RETRIES = 3  # Maximum number of retries for a failed request

PROMPTS = [
    "Write a Python FastAPI backend application that provides a RESTful API for managing a 'Products' inventory. Operations should include CRUD for products (name, description, price, quantity). Use SQLAlchemy ORM to interact with a PostgreSQL database and Pydantic for data validation. Implement JWT-based authentication for write operations.",
    "Develop a Node.js Express application that serves as a blogging platform API. It should support creating, reading, updating, and deleting blog posts and comments. Use Mongoose to connect to a MongoDB database. Implement a GraphQL endpoint for fetching all posts by a specific author.",
    "Create a React single-page application that fetches and displays user data from the public GitHub API (`https://api.github.com/users/{username}`). The application should allow searching for a user and show their profile information and a list of their public repositories. Use Axios for HTTP requests and a state management library like Redux or Zustand.",
    "Write a Python script that scrapes today's top 5 articles from a news website (e.g., BBC News front page) using `requests` and `BeautifulSoup4`. Store the article titles, links, and a short summary into an SQLite database.",
    "Develop a command-line interface (CLI) tool in Go that interacts with a weather API (e.g., OpenWeatherMap API). The CLI should accept a city name and return the current temperature, humidity, and weather description. Handle API errors gracefully.",
    "Implement a Java Spring Boot application that exposes an API for a simple URL shortener. It should take a long URL, generate a short code, store the mapping in a H2 in-memory database (or PostgreSQL if preferred), and redirect to the long URL when the short code is accessed.",
    "Create a Python Flask application that allows users to upload images. The application should resize uploaded images to a thumbnail (e.g., 100x100 pixels) using the Pillow library and store both original and thumbnail paths in a MySQL database.",
    "Write a Node.js application that acts as a WebSocket server for a real-time chat. It should broadcast messages from one client to all other connected clients. Use the `ws` library or `Socket.IO`. Store chat history in a Redis cache for the last 100 messages.",
    "Develop a Python Celery worker that processes background tasks. The task should take a URL, fetch its content using `requests`, count the word frequencies, and store the top 10 words and their counts in a PostgreSQL database. Use RabbitMQ or Redis as the message broker.",
    "Create a Vue.js frontend application that consumes a public GraphQL API (e.g., `https://rickandmortyapi.com/graphql`). The application should display a list of characters with pagination and allow filtering by character status (Alive, Dead, Unknown). Use Apollo Client for Vue.",
    "Write a Python script using `pandas` and `matplotlib` (or `seaborn`) to analyze a CSV file containing sales data (columns: `date`, `product_id`, `quantity`, `price`). The script should calculate total revenue per month and generate a bar chart visualizing this. Save the chart as a PNG.",
    "Develop a Discord bot using Python with the `discord.py` library. The bot should respond to a `!stock {ticker}` command by fetching the current stock price for the given ticker symbol using an API like Alpha Vantage or Yahoo Finance.",
    "Implement a simple microservice in Go that exposes a gRPC endpoint. The service should accept a string and return its SHA256 hash. Include a client in Go that calls this gRPC service.",
    "Write a Python (or Node.js) AWS Lambda function that is triggered by an S3 PUT event. When a new JSON file is uploaded to a specific S3 bucket, the Lambda function should read the JSON, transform its structure (e.g., rename keys, filter data), and save the transformed JSON to another S3 bucket. Use the AWS SDK.",
    "Create a Next.js application with server-side rendering (SSR) that fetches data from a headless CMS (e.g., Strapi, Contentful - assume a simple API endpoint exists) for blog posts. Display a list of posts on the homepage and individual post pages.",
    "Develop a Python script that uses the `tweepy` library (or another Twitter API library) to fetch the latest 10 tweets from a specific public Twitter account and prints their text and creation date to the console. Handle API rate limits.",
    "Implement a Java application using Apache Kafka. Create a producer that sends messages containing sensor data (timestamp, sensor_id, value) to a Kafka topic. Create a consumer that reads from this topic and logs the data to the console or a file.",
    "Write a Ruby on Rails application providing an API for a to-do list. Users should be able to create, read, update (mark as complete), and delete tasks. Use Active Record with a PostgreSQL database. Implement token-based authentication for API access.",
    "Create a Python FastAPI application that serves a pre-trained machine learning model (e.g., a simple scikit-learn classifier saved with `joblib` or `pickle`). The API should accept input features as JSON and return the model's prediction.",
    "Develop a Node.js script that uses a GraphQL client library (like `graphql-request` or Apollo Client for Node.js) to perform a mutation on a public GraphQL API that supports mutations (e.g., a test instance you set up, or a public one like `https://graphqlzero.almansi.me/api`). For example, create a new \"Post\" if the API supports it. Log the response."
]

# --- Helper Function to Sanitize Filenames ---


def sanitize_filename(text, max_length=50):
    """Sanitizes a string to be used as a filename."""
    # Remove non-alphanumeric characters (except spaces and hyphens)
    text = re.sub(r"[^\w\s-]", "", text).strip()
    # Replace spaces and multiple hyphens with a single underscore
    text = re.sub(r"[-\s]+", "_", text)
    # Truncate if too long
    return text[:max_length]


# --- Main Script Logic ---


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    for i, prompt_text in enumerate(PROMPTS):
        print(f"\nProcessing prompt {i+1}/{len(PROMPTS)}...")
        print(f"Prompt: {prompt_text[:100]}...")  # Print a snippet

        params = {"task_description": prompt_text, "generation_mode": GENERATION_MODE}

        retries = 0
        success = False
        while retries <= MAX_RETRIES and not success:
            try:
                response = requests.post(
                    API_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

                try:
                    data = response.json()
                    success = True  # Mark as successful if JSON parsing works
                except json.JSONDecodeError:
                    print(f"Error: API response for prompt {i+1} is not valid JSON.")
                    # Print beginning of non-JSON response
                    print(f"Response text: {response.text[:200]}...")
                    # Save the raw response for debugging
                    filename_prefix = sanitize_filename(prompt_text.split(".")[0])
                    error_filename = os.path.join(
                        OUTPUT_DIR, f"error_prompt_{i+1:02d}_{filename_prefix}.txt"
                    )
                    with open(error_filename, "w", encoding="utf-8") as f_err:
                        f_err.write(
                            f"Prompt:\n{prompt_text}\n\nResponse Status: {response.status_code}\nResponse Text:\n{response.text}"
                        )
                    print(f"Raw error response saved to {error_filename}")
                    break  # Don't retry JSON decode errors, move to next prompt

                if success:
                    # Generate a somewhat descriptive filename
                    filename_prefix = sanitize_filename(
                        prompt_text.split(".")[0]
                    )  # Use first part of prompt
                    output_filename = os.path.join(
                        OUTPUT_DIR, f"result_prompt_{i+1:02d}_{filename_prefix}.json"
                    )

                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    print(
                        f"Successfully fetched and saved analysis to: {output_filename}"
                    )

            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error for prompt {i+1}: {e}")
                if response is not None:
                    print(f"Response content: {response.text[:200]}...")
                retries += 1
                if retries <= MAX_RETRIES:
                    print(
                        f"Retrying in {RETRY_DELAY_SECONDS} seconds... (Attempt {retries}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Max retries reached for prompt {i+1}. Skipping.")
            except requests.exceptions.ConnectionError as e:
                print(
                    f"Connection Error for prompt {i+1}: {e}. Is the API server running at {API_BASE_URL}?"
                )
                retries += 1
                if retries <= MAX_RETRIES:
                    print(
                        f"Retrying in {RETRY_DELAY_SECONDS} seconds... (Attempt {retries}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Max retries reached for prompt {i+1}. Skipping.")
            except requests.exceptions.Timeout as e:
                print(f"Timeout Error for prompt {i+1}: {e}")
                retries += 1
                if retries <= MAX_RETRIES:
                    print(
                        f"Retrying in {RETRY_DELAY_SECONDS} seconds... (Attempt {retries}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Max retries reached for prompt {i+1}. Skipping.")
            except requests.exceptions.RequestException as e:
                print(f"An unexpected error occurred for prompt {i+1}: {e}")
                retries += 1  # Generic retry for other request exceptions
                if retries <= MAX_RETRIES:
                    print(
                        f"Retrying in {RETRY_DELAY_SECONDS} seconds... (Attempt {retries}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"Max retries reached for prompt {i+1}. Skipping.")

            # Small delay between successful requests to be polite to the server, even if local
            if success:
                time.sleep(0.5)


if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")
