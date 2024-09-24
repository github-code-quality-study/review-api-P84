import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download and set up the sentiment analyzer
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Define valid locations
VALID_LOCATIONS = [
    'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California',
    'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California',
    'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California',
    'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California',
    'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah',
    'San Diego, California', 'Tucson, Arizona'
]

# Load reviews from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def analyze_sentiment(self, text):
        return sia.polarity_scores(text)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        method = environ["REQUEST_METHOD"]

        if method == "GET":
            return self.handle_get(environ, start_response)
        elif method == "POST":
            return self.handle_post(environ, start_response)
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"Method not allowed"]

    def handle_get(self, environ, start_response):
        params = parse_qs(environ.get('QUERY_STRING', ''))
        location = params.get('location', [None])[0]
        start_date = params.get('start_date', [None])[0]
        end_date = params.get('end_date', [None])[0]

        filtered_reviews = self.filter_reviews(location, start_date, end_date)
        self.add_sentiments(filtered_reviews)

        response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json")])
        return [response_body]

    def handle_post(self, environ, start_response):
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(content_length).decode('utf-8')
            review_data = dict(parse_qs(request_body))

            location = review_data.get('Location', [None])[0]
            review_body = review_data.get('ReviewBody', [None])[0]

            if not location or not review_body:
                raise ValueError("Missing 'Location' or 'ReviewBody'")

            if location not in VALID_LOCATIONS:
                raise ValueError("Invalid location")

            new_review = self.create_review(location, review_body)
            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [("Content-Type", "application/json")])
            return [response_body]

        except Exception as e:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [str(e).encode("utf-8")]

    def filter_reviews(self, location, start_date, end_date):
        filtered = reviews
        if location:
            if location in VALID_LOCATIONS:
                filtered = [r for r in filtered if r['Location'] == location]
            else:
                return []

        if start_date:
            filtered = [r for r in filtered if r['Timestamp'] >= start_date]

        if end_date:
            filtered = [r for r in filtered if r['Timestamp'] <= end_date]

        return filtered

    def add_sentiments(self, reviews):
        for review in reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

    def create_review(self, location, review_body):
        return {
            'ReviewId': str(uuid.uuid4()),
            'Location': location,
            'ReviewBody': review_body,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sentiment': self.analyze_sentiment(review_body)
        }

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
