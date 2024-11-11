import pandas as pd
from rapidfuzz import process as rapidfuzz_process
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
import re
import googlemaps
 
gmaps = googlemaps.Client(key='AIzaSyDnTe7Ig89V5Xj6awzqomTE_gkkc2KWU0U')
 
def searchResult(events_data, request):
    data = request.json
    user_query = data['user_query']
    user_lat = data.get("latitude")
    user_lon = data.get("longitude")
    default_user_location = {'lat': user_lat, 'lng': user_lon}
    radius_km = 10000000
    known_list = events_data['fullTitle'].unique().tolist()
    events_data['Combined'] = events_data['interest'].astype(str) + " : " + events_data['fullTitle'].astype(str) + " : " + events_data['about'].astype(str)+ " : "+events_data['keyFeatures'].astype(str)
 
    def calculate_distance(user_location, event_lat, event_lng):
        destination = {'lat': event_lat, 'lng': event_lng}
        try:
            result = gmaps.distance_matrix(origins=[user_location], destinations=[destination])
            distance = result['rows'][0]['elements'][0]['distance']['value']
            return distance
        except Exception as e:
            return None
 
    def normalize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        normalized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(normalized_words)
 
    def find_events_within_radius(user_location, radius_km):
        events_data['Distance'] = events_data.apply(lambda row: calculate_distance(user_location, row['latitude'], row['longitude']), axis=1)
        nearby_events = events_data[events_data['Distance'] <= radius_km]
        nearby_events = nearby_events.sort_values(by='Distance')
        return nearby_events
 
    def correct_query(user_query, known_list):
        spell = SpellChecker()
        common_words = set(stopwords.words('english'))
        tokens = re.findall(r"[\w'-]+|[.,!?;]", user_query)
        corrected_tokens = []
        lower_known_list = [str(item).lower() for item in known_list]
       
        for i, token in enumerate(tokens):
            # Skip common words
            if token.lower() in common_words:
                corrected_tokens.append(token)
                continue
           
            # Skip known list words
            if token.lower() in lower_known_list:
                corrected_tokens.append(token)
                continue
           
            # Check if the spelling is correct
            if spell[token.lower()] == token.lower():
                corrected_tokens.append(token)
                continue
           
            # Check if the word is an English word
            if token.lower() in spell:
                corrected_tokens.append(token)
                continue
           
            # Correct only if misspelled
            match_score = rapidfuzz_process.extractOne(token.lower(), lower_known_list, score_cutoff=80)
            if match_score:
                match, score, idx = match_score
                if i > 0 and corrected_tokens[-1].lower() == known_list[idx].lower():
                    continue
                corrected_tokens.append(known_list[idx])
            else:
                corrected_token = spell.correction(token) if 'kms' not in token and not gmaps.geocode(token) else token
                corrected_tokens.append(corrected_token)
       
        corrected_query = ''.join([' ' + i if not i.startswith("'") and i not in ".,!?;" else i for i in corrected_tokens]).strip()
        return corrected_query
 
 
 
    def get_date_range(corrected_query):
        today = datetime.today()
        if 'today' in corrected_query:
            start_date = end_date = today
        elif 'tomorrow' in corrected_query:
            start_date = end_date = today + timedelta(days=1)
        elif 'this weekend' in corrected_query:
            start_date = today + timedelta((5 - today.weekday()) % 7)
            end_date = start_date + timedelta(days=1)
        elif 'next weekend' in corrected_query:
            start_date = today + timedelta((12 - today.weekday()))
            end_date = start_date + timedelta(days=1)
        elif 'next month' in corrected_query:
            first_next_month = today.replace(day=1) + timedelta(days=31)
            start_date = first_next_month.replace(day=1)
            end_date = start_date.replace(month=start_date.month % 12 + 1, day=1) - timedelta(days=1)
        else:
            start_date, end_date = min(events_data['Start Date']), max(events_data['End Date'])
        return start_date, end_date
 
    corrected_query = correct_query(user_query, known_list)
    print(f"Corrected Query: {corrected_query}")
    # start_date, end_date = get_date_range(corrected_query)
    # filtered_df = events_data[(events_data['Start Date'] >= start_date.strftime('%Y-%m-%d')) & (events_data['End Date'] <= end_date.strftime('%Y-%m-%d'))]
 
    event_categories = ['Sports', 'Tech and Innovation', 'Music and Entertainment', 'Dance', 'Art and Culture', 'Business and Networking', 'Books and Education', 'Adventure', 'Fitness', 'Travel', 'Photography', 'Food and Drinks', 'Fashion', 'Other']
    # sub_categories = events_data['Sub Category'].dropna().unique().tolist()
    matched_categories = [category for category in event_categories if re.search(category, normalize_text(corrected_query), re.IGNORECASE)]
    print(f"matched_categories: {matched_categories}")
    # matched_sub_categories = [subcategory for subcategory in sub_categories if re.search(subcategory, normalize_text(corrected_query), re.IGNORECASE)]
 
    location_related = False
    user_location = default_user_location
    patterns = [
        r'within (\d+)\s*kms? (\w+)', r'within (\d+)\s*kms?', r'in (\d+)\s*kms?', r'near (\w+)',
        r'around (\w+)', r'in (\w+)', r'at (\w+)', r'near(\s+\w+)*', r'around(\s+\w+)*'
    ]
    for pattern in patterns:
        match = re.search(pattern, corrected_query, re.IGNORECASE)
 
        if match:
            location_related = True
            if 'near me' in corrected_query or 'around me' in corrected_query or 'nearby' in corrected_query:
                break
            else:
                if len(match.groups()) == 2:
                    radius_km = int(match.group(1))
                    user_location_name = match.group(2)
                    geocode_origin = gmaps.geocode(user_location_name, components={'locality': 'Pune'})
                    origin_coords = geocode_origin[0]['geometry']['location']
                    user_location = {'lat': origin_coords['lat'], 'lng': origin_coords['lng']}
                else:  # if len(match.groups()) == 1
                    if 'km' in corrected_query or 'kms' in corrected_query:
                        radius_km = int(match.group(1))
                        print("radius_km in user query: ", radius_km)
                    else:
                        user_location_name = match.group(1)
                        geocode_origin = gmaps.geocode(user_location_name, components={'locality': 'Pune'})
                        origin_coords = geocode_origin[0]['geometry']['location']
                        user_location = {'lat': origin_coords['lat'], 'lng': origin_coords['lng']}
            break
 
    if location_related:
        print(f"location_related: {location_related}")
        nearby_events = find_events_within_radius(user_location, radius_km)
 
        if matched_categories:
            nearby_events = nearby_events[(nearby_events['interest_list'].apply(lambda x: matched_categories in x))]
            other_events = events_data[(events_data['interest_list'].apply(lambda x: matched_categories in x))]
            other_events = other_events.sort_values(by='Distance')
            combined_df = pd.concat([nearby_events, other_events]).drop_duplicates(subset=['Event ID'])
        else:
            combined_df = nearby_events
            print(combined_df)
    else:
        if matched_categories:
            filtered_df = events_data[events_data['interest_list'].apply(lambda x: any(cat in x for cat in matched_categories))]
            combined_df = filtered_df
        else:
            combined_df = events_data
 
    vectorizer = TfidfVectorizer(stop_words=['events', 'event'])
    tfidf_matrix = vectorizer.fit_transform(combined_df['Combined'].tolist())
    query_vector = vectorizer.transform([corrected_query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    combined_df['Cosine Similarity'] = cosine_similarities
    combined_df = combined_df.sort_values(by=['Distance', 'Cosine Similarity'], ascending=[True, False]) if location_related else combined_df.sort_values(by='Cosine Similarity', ascending=False)
 
    combined_df = combined_df.to_json(orient='records')
    return combined_df