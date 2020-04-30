import time
# Elasticsearch server name
ES_SERVER = "social1.cm.in.tum.de"
# Use local if using local elasticsearch server
# ES_SERVER = "127.0.0.1"

# Queries to look for the respective word in the ingredients list
PASTA = {"bool":{"should":[{"term":{"proccessed_ing_list.keyword":"pasta"}},{"term":{"proccessed_ing_list.keyword":"macaroni"}}]}}
FISH = {"bool":{"should":[{"term":{"proccessed_ing_list.keyword":"fish"}}]}}

# Queries to look for the respective word in the title
PIZZA = {"bool":{"should":[{"term":{"title":"pizza"}}]}}
BURGER = {"bool":{"should":[{"term":{"title":"burger"}}]}}
CAKE = {"bool":{"should":[{"term":{"title":"cake"}}]}}
SALAD =  {"bool":{"should":[{"term":{"title":"salad"}}]}}

# Generates a query to get a random sample using the current UNIX time as seed
def get_random_query(s):
    RANDOM = {"function_score": {"functions": [{"random_score": {"seed": f"{s}"}}]}}
    return RANDOM
