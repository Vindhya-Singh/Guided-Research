import json
import re
from elasticsearch import Elasticsearch
from esengine import *
import os

from recipe import *
from constants import *

def get_filtered_ingredients(file_name):
    """Get the standard ingeredienst by reading the file obtained by querying
    DBpedia. The returned list is returned sorted accoring to the number of spaces
    in the text, which is useful when matching ingredients from the 1MRecipe
    corpus. This way multi-word ingredients are matched first.

    Parameters
    ----------
    file_name : string
        Path to the file containg the DBpedia ingredients.

    Returns
    -------
    list
        List of strings containing DBpedia ingredients.

    """
    with open(file_name) as f:
        db_ingredients = f.read().splitlines()
    db_ingredients = [x.lower().replace('"','') for x in db_ingredients]
    db_ingredients.sort(key=lambda x: x.count(' '), reverse=True)
    return db_ingredients

def load_data(file_name, ing_file_name, local=False, save=True, reduced=True):
    """Reads a local JSON file that is obtained from Recipe1M corpus.

    Parameters
    ----------
    file_name : string
        Path to the JSON file that contains the recipes.
    ing_file_name : string
        Path to the DBpedia file that contains the ingredients.
    local : boolean
        Decides whether to leave out some of the parameters required by ESengine
        while constructing an instance of the class. Set to True if data are being
        read from a local JSON file.
    save : boolean
        Decides whether to upload the read recipes to elasticsearch or not. Set to
        False if you want to use the data locally and not upload them to ES.
    reduced : boolean
        Decides which version of the recipe class to use: Recipe or ReducedRecipe.
        Refer to recipe.py to check the differences.

    Returns
    -------
    list
        List of objects of type Recipe or ReducedRecipe.

    """
    db_ingredients = get_filtered_ingredients(ing_file_name)
    with open(file_name, 'r', encoding='iso-8859-1') as data_file:
        try:
            json_data = data_file.read().strip()
        except Exception as e:
            print(e)
            print(f"Could not read file {filename}")
    if not json_data[0] == '[':
        json_data = '['+json_data
    if not json_data[-1] == ']':
        if json_data[-1] == ',':
            json_data = json_data[:-1]+']'
        else:
            json_data = json_data+']'
    all_data = json.loads(json_data)
    recipe_list = []
    for r in all_data:
        original_ing_list = [i["text"].lower() for i in r["ingredients"]]
        url = r["url"]
        id = r["title"]
        title = r["title"]
        instructions = [i["text"].lower() for i in r["instructions"]]
        if reduced:
            rec = ReducedRecipe(id=id, title=title, instruction_list=instructions, original_ing_list=original_ing_list, local=local)
            rec.filter_recipe(db_ingredients, original_ing_list)
        else:
            rec = Recipe(id=id, title=title, original_ing_list=original_ing_list,
                     instruction_list=instructions, url=url, local=local)
            rec.get_one_hot_encoding(db_ingredients)
            rec.filter_recipe(db_ingredients)

        recipe_list.append(rec)
        if save:
            rec.save()
    return recipe_list

def upload_all_to_es(dir, ing_file_name, reduced=True):
    """Uploads the recipes contained in the JSOn files inside dir to the ES server.
    Because working with a single JSON file was infeasable, spliiting the file
    was required. The different parts are in contained the dir directory as smaller
    JSON files. (Use the command "split" in the terminal to produce the segments of the
    JSON file)

    Parameters
    ----------
    dir : string
        Path to the directory containing the JSON segments of the Recipe1M corpus.
    ing_file_name : string
        Path to the DBpedia file that contains the ingredients.
    reduced : boolean
        Decides which version of the recipe class to use: Recipe or ReducedRecipe.
        Refer to recipe.py to check the differences.
    Returns
    -------
    None

    """
    if reduced:
        index = "reduced_all_recipes"
    else:
        index = "all_recipes"
    es = Elasticsearch(host=ES_SERVER, port="9200")
    try:
        es.indices.delete(index=index)
    except Exception as e:
        print(e)
    es.indices.create(index=index)
    files = os.listdir(dir)
    l = len(files)
    for i,f in enumerate(files):
        load_data(dir+"/"+f, ing_file_name, local=True, reduced=reduced)
        print(f"Uploaded {i}/{l} files")

def load_data_from_es(q, size=100, reduced=True):
    """Loads the data from Elasticsearch server.

    Parameters
    ----------
    q : strings
        Query to search for in ES server. Refer to constants.py to check the predefined
        queries used for the different food groups
    size : int
        Number of recipes to retrieve, ordered according to score given by Elasticsearch
        for that specific query. Maximum number is 10000.
    reduced : boolean
        Decides which version of the recipe class to use: Recipe or ReducedRecipe.
        Refer to recipe.py to check the differences. Here it decides which elasticsearch
        index to query.

    Returns
    -------
    list
        List of objects of type Recipe or ReducedRecipe.

    """
    print(f"Getting {size} documents for {q}")
    payload = Payload(query=q).size(size)
    if reduced:
        recipe_list = list(ReducedRecipe.search(payload))
    else:
        recipe_list = list(Recipe.search(payload))
    return recipe_list
