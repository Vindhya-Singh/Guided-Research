#!pip3 install simhash
import numpy as np
import ipdb
from elasticsearch import Elasticsearch
from esengine import *
#from simhash import simhash

from constants import ES_SERVER

# Accumlator for the missing ingredients
missing_ingredients = {}
class Recipe(Document):
    """Class for easy representation of recipes. The class inherits from class
    Document from the esengine package for easy interfacing with elasticsearch. """
    # index to store the recipes in
    _index = 'all_recipes'
    # default document type
    _doctype = 'recipe'
    # default client instance
    _es = Elasticsearch(host=ES_SERVER, port="9200", timeout=100000)

    # Main attributes of a recipe
    # Attributes directly read out of the JSON file
    id = StringField()
    title = StringField()
    original_ing_list = ArrayField(StringField())
    instruction_list = ArrayField(StringField())
    url = StringField()
    # Atributes processed before stroing in Elasticsearch
    instructions_text = StringField()
    ingredients_text = StringField()
    proccessed_ing_list =  ArrayField(StringField())
    one_hot =  ArrayField(StringField())

    # Constructor
    def __init__(self, id, title, original_ing_list,
                instruction_list, url, instructions_text=None, ingredients_text=None,
                proccessed_ing_list=None, one_hot=None, local=False):
        """Constructor for the Recipe class. Some of the attributes are None by
        default, so that the same constructor could be used when reading data
        from Elasticsearch as well as from a local file.

        Parameters
        ----------
        id : string
            ID of recipe as observed in the JSON file.
        title : string
            Title of recipe as observed in the JSON file.
        original_ing_list : list
            List of ingredients as observed in the JSON file.
        instruction_list : type
            List of cooking instructions as observed in the JSON file.
        url : string
            URL of recipe source as observed in the JSON file.
        instructions_text : string
            Contains the concatenation of the elements in instructions list
            separated by a " ".
        ingredients_text : type
            Contains the concatenation of the elements in ingredients list
            separated by a " ".
        proccessed_ing_list : list
            List of "clean" ingredients after filtering agianst the ones extracted
            from DBpedia.
        one_hot : list
            List of boolean of the same length as the number of elements extracted
            from DBpedia. A True value indicates the presence of the ingredient at
            that position in the recipe.
        local : boolean
            Decides whether to expect the attributes that need processing to be passed
            to the constructer or to construct them while reading the other values.
            The values are passed by default when the recipe is constructed by
            reading it from Elasticsearch.

        Returns
        -------
        Recipe
            Instance of the Recipe class.

        """

        self.id = id
        self.title = title
        self.original_ing_list = original_ing_list
        self.instruction_list = instruction_list
        self.url = url
        if local:
            # If reading from a file
            self.instructions_text = " ".join(self.instruction_list)
            self.ingredients_text = " ".join(self.original_ing_list)
            self.proccessed_ing_list = []
            self.one_hot = []
        else:
            # If reading from Elasticsearch
            self.instructions_text = instructions_text
            self.ingredients_text = ingredients_text
            self.proccessed_ing_list = proccessed_ing_list
            self.one_hot = one_hot

    def __repr__(self):
        """Used for proper printing of the class instances.

        Returns
        -------
        string
            When using the function print on a class instance, the title and the
            clean ingredients list will be printed for proper representation
            of the recipe.

        """
        return self.title+"\n"+str(self.proccessed_ing_list)

    def __str__(self):
        """Used for proper printing of the class instances.

        Returns
        -------
        string
            When using the function print on a class instance, the title and the
            clean ingredients list will be printed for proper representation
            of the recipe.

        """
        return self.title+"\n"+str(self.proccessed_ing_list)

    def filter_recipe(self, db_ingredients):
        """Normalizes the ingredients list by matching each ingredient to the equivalent
        match from the DBpedia ingredients list.

        Parameters
        ----------
        db_ingredients : list
            List of strings of the ingredients extracted from DBpedia.

        Returns
        -------
        None
            Populates directly the proccessed_ing_list attribute of the recipe.

        """
        for i in self.original_ing_list:
            match_found = False
            for db_ing in db_ingredients:
                if db_ing in i:
                    if db_ing == 'tea' and 'teaspoon' in i:
                        break
                    match_found = True
                    if db_ing not in self.proccessed_ing_list:
                        self.proccessed_ing_list.append(db_ing)
                    break
        # Uncomment the next segment to investigate the ingredients that were
        # not matched against anything in the DBpedia list

        #     if not match_found:
        #         missing_ing = input(f"Could not find match for {i}:")
        #         if missing_ing in missing_ingredients:
        #             missing_ingredients[missing_ing] += 1
        #         else:
        #             missing_ingredients[missing_ing] = 1
        # print(missing_ingredients)


    def print_processed_ingredients(self):
        """Prints the clean list of ingredients. Was uesed for easy debugging."""
        print(self.proccessed_ing_list)

    def print_original_ingredients(self):
        """Prints the original list of ingredients. Was uesed for easy debugging."""
        print(self.original_ing_list)

    def get_one_hot_encoding(self, all_ings):
        """Creates the 1-hot-vector of the ingredients list.

        Parameters
        ----------
        all_ings : list
            List of strings of the ingredients extracted from DBpedia.

        Returns
        -------
        None
            Populates directly the one_hot attribute of the recipe.

        """
        self.one_hot = np.isin(all_ings, self.proccessed_ing_list)

class ReducedRecipe(Document):
    """Class for easy representation of recipes. The class inherits from class
    Document from the esengine package for easy interfacing with elasticsearch.
    This class differs from the previous one in that it is optimized in the stored
    attributes in order to reduce the required memory. It leaves out the 1-hot-vector,
    original_ing_list, instruction_list as well as the URL attributes"""
    # index to store the recipes in
    _index = 'reduced_all_recipes'
    # _index = 'all_recipes'
    # default document type
    _doctype = 'recipe'
    # default client instance
    _es = Elasticsearch(host=ES_SERVER, port="9200", timeout=100000)

    # Main attributes of a recipe
    # Attributes directly read out of the JSON file
    id = StringField()
    title = StringField()
    # Atributes processed before stroing in Elasticsearch
    instructions_text = StringField()
    ingredients_text = StringField()
    proccessed_ing_list =  ArrayField(StringField())

    # Constructor
    def __init__(self, id, title, instruction_list=None, original_ing_list=None, instructions_text=None, ingredients_text=None,
                proccessed_ing_list=None, local=False):

        """Constructor for the Recipe class. Some of the attributes are None by
        default, so that the same constructor could be used when reading data
        from Elasticsearch as well as from a local file.

        Parameters
        ----------
        id : string
            ID of recipe as observed in the JSON file.
        title : string
            Title of recipe as observed in the JSON file.
        instruction_list : type
            List of cooking instructions as observed in the JSON file.
        original_ing_list : list
            List of ingredients as observed in the JSON file.
        instructions_text : string
            Contains the concatenation of the elements in instructions list
            separated by a " ".
        ingredients_text : type
            Contains the concatenation of the elements in ingredients list
            separated by a " ".
        proccessed_ing_list : list
            List of "clean" ingredients after filtering agianst the ones extracted
            from DBpedia.
        local : boolean
            Decides whether to expect the attributes that need processing to be passed
            to the constructer or to construct them while reading the other values.
            The values are passed by default when the recipe is constructed by
            reading it from Elasticsearch.

        Returns
        -------
        Recipe
            Instance of the Recipe class.

        """

        self.id = id
        self.title = title

        if local:
            # If reading from a file
            self.instructions_text = " ".join(instruction_list)
            self.ingredients_text = " ".join(original_ing_list)
            self.proccessed_ing_list = []
        else:
            # If reading from Elasticsearch
            self.instructions_text = instructions_text
            self.ingredients_text = ingredients_text
            self.proccessed_ing_list = proccessed_ing_list

    def __repr__(self):
        """Used for proper printing of the class instances.

        Returns
        -------
        string
            When using the function print on a class instance, the title and the
            clean ingredients list will be printed for proper representation
            of the recipe.

        """
        return self.title+"\n"+str(self.proccessed_ing_list)
      
    def __str__(self):
        """Used for proper printing of the class instances.

        Returns
        -------
        string
            When using the function print on a class instance, the title and the
            clean ingredients list will be printed for proper representation
            of the recipe.

        """
        return self.title+"\n"+str(self.proccessed_ing_list)
        
    def filter_recipe(self, db_ingredients, original_ing_list):
        """Normalizes the ingredients list by matching each ingredient to the equivalent
        match from the DBpedia ingredients list.

        Parameters
        ----------
        db_ingredients : list
            List of strings of the ingredients extracted from DBpedia.

        Returns
        -------
        None
            Populates directly the proccessed_ing_list attribute of the recipe.

        """
        for i in original_ing_list:
            match_found = False
            for db_ing in db_ingredients:
                if db_ing in i:
                    if db_ing == 'tea' and 'teaspoon' in i:
                        break
                    match_found = True
                    if db_ing not in self.proccessed_ing_list:
                        self.proccessed_ing_list.append(db_ing)
                    break
        # Uncomment the next segment to investigate the ingredients that were
        # not matched against anything in the DBpedia list

        #     if not match_found:
        #         missing_ing = input(f"Could not find match for {i}:")
        #         if missing_ing in missing_ingredients:
        #             missing_ingredients[missing_ing] += 1
        #         else:
        #             missing_ingredients[missing_ing] = 1
        # print(missing_ingredients)


    def print_processed_ingredients(self):
        """Prints the original list of ingredients. Was uesed for easy debugging."""
        print(self.proccessed_ing_list)
