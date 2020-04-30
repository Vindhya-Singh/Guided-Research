import numpy as np
from preprocess import get_filtered_ingredients
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ipdb
import multiprocessing as mp
import copy
import time

def get_top_n_from_mat(matrix, n):
    """Given any matrix, returns the indices of the top n entries in the matrix.

    Parameters
    ----------
    matrix : matirx
        An arbitrarty 2D matrix
    n : interger
        Number of top entries to return.

    Returns
    -------
    list
        List of tuples that contains the indices of the top elements.

    """
    idx = np.unravel_index(np.argsort(matrix.ravel())[-n:], matrix.shape)
    return idx

def get_top_n_with_lookup(matrix, n, lookup, print_res=False):
    """Given any matrix, returns the indices of the top n entries in the matrix and
    prints the string representation of the indices of the top elements.

    Parameters
    ----------
    matrix : matirx
        An arbitrarty 2D matrix
    n : interger
        Number of top entries to return.
    lookup : dict
        Maps the index to the string representation.
    print_res : boolean
        Decides whether to print the lookup value of the top elements or not.

    Returns
    -------
    list
        List of tuples that contains the indices of the top elements.

    """
    idx = np.unravel_index(np.argsort(matrix.ravel())[-n:], matrix.shape)
    res = []
    if print_res:
        print("Most similar Pairs")
    for i in range(len(idx[0])):
        if (idx[1][i],idx[0][i]) not in res:
            res.append((idx[0][i],idx[1][i]))
            if print_res:
                print("%s  <--->  %s"%(lookup[idx[0][i]],lookup[idx[1][i]]))
    return res



def get_logical_and_distance(recipe_list, print_top_sim=True):
    """Returns the matrix that contains the logical AND similarities between the
    1-hot-vectors of the recipes in recipe_list

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe, that have a "one_hot" attribute.
    print_top_sim : boolean
        Decides whether to print top similar recipe pairs or not.

    Returns
    -------
    matrix
        If the length of recipe_list is n, then the returned matrix is nxn that
        contains the logical AND similarity between every pair of recipes.

    """
    one_hot_list = [r.one_hot for r in recipe_list]
    dim = len(one_hot_list)
    distances = np.zeros((dim,dim))
    for i in range(0, len(one_hot_list)):
        for j in range(i+1, len(one_hot_list)):
            current_distance = np.sum(np.logical_and(one_hot_list[i].reshape(-1),one_hot_list[j].reshape(-1)))
            distances[i][j] = current_distance
            distances[j][i] = current_distance

    top_sim = np.unravel_index(distances.argmax(), distances.shape)
    #distances = np.reciprocal(distances)
    print("Most similar recipes: ")
    print(top_sim)
    print("Similarity value: %f" % distances[top_sim[0]][top_sim[1]])
    if(print_top_sim):
        print("---- Recipe %i ----"%top_sim[0])
        print(recipe_list[top_sim[0]])
        print("---- Recipe %i: ----"%top_sim[1])
        print(recipe_list[top_sim[1]])
    return distances


def get_logical_and_distance_with_percentage(recipe_list, print_top_sim=True):
    """Returns the matrix that contains the logical AND similarities between the
    1-hot-vectors of the recipes in recipe_list, divided by the length of the maximum
    ingredients list in each pair in order to account for cases where there is a
    long overlap that does not really indicate similarity, e.g. seasonings.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe, that have a "one_hot" attribute.
    print_top_sim : boolean
        Decides whether to print top similar recipe pairs or not.

    Returns
    -------
    matrix
        If the length of recipe_list is n, then the returned matrix is nxn that
        contains the logical AND similarity between every pair of recipes divided
        by the length of the maximum ingredients list in each pair.

    """
    one_hot_list = [r.one_hot for r in recipe_list]
    dim = len(one_hot_list)
    distances = np.zeros((dim,dim))
    for i in range(0, len(one_hot_list)):
        for j in range(i+1, len(one_hot_list)):
            current_distance = np.sum(np.logical_and(one_hot_list[i].reshape(-1),one_hot_list[j].reshape(-1)))
            current_distance = current_distance/max(len(one_hot_list[i]), len(one_hot_list[j]))
            distances[i][j] = current_distance
            distances[j][i] = current_distance

    top_sim = np.unravel_index(distances.argmax(), distances.shape)
    #distances = np.reciprocal(distances)
    print("Most similar recipes: ")
    print(top_sim)
    print("Similarity value: %f" % distances[top_sim[0]][top_sim[1]])
    if(print_top_sim):
        print("---- Recipe %i ----"%top_sim[0])
        print(recipe_list[top_sim[0]])
        print("---- Recipe %i: ----"%top_sim[1])
        print(recipe_list[top_sim[1]])
    return distances

def get_cosine_distance(recipe_list, print_top_sim=True):
    """Returns the matrix that contains the cosine similarities between the
    1-hot-vectors of the recipes in recipe_list.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe, that have a "one_hot" attribute.
    print_top_sim : boolean
        Decides whether to print top similar recipe pairs or not.

    Returns
    -------
    matrix
        If the length of recipe_list is n, then the returned matrix is nxn that
        contains the cosine similarity between the 1-hot-vectors of every pair
        of recipes.

    """
    one_hot_list = [r.one_hot for r in recipe_list]
    dim = len(one_hot_list)
    distances = cosine_similarity(one_hot_list)
    np.fill_diagonal(distances, 0)
    top_sim = np.unravel_index(distances.argmax(), distances.shape)
    #distances = np.reciprocal(distances)
    print("Most similar recipes: ")
    print(top_sim)
    print("Similarity value: %f" % distances[top_sim[0]][top_sim[1]])
    if(print_top_sim):
        print("---- Recipe %i ----"%top_sim[0])
        print(recipe_list[top_sim[0]])
        print("---- Recipe %i: ----"%top_sim[1])
        print(recipe_list[top_sim[1]])
    return distances

def get_tfidf_cosine_distance(recipe_list, print_top_sim=True):
    """Returns the matrix that contains the cosine similarities between the
    TF-IDF representations of the recipes in recipe_list. The TF-IDF representations
    are calculated on the concatentation of the ingredients and instructions lists
    after removing the stoopwords and considering the uni-, bi- and trigrams.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    print_top_sim : boolean
        Decides whether to print top similar recipe pairs or not.

    Returns
    -------
    matrix
        If the length of recipe_list is n, then the returned matrix is nxn that
        contains the cosine similarity between the TF-IDF representations
        of every pair of recipes.

    """
    corpus = [f"{x.instructions_text} {x.ingredients_text}" for x in recipe_list]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    distances = cosine_similarity(X)
    np.fill_diagonal(distances, 0)
    top_sim = np.unravel_index(distances.argmax(), distances.shape)
    print("Most similar recipes: ")
    print(top_sim)
    print("Similarity value: %f" % distances[top_sim[0]][top_sim[1]])
    if(print_top_sim):
        print("---- Recipe %i ----"%top_sim[0])
        print(recipe_list[top_sim[0]])
        print("---- Recipe %i: ----"%top_sim[1])
        print(recipe_list[top_sim[1]])
    return distances


def get_ing_pairs(rec_A, rec_B):
    """Returns all pairs of ingredients where ing_A is in rec_A and not in rec_B,
    and ing_B is in rec_B and not in rec_A.

    Parameters
    ----------
    rec_A : Recipe or ReducedRecipe
        Instance of a recipe.
    rec_B : Recipe or ReducedRecipe
        Instance of a recipe.

    Returns
    -------
    list
        List of tuples of ingredient pairs.

    """
    res = []
    set_A = set(rec_A.proccessed_ing_list)
    set_B = set(rec_B.proccessed_ing_list)
    diff_AB = set_A - set_B
    diff_BA = set_B - set_A

    if diff_AB and diff_BA:
        for i in diff_AB:
            for j in diff_BA:
                res.append((i,j))
    return res

def subprocess(params):
    """The subprocess procedure to be called for each child process. Each subprocess
    calculates a version of the ingredients similarity matrix. The summation of
    the results of the subprocesses results in the final ingredients similarity
    matrix.

    Parameters
    ----------
    params : tuple
        Tuple that contains the parameters for the subprocess for calculating the
        ingredients similarity matrix. These parameters are in given in the following
        order:
            -recipe_list : list
                List of objects of type Recipe or ReducedRecipe.
            -start : int
                Start column index for the sub process to calculate the ingredients
                similarity matrix
            -end : int
                End column index for the sub process to calculate the ingredients
                similarity matrix
            -ing_2_id : dict
                Mapping from ingredient text to ingredient id.
            -rec_sim_mat : matrix
                matrix that contains the similarity between the TF-IDF representations
                of every pair of recipes.
            -num_of_ing : int
                Indicates the dimension of the resulting ingredients similarity
                matrix.


    Returns
    -------
    matrix
        Matrix containing the PMI similarity between every pair of recipe as calculated
        from the part of recipe_list passed to the respective subprocess.

    """
    recipe_list = params[0]
    start = params[1]
    end = params[2]
    ing_2_id = params[3]
    rec_sim_mat = params[4]
    num_of_ing = params[5]
    print(f"Working on indices {start} and {end}")
    sim_accum_mat = np.zeros((num_of_ing,num_of_ing))
    for i in range(0, len(recipe_list)):
        for j in range(start, end):
            if j>i:
                current_sim = rec_sim_mat[i][j-start]
                if current_sim>0:
                    for p in get_ing_pairs(recipe_list[i], recipe_list[j]):
                        #print(f"Adding similarity between {p[0]} and {p[1]}")
                        sim_accum_mat[ing_2_id[p[0]]][ing_2_id[p[1]]] += current_sim
                        sim_accum_mat[ing_2_id[p[1]]][ing_2_id[p[0]]] += current_sim
    return sim_accum_mat

def get_ingredient_similarity(recipe_list, sim_matrix, ing_file_name):
    """Wrapper function that calls the subprocess function on 10 cores.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    sim_matrix : matirx
        Matrix that contains the similarity between every pair of recipes.
    ing_file_name : string
        Name of the file that contains the DBpedia ingredients list.

    Returns
    -------
    Tuple
        sim_accum_mat : matrix
            Unnormalized PMI similarities between every pair of recipes.
        id_2_ing : dict
            Mapping from ingredient id to ingredient text.
        ing_2_id : dict
            Mapping from ingredient text to ingredient id.

    """
    db_ingredients = get_filtered_ingredients(ing_file_name)
    ing_2_id = {y:x for (x,y) in enumerate(db_ingredients)}
    id_2_ing = {y:x for (y,x) in enumerate(db_ingredients)}
    dim = len(db_ingredients)
    num_of_cores = 10
    c = len(recipe_list)
    idx = [i for i in range(0,c,c//num_of_cores)]+[c]
    pool = mp.Pool(processes=num_of_cores)
    start = time.time()
    processes = pool.map(subprocess,[(recipe_list, idx[x], idx[x+1], ing_2_id, sim_matrix[:,idx[x]:idx[x+1]], dim) for x in range(num_of_cores)])
    end = time.time()
    print(f"Multiprocessing approach took: {end - start}")
    print("Top 10 similar ingredients:")
    sim_accum_mat = np.zeros((dim,dim))
    for m in processes:
        sim_accum_mat = sim_accum_mat+m
    idx = get_top_n_from_mat(sim_accum_mat,10)
    return sim_accum_mat, ing_2_id, id_2_ing

def calculate_score(recipe_sim_mat, recipe_list):
    """Calculates the normalized PMI score between every pair of ingredients.

    Parameters
    ----------
    recipe_sim_mat : matirx
        Matrix that contains the similarity between every pair of recipes.
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.

    Returns
    -------
    Tuple
        x : matrix
            Normalized PMI similarities between every pair of recipes.
        id_2_ing : dict
            Mapping from ingredient id to ingredient text.
        ing_2_id : dict
            Mapping from ingredient text to ingredient id.

    """
    sim_mat, ing_2_id, id_2_ing = get_ingredient_similarity(recipe_list, recipe_sim_mat, "raw_data/dbpedia_ingredients.txt")
    c_sum = sim_mat.sum(axis=0)
    r_sum = sim_mat.sum(axis=1)
    mat1 = np.outer(c_sum,r_sum)
    mat1[mat1==0] = 1
    x = sim_mat/mat1
    get_top_n_with_lookup(x,50,id_2_ing)

    return x, ing_2_id, id_2_ing
