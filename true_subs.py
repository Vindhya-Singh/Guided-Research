from sub_find import *
from helpers import find_combinations
import operator
import numpy as np

def get_potential_subs(recipe_list, sim_mat, n, results_file, true_pairs=[], false_pairs=[]):
    """Lets the user annotate true substitutions from recipes. Writes the
    substitution pair to a file a long with the recipes where the substitution
    was found as an evidence that this substitution is valid.

    Parameters
    ----------
    recipe_list : list
        List of objects of type Recipe or ReducedRecipe.
    sim_mat : matrix
        Matrix containing the recipe similarity.
    n : int
        Number of top similar recipes to consider when annotating true substitutions.
    results_file : string
        Path to the file to write the results to, created if not existing.
    true_pairs : list
        List of true substitution pairs found through previous calls of the same
        function. Used to speed up the annotation by not asking the user to annotate
        a given pair if it is in this list.
    false_pairs : type
        List of false substitution pairs found through previous calls of the same
        function. Used to speed up the annotation by not asking the user to annotate
        a given pair if it is in this list.

    Returns
    -------
    Tuple
        true_pairs: List of tuples where each tuple is a true substitution pair.
        false_pairs:  List of tuples where each tuple is a false substitution pair.

    """
    f = open(results_file,"w")
    idx = get_top_n_with_lookup(sim_mat, n ,recipe_list)
    enough = 20
    found_pairs = 0
    for i in idx:
        rec_A = set(recipe_list[i[0]].proccessed_ing_list)
        rec_B = set(recipe_list[i[1]].proccessed_ing_list)
        diff_AB = list(rec_A - rec_B)
        diff_BA = list(rec_B - rec_A)

        subs = find_combinations(diff_AB, diff_BA)
        if subs:
            f.write(f"{recipe_list[i[0]]}\n")
            f.write('AND\n')
            f.write(f"{recipe_list[i[1]]}\n")

            print(f"From {recipe_list[i[0]]} and {recipe_list[i[1]]} we can substitute:\n")
            for s in subs:

                print(f"{s[0]} <---> {s[1]}")
                sorted_pair = sorted([s[0], s[1]])

                if (not (sorted_pair[0], sorted_pair[1]) in true_pairs) and (not (sorted_pair[0], sorted_pair[1]) in false_pairs) and found_pairs<enough:
                    add_pair = input(f"Add \033[4m{s[0]}\033[0m and \033[4m{s[1]}\033[0m to true pairs:")
                    if add_pair == "y":
                        true_pairs.append((sorted_pair[0], sorted_pair[1]))
                        found_pairs += 1
                        print(f"Added {found_pairs} pairs")
                        f.write(f"{sorted_pair[0]} <---> {sorted_pair[1]}\n")
                    else:
                        false_pairs.append((sorted_pair[0], sorted_pair[1]))
                else:
                    print(f"Already added {s[0]} <---> {s[1]}")

        print("\n")
    f.close()
    return true_pairs, false_pairs


def get_rank(ing_A, ing_B, ing_sim_mat, ing_2_id):
    """Gets the rank of ing_B in the substitution candidates of ing_A. Both ingredients
    have to be in the corpus, meaning they have to be found in the list of
    ingredients extracted from DBpedia.

    Parameters
    ----------
    ing_A : string
        String representation of an ingredient.
    ing_B : string
        String representation of an ingredient.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.

    Returns
    -------
    int
        Rank of ing_B in the substitution candidates of ing_A.

    """
    id_A = ing_2_id[ing_A]
    id_B = ing_2_id[ing_B]
    scores = ing_sim_mat[id_A]
    temp = scores.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))
    return ranks[id_B]+1

def get_top_ranked(ing_A, n, ing_sim_mat, id_2_ing, ing_2_id):
    """Get the top n substitution candiates for ingredient ing_A.

    Parameters
    ----------
    ing_A : string
        String representation of an ingredient.
    n : int
        Number of top substitution candiates to return.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    id_2_ing : dict
        Mapping from ingredient id to ingredient text.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.

    Returns
    -------
    list
        List of substitution candidates sorted according to their rank.

    """
    id_A = ing_2_id[ing_A]
    scores = ing_sim_mat[id_A]
    temp = scores.argsort()[::-1][:n]

    for i in temp:
        print(id_2_ing[i])
    return [id_2_ing[i] for i in temp]

def get_results(pairs, ing_sim_mat, ing_2_id, file_name, method="avg"):
    """Calculates the average Mean Reciprocal Rank for a given list of true
    substitution pairs. By default, the average between the rank for substituting
    ing_A with ing_B and the rank for substituting ing_B with ing_A.

    Parameters
    ----------
    pairs : list
        List of tuples of the shape (ing_A, ing_B) where each tuple is a true
        substitution pair.
    ing_sim_mat : matrix
        Matrix containing the ingredient similarity computed via PMI.
    ing_2_id : dict
        Mapping from ingredient text to ingredient id.
    file_name : string
        Name of the file to save the evaluation results to. The file contains all
        ranks of the substitution pairs and the average MRR in the last line.
    method : string
        If "avg" then the average of the ranks of substituting ing_A with ing_B,
        and ing_B with ing_A. Otherwise, only the direction of the substitution that yielded
        the smaller rank is considered.

    Returns
    -------
    None
        Writes results to file.

    """
    result = []
    f = open(file_name, "w")
    for p in pairs:
        if method == "avg":
            score = np.mean([get_rank(p[0], p[1],  ing_sim_mat, ing_2_id), get_rank(p[1], p[0],  ing_sim_mat, ing_2_id)])
        else:
            score = min(get_rank(p[0], p[1],  ing_sim_mat, ing_2_id), get_rank(p[1], p[0],  ing_sim_mat, ing_2_id))
        s = f'Score between {p[1]} and {p[0]} is {score}'
        print(s)
        f.write(s+"\n")
        result.append(1/score)
    print(f'Final score: {np.mean(result)}')
    f.write(f'Final score: {np.mean(result)}')
