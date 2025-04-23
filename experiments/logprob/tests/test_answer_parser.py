import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.getcwd()))
import answer_parser
#import parse_paren_answers_with_weights


def test_parse_paren_answers_with_weights():
    answers = ['(A)', '(B)', '(C)', '(D)']
    row = pd.Series({'logprobs':[
                {'token': ' (', 'logprob':-.5},
                {'token': 'A', 'logprob':-.1},
                {'token': ') ', 'logprob':-.5}
                ]})
    answer = answer_parser.parse_paren_answers_with_weights(answers, row)
    assert len(answer.keys()) == 1
    assert next(iter(answer.values())) == -0.3666666666666667
    row = pd.Series({'logprobs':[
                {'token': 'The', 'logprob':-.1},
                {'token': ' answer', 'logprob':-.1},
                {'token': ' is', 'logprob':-.1},
                {'token': ' (', 'logprob':-.5},
                {'token': 'A', 'logprob':-.1},
                {'token': ') ', 'logprob':-.5}
                ]})
    answer = answer_parser.parse_paren_answers_with_weights(answers, row)
    assert len(answer.keys()) == 1
    assert next(iter(answer.values())) == -0.2333333333333333

BUG1 = pd.Series({'logprobs':[{'token': 'Since', 'logprob': -0.65141225}, {'token': ' x', 'logprob': -0.16923131}, {'token': ' must', 'logprob': -0.22161998}, {'token': ' be', 'logprob': -0.014135539}, {'token': ' greater', 'logprob': -0.5395493}, {'token': ' than', 'logprob': -6.289474e-05}, {'token': ' ', 'logprob': -0.0072783954}, {'token': '10', 'logprob': -0.028866405}, {'token': '/', 'logprob': -0.00018732868}, {'token': '3', 'logprob': -3.333223e-05}, {'token': ',', 'logprob': -1.0474288}, {'token': ' the', 'logprob': -0.41816568}, {'token': ' only', 'logprob': -1.1427946}, {'token': ' possible', 'logprob': -0.44171816}, {'token': ' values', 'logprob': -0.23664701}, {'token': ' for', 'logprob': -0.11511807}, {'token': ' x', 'logprob': -0.009104819}, {'token': ' are', 'logprob': -0.2784368}, {'token': ' between', 'logprob': -0.8357571}, {'token': ' ', 'logprob': -0.012123318}, {'token': '10', 'logprob': -0.1312384}, {'token': '/', 'logprob': -0.00034106473}, {'token': '3', 'logprob': -9.9490266e-05}, {'token': ' and', 'logprob': -0.006875568}, {'token': ' ', 'logprob': -0.00020663968}, {'token': '10', 'logprob': -0.21288973}, {'token': '.', 'logprob': -0.7888412}, {'token': ' This', 'logprob': -0.5506185}, {'token': ' means', 'logprob': -0.44867226}, {'token': ' that', 'logprob': -0.17618583}, {'token': ' the', 'logprob': -0.51308036}, {'token': ' shortest', 'logprob': -1.4323639}, {'token': ' segment', 'logprob': -0.0026892058}, {'token': ' must', 'logprob': -0.16452357}, {'token': ' be', 'logprob': -0.22127083}, {'token': ' between', 'logprob': -0.9608853}, {'token': ' ', 'logprob': -0.030007223}, {'token': '10', 'logprob': -0.14173393}, {'token': '/', 'logprob': -6.6232446e-05}, {'token': '3', 'logprob': -3.619312e-05}, {'token': ' and', 'logprob': -0.019115614}, {'token': ' ', 'logprob': -3.202099e-05}, {'token': '10', 'logprob': -0.003107954}, {'token': ' units', 'logprob': -0.6099187}, {'token': ' long', 'logprob': -1.4203862}, {'token': '.\n\n', 'logprob': -0.40099224}, {'token': 'The', 'logprob': -0.34157464}, {'token': ' probability', 'logprob': -0.07032596}, {'token': ' that', 'logprob': -0.49063376}, {'token': ' a', 'logprob': -0.5853669}, {'token': ' randomly', 'logprob': -0.123128936}, {'token': ' chosen', 'logprob': -0.021422544}, {'token': ' point', 'logprob': -0.064873956}, {'token': ' A', 'logprob': -0.93143857}, {'token': ' will', 'logprob': -0.8963244}, {'token': ' result', 'logprob': -1.3052866}, {'token': ' in', 'logprob': -1.2829201e-05}, {'token': ' a', 'logprob': -0.3483443}, {'token': ' segment', 'logprob': -0.3004961}, {'token': ' length', 'logprob': -1.0031247}, {'token': ' between', 'logprob': -0.40982142}, {'token': ' ', 'logprob': -0.00056625705}, {'token': '10', 'logprob': -5.0378356e-05}, {'token': '/', 'logprob': -8.208653e-05}, {'token': '3', 'logprob': -1.9385403e-05}, {'token': ' and', 'logprob': -0.0005544632}, {'token': ' ', 'logprob': -7.58424e-06}, {'token': '10', 'logprob': -6.253713e-05}, {'token': ' is', 'logprob': -0.039299272}, {'token': ' ', 'logprob': -1.1700448}, {'token': '7', 'logprob': -0.27452984}, {'token': '/', 'logprob': -0.002195814}, {'token': '10', 'logprob': -0.0008323783}, {'token': ',', 'logprob': -0.98488677}, {'token': ' or', 'logprob': -0.48359603}, {'token': ' ', 'logprob': -6.81397e-05}, {'token': '70', 'logprob': -0.0014479756}, {'token': '%.', 'logprob': -0.68826306}, {'token': ' Therefore', 'logprob': -0.8351104}, {'token': ',', 'logprob': -0.00348912}, {'token': ' the', 'logprob': -0.0077408976}, {'token': ' probability', 'logprob': -0.04535486}, {'token': ' that', 'logprob': -0.010493774}, {'token': ' the', 'logprob': -0.11272316}, {'token': ' three', 'logprob': -0.007483815}, {'token': ' smaller', 'logprob': -0.11955016}, {'token': ' segments', 'logprob': -6.4325184e-05}, {'token': ' could', 'logprob': -0.11759061}, {'token': ' form', 'logprob': -2.546479e-05}, {'token': ' the', 'logprob': -0.29152972}, {'token': ' sides', 'logprob': -4.572941e-05}, {'token': ' of', 'logprob': -3.7697225e-06}, {'token': ' a', 'logprob': -0.00018697108}, {'token': ' triangle', 'logprob': -8.299462e-06}, {'token': ' is', 'logprob': -0.0008966933}, {'token': ' ', 'logprob': -0.05308374}, {'token': '70', 'logprob': -8.566264e-05}, {'token': '%.\n\n', 'logprob': -0.27861685}, {'token': 'Therefore', 'logprob': -0.8379722}, {'token': ',', 'logprob': -0.0068604127}, {'token': ' the', 'logprob': -0.0115790535}, {'token': ' correct', 'logprob': -0.56308806}, {'token': ' answer', 'logprob': -0.039896257}, {'token': ' is', 'logprob': -0.005778408}, {'token': ' not', 'logprob': -0.06073029}, {'token': ' listed', 'logprob': -1.2168511}, {'token': ' in', 'logprob': -0.87946403}, {'token': ' the', 'logprob': -0.0010274507}, {'token': ' options', 'logprob': -0.39226788}, {'token': ' provided', 'logprob': -0.34338883}, {'token': '.', 'logprob': -0.04789183}]})

    
def test_bug1():
    answers = ['(A)']
    answer = answer_parser.parse_paren_answers_with_weights(answers, BUG1)
    assert answer == {'(A)': -0.5853669}

