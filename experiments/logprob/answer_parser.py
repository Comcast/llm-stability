import pandas as pd
import re


def parse_paren_answers_with_weights(answers: list, 
                                    row:pd.Series) -> (str, None):
    """
    Returns parsed answer from list pulled from 'response' index of Series. 
    Applies uniqueness presupposed breadth first search a salience ranking (last sentence, antepenultimate + penultimate + last, entire answer) and then through a backoff sequence per salience level as follows:
    exact match substring match, case insensitive substring match and finally
    a case insensitive search with `()` replaced with word boundaries. Has
    ability to use LLM answer parsing but not implemented. 
    Args:
        answers (list): Answers in the form [(A), (B), (C), (D)]
        row (pd.Series): Row form dataframe being evaluated
    Returns: 
        str: Answer if found in original form or None if no answer found
        float: Weight
    Raises:
        LookupError if there is not a unique solution 

    Examples Blown UP:
    "Since all the statements (A), (B), (C), and (D) are true, none of them is false. Therefore, there seems to be a mistake in the problem statement or the options provided. However, based on the given options and the analysis, none of the statements is false."

    "So, the answer is:
(A) 0. (B) 1. (C) 2. (D) 3.

The correct answer is:
None of the given options are correct. The dimension that cannot be the dimension of ( V cap W ) is 4."
    """
    matches = {}
    intent_backoff = ['answer is',
                     'exact_word', 
                     'case_insensitive', 'sub_string'] #llm unused
    #sentences = row['response'].split('\n')
    sents_tok_probs = [[]]
    for token_prob in row['logprobs']:
        if '\n' in token_prob['token']:
            sents_tok_probs[-1].append(token_prob)
            sents_tok_probs.append([])
        else:
            sents_tok_probs[-1].append(token_prob)
    for start in [-1, -3, -len(sents_tok_probs)]: #structural salience
        tok_ends = []
        buffer_len = 0
        text = ''
        salience_domain_tok_probs = []
        for sent in sents_tok_probs[start:]:
            for tok_prob in sent:
                text += tok_prob['token']
                buffer_len += len(tok_prob['token'])
                tok_ends.append(buffer_len)
                salience_domain_tok_probs.append(tok_prob)
        for match_type in intent_backoff:
            for i, answer in enumerate(answers):
                answer_re = None
                if match_type == 'answer is':
                    escaped_parens = \
                        answer.replace('(', r'\(').replace(')', r'\)')
                    answer_re = fr'(T|t)he answer is {escaped_parens}'
                if match_type == 'exact_word':
                    answer_re =\
                            answer.replace('(', r'\(').replace(')', r'\)')
                if match_type == 'case_insensitive':
                    answer_re =\
                            answer.replace(')', r'\)').replace('(', r'(?i)\(')
                if match_type == 'sub_string':
                    answer_re =\
                        answer.replace(')', r'\b').replace('(', r'(?i)\b')
                m = re.search(answer_re, text)
                if m:
                    start_t = next(i for i, offset in \
                        enumerate(tok_ends) if offset >= m.start())
                    end_t = next(i for i, offset in \
                        enumerate(tok_ends) if offset >= m.end())
                    logprobs = [e['logprob'] for e in \
                                    salience_domain_tok_probs[start_t: end_t + 1]]
                    matches[answer] = sum(logprobs) / len(logprobs)

            if len(matches.keys()) == 1:
                return matches
            if len(matches) > 1:
                raise LookupError(f"Blown UP: {text}")