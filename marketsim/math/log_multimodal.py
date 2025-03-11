import numpy as np
from scipy.special import gammaln
import torch

def logmultinomial(*args):
    """
    Compute the log of the multinomial coefficient.
    log calculations are more numerically stable is the general idea.
    based on: 
    https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Ki5t_rMAAAAJ&citation_for_view=Ki5t_rMAAAAJ:2osOgNQ5qMEC
    """
    args = np.array(args)
    numerator = np.sum(args)
    denominator = np.sum(gammaln(args + 1))
    return gammaln(numerator + 1) - denominator

def logmultinomial_torch(*args):
    """
    Compute the log of the multinomial coefficient.
    log calculations are more numerically stable is the general idea.
    based on: 
    https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Ki5t_rMAAAAJ&citation_for_view=Ki5t_rMAAAAJ:2osOgNQ5qMEC
    """
    args = torch.tensor(args)
    numerator = torch.sum(args)
    denominator = torch.sum(gammaln(args + 1))
    return gammaln(numerator + 1) - denominator






