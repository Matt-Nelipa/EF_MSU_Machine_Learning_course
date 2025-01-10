from typing import List
import numpy as np

def _compute_binary_relevance(
    recommended_items_list: List[int],
    true_items_list: List[int],
) -> List[int]:
  # your code here:
  return [1 if i in true_items_list else 0 for i in recommended_items_list]


def ap_at_k(
    recommended_items_list: List[int],
    true_items_list: List[int],
    k: int
) -> float:
  # your code here:
  if true_items_list is None or len(true_items_list) == 0:
        return 0.0

  if len(recommended_items_list) > k:
      recommended_items_list = recommended_items_list[:k]

  score = 0.0
  num_hits = 0.0

  for i, p in enumerate(recommended_items_list):
      if (i+1 <= len(true_items_list) and i+1 <= k) or (i+1 <= k):
          if p in true_items_list:
              num_hits += 1.0
              print("ok")
              score += num_hits / (i + 1.0)
  print(score/min(len(true_items_list), k))
  return score / min(len(true_items_list), k)



def map_at_k(
    recommended_items_lists: List[List[int]],
    true_items_lists: List[List[int]],
    k: int,
) -> float:
  """
  Computes ap@k for all buyers
  """
  assert len(recommended_items_lists) == len(true_items_lists), \
  'len(true_items_list) != len(recommended_items_list)'

  return np.mean([ap_at_k(a, p, k) for a, p in zip(recommended_items_lists, true_items_lists)])